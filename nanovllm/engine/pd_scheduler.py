"""
PD分离调度器 (Prefill-Decode Disaggregated Scheduler)

在单GPU上实现Prefill和Decode的分离调度，主要特性：
1. Prefill和Decode批次完全分开
2. 支持Chunked Prefill（分块预填充）
3. 可配置的调度策略（decode优先或prefill优先）
"""
from collections import deque
from enum import Enum, auto

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class SchedulePhase(Enum):
    """当前调度阶段"""
    PREFILL = auto()
    DECODE = auto()
    IDLE = auto()


class PDScheduler:
    """
    PD分离调度器
    
    将Prefill和Decode阶段完全分开调度：
    - Prefill阶段：处理新请求的prompt
    - Decode阶段：生成token
    
    支持Chunked Prefill，将长prompt分块处理
    """

    def __init__(self, config: Config):
        self.config = config
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        
        # PD分离特定配置
        self.prefill_chunk_size = config.prefill_chunk_size
        self.max_prefill_batch_tokens = config.max_prefill_batch_tokens
        self.max_decode_batch_size = config.max_decode_batch_size
        self.min_decode_batch_size = config.min_decode_batch_size
        self.schedule_policy = config.pd_schedule_policy
        
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        
        # 分离的队列
        self.waiting: deque[Sequence] = deque()      # 等待prefill的序列
        self.prefilling: deque[Sequence] = deque()   # 正在进行chunked prefill的序列
        self.decoding: deque[Sequence] = deque()     # 等待/正在decode的序列
        
        # 统计信息
        self.num_prefill_batches = 0
        self.num_decode_batches = 0

    def is_finished(self):
        return not self.waiting and not self.prefilling and not self.decoding

    def add(self, seq: Sequence):
        """添加新序列到等待队列"""
        seq.status = SequenceStatus.WAITING
        self.waiting.append(seq)

    def _has_prefill_work(self) -> bool:
        """检查是否有prefill工作"""
        return bool(self.waiting) or bool(self.prefilling)

    def _has_decode_work(self) -> bool:
        """检查是否有decode工作"""
        return bool(self.decoding)

    def _schedule_prefill(self) -> tuple[list[Sequence], dict]:
        """
        调度Prefill批次
        
        支持Chunked Prefill：
        - 对于长prompt，分多次prefill
        - 每次prefill处理prefill_chunk_size个token
        
        Returns:
            - scheduled_seqs: 调度的序列列表
            - prefill_info: prefill信息，包含每个序列的token范围
        """
        scheduled_seqs = []
        prefill_info = {
            "chunk_starts": [],  # 每个序列的起始位置
            "chunk_ends": [],    # 每个序列的结束位置
        }
        num_batched_tokens = 0
        
        # 首先处理正在进行chunked prefill的序列
        remaining_prefilling = deque()
        while self.prefilling:
            seq = self.prefilling.popleft()
            
            # 计算本次chunk的大小
            remaining = seq.remaining_prefill_tokens
            chunk_size = min(remaining, self.prefill_chunk_size)
            
            if num_batched_tokens + chunk_size > self.max_prefill_batch_tokens:
                remaining_prefilling.append(seq)
                continue
            
            # 确保有足够的block
            needed_blocks = (seq.num_prefilled_tokens + chunk_size + self.block_manager.block_size - 1) // self.block_manager.block_size
            current_blocks = len(seq.block_table)
            if needed_blocks > current_blocks:
                if not self._try_allocate_blocks(seq, needed_blocks - current_blocks):
                    remaining_prefilling.append(seq)
                    continue
            
            chunk_start = seq.num_prefilled_tokens
            chunk_end = seq.num_prefilled_tokens + chunk_size
            
            prefill_info["chunk_starts"].append(chunk_start)
            prefill_info["chunk_ends"].append(chunk_end)
            
            num_batched_tokens += chunk_size
            scheduled_seqs.append(seq)
        
        self.prefilling = remaining_prefilling
        
        # 然后从waiting队列调度新序列
        while self.waiting and num_batched_tokens < self.max_prefill_batch_tokens:
            seq = self.waiting[0]
            
            # 尝试分配blocks（这可能会设置num_cached_tokens如果有prefix caching）
            if not self.block_manager.can_allocate(seq):
                break
            
            self.waiting.popleft()
            self.block_manager.allocate(seq)
            
            # 考虑prefix caching：从num_cached_tokens开始
            seq.num_prefilled_tokens = seq.num_cached_tokens
            
            # 即使所有token都被cache命中，仍需要至少处理最后一个token来生成first decode token
            # 如果全部被cache，至少处理最后一个token
            if seq.num_prefilled_tokens >= seq.num_prompt_tokens:
                seq.num_prefilled_tokens = seq.num_prompt_tokens - 1
            
            seq.status = SequenceStatus.PREFILLING
            
            # 计算首次chunk大小（从cached位置开始）
            remaining = seq.num_prompt_tokens - seq.num_prefilled_tokens
            chunk_size = min(remaining, self.prefill_chunk_size)
            # 确保至少处理1个token
            chunk_size = max(chunk_size, 1)
            
            if num_batched_tokens + chunk_size > self.max_prefill_batch_tokens:
                # 如果放不下，需要回退
                self.block_manager.deallocate(seq)
                seq.status = SequenceStatus.WAITING
                seq.num_prefilled_tokens = 0
                self.waiting.appendleft(seq)
                break
            
            chunk_start = seq.num_prefilled_tokens
            chunk_end = seq.num_prefilled_tokens + chunk_size
            
            prefill_info["chunk_starts"].append(chunk_start)
            prefill_info["chunk_ends"].append(chunk_end)
            
            num_batched_tokens += chunk_size
            scheduled_seqs.append(seq)
        
        if scheduled_seqs:
            self.num_prefill_batches += 1
        
        return scheduled_seqs, prefill_info

    def _schedule_decode(self) -> list[Sequence]:
        """
        调度Decode批次
        
        Returns:
            调度的序列列表
        """
        scheduled_seqs = []
        remaining = deque()
        
        while self.decoding and len(scheduled_seqs) < self.max_decode_batch_size:
            seq = self.decoding.popleft()
            
            # 检查是否可以append（可能需要新block）
            while not self.block_manager.can_append(seq):
                # 尝试抢占
                if self.decoding:
                    self._preempt(self.decoding.pop())
                else:
                    # 无法继续，将序列放回
                    self._preempt(seq)
                    seq = None
                    break
            
            if seq is not None:
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        
        # 将剩余的序列放回队列
        while self.decoding:
            remaining.append(self.decoding.popleft())
        self.decoding = remaining
        
        # 将调度的序列放回队列头部（保持顺序）
        for seq in reversed(scheduled_seqs):
            self.decoding.appendleft(seq)
        
        if scheduled_seqs:
            self.num_decode_batches += 1
        
        return scheduled_seqs

    def _try_allocate_blocks(self, seq: Sequence, num_blocks: int) -> bool:
        """尝试为序列分配更多blocks"""
        if len(self.block_manager.free_block_ids) < num_blocks:
            return False
        
        for _ in range(num_blocks):
            block_id = self.block_manager.free_block_ids[0]
            self.block_manager._allocate_block(block_id)
            seq.block_table.append(block_id)
        return True

    def _preempt(self, seq: Sequence):
        """抢占序列"""
        if seq.is_prefill_complete:
            # 如果prefill已完成，回退到prefilling队列重新开始
            seq.status = SequenceStatus.PREFILLING
            seq.num_prefilled_tokens = 0
            self.block_manager.deallocate(seq)
            self.prefilling.appendleft(seq)
        else:
            # 否则回退到waiting队列
            seq.status = SequenceStatus.WAITING
            seq.num_prefilled_tokens = 0
            self.block_manager.deallocate(seq)
            self.waiting.appendleft(seq)

    def schedule(self) -> tuple[list[Sequence], bool, dict | None]:
        """
        主调度函数
        
        根据调度策略选择执行Prefill或Decode
        
        Returns:
            - scheduled_seqs: 调度的序列列表
            - is_prefill: 是否是prefill阶段
            - prefill_info: prefill信息（仅prefill时有效）
        """
        if self.schedule_policy == "decode_first":
            # Decode优先：如果有足够的decode工作，先做decode
            # 使用min_decode_batch_size避免过小的decode batch
            has_enough_decode = len(self.decoding) >= self.min_decode_batch_size
            has_prefill = self._has_prefill_work()
            
            # 只有当decode队列足够大，或者没有prefill工作时才做decode
            if self._has_decode_work() and (has_enough_decode or not has_prefill):
                seqs = self._schedule_decode()
                if seqs:
                    return seqs, False, None
            
            if has_prefill:
                seqs, prefill_info = self._schedule_prefill()
                if seqs:
                    return seqs, True, prefill_info
        else:
            # Prefill优先：如果有prefill工作，先做prefill
            if self._has_prefill_work():
                seqs, prefill_info = self._schedule_prefill()
                if seqs:
                    return seqs, True, prefill_info
            
            if self._has_decode_work():
                seqs = self._schedule_decode()
                if seqs:
                    return seqs, False, None
        
        return [], False, None

    def postprocess_prefill(self, seqs: list[Sequence], prefill_info: dict, token_ids: list[int] | None):
        """
        处理Prefill后的状态更新
        
        Args:
            seqs: prefill的序列
            prefill_info: prefill信息
            token_ids: 生成的token（只包含完成prefill的序列的token）
        """
        # token_ids只包含完成prefill的序列的采样结果，按顺序对应
        token_idx = 0
        
        for i, seq in enumerate(seqs):
            chunk_end = prefill_info["chunk_ends"][i]
            seq.num_prefilled_tokens = chunk_end
            
            if seq.is_prefill_complete:
                # Prefill完成，转入decode阶段
                seq.status = SequenceStatus.DECODING
                seq.prefill_complete = True
                self.decoding.append(seq)
                
                # 添加第一个生成的token
                if token_ids is not None and token_idx < len(token_ids):
                    seq.append_token(token_ids[token_idx])
                    token_idx += 1
            else:
                # 还需要继续prefill
                self.prefilling.append(seq)

    def postprocess_decode(self, seqs: list[Sequence], token_ids: list[int]):
        """
        处理Decode后的状态更新
        
        Args:
            seqs: decode的序列
            token_ids: 生成的token
        """
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            
            # 检查是否完成
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.decoding.remove(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool, prefill_info: dict | None = None):
        """
        统一的后处理函数（兼容原有接口）
        """
        if is_prefill:
            self.postprocess_prefill(seqs, prefill_info, token_ids)
        else:
            self.postprocess_decode(seqs, token_ids)

