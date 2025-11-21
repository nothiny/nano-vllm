from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:
    """
    调度器：负责管理请求的调度和资源分配
    
    核心职责：
    1. 维护请求队列（waiting 和 running）
    2. 决定每个 iteration 执行哪些请求（Prefill 或 Decode）
    3. 处理资源不足时的抢占 (Preemption)
    4. 后处理生成的 token 并检查完成条件
    
    调度策略：
    - 优先处理 Prefill（新请求）
    - 如果没有新请求，处理 Decode（继续生成）
    - 资源不足时，抢占优先级最低的请求（LIFO 策略）
    """

    def __init__(self, config: Config):
        """
        初始化调度器
        
        Args:
            config: 全局配置对象
        """
        # 批次限制
        self.max_num_seqs = config.max_num_seqs  # 最大并发序列数
        self.max_num_batched_tokens = config.max_num_batched_tokens  # 单次 iteration 最大 token 数
        self.eos = config.eos  # EOS token ID
        
        # 块管理器：负责 KV Cache 的分配和回收
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        
        # 请求队列
        self.waiting: deque[Sequence] = deque()  # 等待队列：新提交或被抢占的请求
        self.running: deque[Sequence] = deque()  # 运行队列：正在生成的请求

    def is_finished(self):
        """检查是否所有请求都已完成"""
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        """添加新请求到等待队列"""
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """
        核心调度逻辑：决定本次 iteration 执行哪些请求
        
        Returns:
            (scheduled_seqs, is_prefill):
            - scheduled_seqs: 本次要执行的序列列表
            - is_prefill: True 表示 Prefill 阶段，False 表示 Decode 阶段
        
        调度策略：
        1. 优先处理 Prefill（等待队列中的请求）
        2. 如果等待队列为空，处理 Decode（运行队列中的请求）
        3. 受限于 max_num_seqs 和 max_num_batched_tokens
        """
        # ===== Phase 1: Prefill 调度 =====
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            
            # 检查资源约束：
            # 1. Token 数量不能超过 max_num_batched_tokens
            # 2. KV Cache 必须有足够的空闲块
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            
            num_seqs += 1
            # 为序列分配 KV Cache 块
            self.block_manager.allocate(seq)
            # 只计算未缓存的 token 数量（前缀缓存优化）
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            
            # 更新状态并移动到运行队列
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        
        # 如果成功调度了 Prefill 请求，直接返回
        if scheduled_seqs:
            return scheduled_seqs, True

        # ===== Phase 2: Decode 调度 =====
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            
            # 检查是否有足够的 KV Cache 空间追加新 token
            while not self.block_manager.can_append(seq):
                # 资源不足：需要抢占（释放）其他请求
                if self.running:
                    # 抢占优先级最低的请求（队列末尾，LIFO 策略）
                    self.preempt(self.running.pop())
                else:
                    # 如果只剩当前请求，抢占自己（下次再试）
                    self.preempt(seq)
                    break
            else:
                # 资源充足：调度该请求
                num_seqs += 1
                # 预分配下一个 token 的 KV Cache 空间
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        
        assert scheduled_seqs
        # 将调度的序列放回队列头部（保持 FIFO 顺序）
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        """
        抢占（Preempt）：暂停一个请求，释放其 KV Cache
        
        被抢占的请求会被放回等待队列，之后重新调度时需要重新分配 KV Cache。
        
        Args:
            seq: 要抢占的序列
        """
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)  # 释放 KV Cache 块
        self.waiting.appendleft(seq)  # 放回等待队列头部（高优先级）

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        """
        后处理：更新序列状态，检查是否完成
        
        在每次生成 token 后调用，负责：
        1. 将新 token 添加到序列
        2. 检查完成条件（EOS 或达到 max_tokens）
        3. 清理已完成的序列
        
        Args:
            seqs: 本次执行的序列列表
            token_ids: 对应生成的 token IDs
        """
        for seq, token_id in zip(seqs, token_ids):
            # 添加新生成的 token
            seq.append_token(token_id)
            
            # 检查完成条件：
            # 1. 遇到 EOS token（且未设置 ignore_eos）
            # 2. 达到最大生成长度
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)  # 释放 KV Cache
                self.running.remove(seq)  # 从运行队列移除
