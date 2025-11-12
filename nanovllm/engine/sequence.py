"""序列（请求）的数据结构和状态管理"""
from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    """
    序列（请求）的生命周期状态
    
    一个请求的典型状态转换：
    WAITING → RUNNING → FINISHED
    """
    WAITING = auto()   # 等待中：请求已提交，等待被调度执行
    RUNNING = auto()   # 运行中：正在生成 token
    FINISHED = auto()  # 已完成：生成结束（达到 max_tokens 或遇到 EOS）
    
    # auto() 的作用：
    # - 自动分配枚举值（1, 2, 3, ...）
    # - 避免手动维护数字，防止重复
    # - 等价于：WAITING=1, RUNNING=2, FINISHED=3


class Sequence:
    """
    序列（Sequence）：代表一个推理请求
    
    核心职责：
    1. 管理 token 序列（输入 + 生成的输出）
    2. 跟踪状态（WAITING/RUNNING/FINISHED）
    3. 管理 KV 缓存的块分配（block_table）
    4. 区分 prompt tokens 和 completion tokens
    """
    
    # 类变量：所有序列共享
    block_size = 256  # KV 缓存块大小，与 Config.kvcache_block_size 一致
    counter = count()  # 全局计数器，为每个序列分配唯一 ID（生成 0, 1, 2, ...）

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        """
        初始化一个新的推理请求
        
        Args:
            token_ids: 输入的 prompt token IDs
            sampling_params: 采样参数（温度、最大长度等）
        """
        # ===== 基本属性 =====
        self.seq_id = next(Sequence.counter)  # 唯一序列ID
        self.status = SequenceStatus.WAITING  # 初始状态：等待调度
        
        # ===== Token 管理 =====
        self.token_ids = copy(token_ids)  # 复制输入 tokens（避免外部修改）
        self.last_token = token_ids[-1]   # 最后一个 token（用于快速访问）
        self.num_tokens = len(self.token_ids)  # 当前总 token 数
        self.num_prompt_tokens = len(token_ids)  # 输入 prompt 的长度（固定）
        
        # ===== KV 缓存管理 =====
        self.num_cached_tokens = 0  # 已缓存的 token 数量（用于前缀缓存优化）
        self.block_table = []  # 块表：存储分配的 KV 缓存块 ID
                               # 例如 [5, 12, 23] 表示使用了第 5, 12, 23 号块
        
        # ===== 采样参数（从 sampling_params 复制） =====
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    # ===== 基本接口（让 Sequence 行为像列表） =====
    
    def __len__(self):
        """返回序列长度"""
        return self.num_tokens

    def __getitem__(self, key):
        """支持索引访问：seq[0], seq[1:3] 等"""
        return self.token_ids[key]

    # ===== 状态查询 =====
    
    @property
    def is_finished(self):
        """是否已完成生成"""
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        """已生成的 token 数量（输出长度）"""
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        """获取输入 prompt 的 token IDs"""
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        """获取生成的输出 token IDs"""
        return self.token_ids[self.num_prompt_tokens:]

    # ===== KV 缓存块管理 =====
    
    @property
    def num_cached_blocks(self):
        """已缓存的完整块数量"""
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        """
        当前序列需要的总块数
        
        计算方式：向上取整
        例如：257 tokens / 256 = 2 块（第1块满，第2块只有1个token）
        """
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        """
        最后一个块中的 token 数量
        
        例如：270 tokens，块大小 256
        - 总共需要 2 块
        - 最后一块有 270 - 256 = 14 个 tokens
        """
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        """
        获取第 i 个块的 token IDs
        
        Args:
            i: 块索引（从 0 开始）
        Returns:
            该块对应的 token IDs（最多 block_size 个）
        """
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    # ===== Token 操作 =====
    
    def append_token(self, token_id: int):
        """
        添加新生成的 token
        
        在生成过程中，每次采样产生新 token 后调用此方法
        """
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    # ===== 序列化支持（用于进程间通信或持久化） =====
    
    def __getstate__(self):
        """
        序列化状态（pickle 使用）
        
        优化：
        - Prefill 阶段（生成前）：保存完整 token_ids
        - Decode 阶段（生成中）：只保存 last_token（节省内存）
        """
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    def __setstate__(self, state):
        """
        反序列化状态（pickle 使用）
        
        根据保存时的状态恢复对象
        """
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            # Prefill 阶段：恢复完整 token_ids
            self.token_ids = state[-1]
        else:
            # Decode 阶段：只恢复 last_token
            self.last_token = state[-1]
