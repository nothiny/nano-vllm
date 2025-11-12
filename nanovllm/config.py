"""LLM 引擎配置"""
import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    """
    LLM 推理引擎的全局配置
    
    主要控制：批处理策略、内存分配、并行化、KV缓存管理
    """
    
    # ===== 基础配置 =====
    model: str  # 模型路径，如 "~/huggingface/Qwen3-0.6B/"
    
    # ===== 批处理配置 =====
    max_num_batched_tokens: int = 16384  
    # 单次前向传播最多处理的 token 数量
    # - 这是批处理的核心限制：sum(所有序列的token数) ≤ 此值
    # - 越大：吞吐量越高，但显存占用越多
    # - 例如：可以是 16个1024长度的序列，或 32个512长度的序列
    
    max_num_seqs: int = 512
    # 同时处理的最大序列（请求）数量
    # - 限制并发请求数，防止调度器过载
    # - 配合 max_num_batched_tokens 一起限制批处理大小
    
    max_model_len: int = 4096
    # 单个序列的最大长度（输入+输出）
    # - 会和模型的 max_position_embeddings 取较小值
    # - 超过此长度的请求会被拒绝或截断
    
    # ===== 内存管理 =====
    gpu_memory_utilization: float = 0.9
    # GPU 显存利用率（0.0-1.0）
    # - 0.9 表示使用 90% 的可用显存
    # - 剩余 10% 留给 PyTorch 的内部操作和碎片
    # - 显存紧张时可降低到 0.8；显存充足可提高到 0.95
    
    # ===== 并行化配置 =====
    tensor_parallel_size: int = 1
    # 张量并行的 GPU 数量
    # - 1: 单GPU推理
    # - 2/4/8: 将模型切分到多个GPU上（需要多GPU支持）
    # - 用于加载超大模型（单GPU放不下）
    
    # ===== 性能优化 =====
    enforce_eager: bool = False
    # 是否强制使用即时执行模式（禁用 CUDA graph）
    # - False: 启用 CUDA graph，性能更好（推荐用于生产）
    # - True: 即时执行，便于调试和开发
    
    # ===== 模型配置 =====
    hf_config: AutoConfig | None = None
    # Hugging Face 模型配置对象
    # - 会在 __post_init__ 中自动加载
    # - 包含模型架构、词表大小、层数等信息
    
    eos: int = -1
    # End-of-Sequence (结束符) 的 token ID
    # - -1 表示从模型配置中读取
    # - 用于判断生成何时结束
    
    # ===== KV 缓存配置（PagedAttention 核心） =====
    kvcache_block_size: int = 256
    # KV 缓存块的大小（以 token 为单位）
    # - PagedAttention 将 KV 缓存分块管理，类似操作系统的分页
    # - 必须是 256 的倍数（硬件对齐要求）
    # - 更大的块：管理开销小，但可能浪费内存
    # - 更小的块：内存利用率高，但管理开销大
    
    num_kvcache_blocks: int = -1
    # KV 缓存块的总数量
    # - -1 表示根据 gpu_memory_utilization 自动计算
    # - 总 KV 缓存大小 = num_kvcache_blocks × kvcache_block_size
    # - 这决定了系统能同时服务多少请求

    def __post_init__(self):
        """初始化后的验证和配置加载"""
        
        # 1. 验证模型路径存在
        assert os.path.isdir(self.model), f"模型路径不存在: {self.model}"
        
        # 2. 验证 KV 缓存块大小必须是 256 的倍数（GPU 内存对齐要求）
        assert self.kvcache_block_size % 256 == 0, "kvcache_block_size 必须是 256 的倍数"
        
        # 3. 验证张量并行大小在合理范围内
        assert 1 <= self.tensor_parallel_size <= 8, "tensor_parallel_size 必须在 1-8 之间"
        
        # 4. 加载 Hugging Face 模型配置
        self.hf_config = AutoConfig.from_pretrained(self.model)
        
        # 5. 调整最大序列长度：不能超过模型的位置编码限制
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        
        # 6. 验证批处理配置合理性：批处理大小至少要能容纳一个最长序列
        assert self.max_num_batched_tokens >= self.max_model_len, \
            f"max_num_batched_tokens ({self.max_num_batched_tokens}) 必须 >= max_model_len ({self.max_model_len})"
