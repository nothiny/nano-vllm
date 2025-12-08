import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    # PD分离相关配置
    enable_pd_disaggregation: bool = False  # 是否启用PD分离
    prefill_chunk_size: int = 4096  # chunked prefill的块大小
    max_prefill_batch_tokens: int = 8192  # prefill阶段最大batch token数
    max_decode_batch_size: int = 256  # decode阶段最大batch大小
    min_decode_batch_size: int = 1  # decode阶段最小batch大小（用于decode_first策略）
    pd_schedule_policy: str = "decode_first"  # 调度策略: "decode_first" 或 "prefill_first"

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
