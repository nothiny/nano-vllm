"""采样参数配置"""
from dataclasses import dataclass


@dataclass
class SamplingParams:
    """
    文本生成的采样参数
    
    Attributes:
        temperature: 采样温度，控制生成的随机性
                    - 越大(如2.0)越随机、越有创造性
                    - 越小(如0.1)越确定、越保守
                    - 默认1.0为标准采样
        max_tokens: 最多生成的 token 数量
        ignore_eos: 是否忽略结束符(EOS token)
                   - True: 强制生成到 max_tokens
                   - False: 遇到 EOS 就停止
    """
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
    # 初始化后验证参数，由于dataclass默认执行__init__，所以需要手动定义__post_init__，在__init__执行后执行
    def __post_init__(self):
        # 不支持贪心采样(temperature=0)，必须保持一定随机性
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
