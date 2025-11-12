"""
Nano-vLLM 性能基准测试

这个脚本用于测试 Nano-vLLM 的推理性能，主要测试指标：
- 吞吐量（Throughput）: 每秒生成的 token 数量
- 处理时间（Time）: 完成所有请求的总时间
- 可扩展性: 在大批量并发请求下的表现

测试配置：
- 请求数量: 256 个序列
- 输入长度: 100-1024 tokens（随机）
- 输出长度: 100-1024 tokens（随机）
- 模型: Qwen3-0.6B
"""

import os
import time
from random import randint, seed
from nanovllm import LLM, SamplingParams
# 如果要测试官方 vLLM 进行对比，取消下一行的注释
# from vllm import LLM, SamplingParams


def main():
    # ===== 1. 测试配置 =====
    
    # 设置随机种子，确保每次测试的输入数据一致，便于复现
    seed(0)
    
    # 测试参数配置
    num_seqs = 256  # 总请求数量：256 个序列
    max_input_len = 1024  # 最大输入长度：1024 tokens
    max_ouput_len = 1024  # 最大输出长度：1024 tokens
    
    # ===== 2. 初始化模型 =====
    
    # 指定模型路径
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    
    # 初始化 LLM 引擎
    # - enforce_eager=False: 启用 CUDA graph 优化，提升性能
    # - max_model_len=4096: 设置模型最大序列长度为 4096
    llm = LLM(path, enforce_eager=False, max_model_len=4096)

    # ===== 3. 生成测试数据 =====
    
    # 为每个序列生成随机的 token IDs
    # - 每个序列的长度在 100 到 max_input_len 之间随机
    # - 每个 token ID 在 0 到 10000 之间随机
    # 这样可以模拟真实场景中不同长度的输入
    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, max_input_len))] 
        for _ in range(num_seqs)
    ]
    
    # 为每个序列配置采样参数
    # - temperature=0.6: 采样温度
    # - ignore_eos=True: 忽略 EOS token，确保生成到 max_tokens
    # - max_tokens: 每个序列的输出长度在 100 到 max_ouput_len 之间随机
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) 
        for _ in range(num_seqs)
    ]
    
    # 如果要对比测试官方 vLLM，取消下一行注释
    # vLLM 需要将 token IDs 封装成字典格式
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    # ===== 4. 预热（Warm-up） =====
    
    # 执行一次小规模推理来预热模型
    # 这样可以：
    # 1. 初始化 CUDA kernels
    # 2. 分配必要的内存
    # 3. 编译 CUDA graph（如果启用）
    # 确保后续的基准测试不受初始化开销的影响
    llm.generate(["Benchmark: "], SamplingParams())
    
    # ===== 5. 执行基准测试 =====
    
    # 记录开始时间
    t = time.time()
    
    # 执行批量推理
    # - use_tqdm=False: 禁用进度条，避免影响性能测量
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    
    # 计算总耗时
    t = (time.time() - t)
    
    # ===== 6. 计算和输出性能指标 =====
    
    # 计算总生成 token 数量
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    
    # 计算吞吐量（tokens/秒）
    throughput = total_tokens / t
    
    # 输出性能报告
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")
    
    # 示例输出：
    # Total: 133966tok, Time: 93.41s, Throughput: 1434.13tok/s


if __name__ == "__main__":
    main()
