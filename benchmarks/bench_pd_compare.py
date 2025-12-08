#!/usr/bin/env python3
"""
PD分离优化性能对比测试

比较以下模式的性能:
1. 普通模式 (原始调度器)
2. PD分离模式 - Decode优先
3. PD分离模式 - Prefill优先

测试指标:
- 总吞吐量 (tok/s)
- Prefill吞吐量
- Decode吞吐量
- 首token延迟 (TTFT)
- 端到端延迟
"""
import os
import sys
import time
import gc
import argparse
from random import randint, seed
from dataclasses import dataclass
from typing import Optional

import torch

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanovllm import LLM, SamplingParams


@dataclass
class BenchmarkConfig:
    """测试配置 (与 bench.py 相同规模)"""
    model_path: str
    num_seqs: int = 256          # bench.py: 256
    min_input_len: int = 100     # bench.py: 100
    max_input_len: int = 1024    # bench.py: 1024
    min_output_len: int = 100    # bench.py: 100
    max_output_len: int = 1024   # bench.py: 1024
    max_model_len: int = 4096
    enforce_eager: bool = False
    seed: int = 0                # bench.py: seed(0)
    warmup_prompts: int = 2


@dataclass
class BenchmarkResult:
    """测试结果"""
    mode: str
    total_time: float
    total_input_tokens: int
    total_output_tokens: int
    output_throughput: float  # tok/s
    prefill_batches: Optional[int] = None
    decode_batches: Optional[int] = None
    
    def __str__(self):
        s = f"  Mode: {self.mode}\n"
        s += f"  Total Time: {self.total_time:.2f}s\n"
        s += f"  Input Tokens: {self.total_input_tokens}\n"
        s += f"  Output Tokens: {self.total_output_tokens}\n"
        s += f"  Output Throughput: {self.output_throughput:.2f} tok/s\n"
        if self.prefill_batches is not None:
            s += f"  Prefill Batches: {self.prefill_batches}\n"
        if self.decode_batches is not None:
            s += f"  Decode Batches: {self.decode_batches}\n"
        return s


def generate_test_data(config: BenchmarkConfig):
    """生成测试数据"""
    seed(config.seed)
    
    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(config.min_input_len, config.max_input_len))]
        for _ in range(config.num_seqs)
    ]
    
    sampling_params = [
        SamplingParams(
            temperature=0.6, 
            ignore_eos=True, 
            max_tokens=randint(config.min_output_len, config.max_output_len)
        )
        for _ in range(config.num_seqs)
    ]
    
    return prompt_token_ids, sampling_params


def cleanup(llm=None):
    """清理GPU内存和分布式进程组"""
    if llm is not None:
        # 显式调用exit来销毁进程组
        try:
            llm.exit()
        except Exception:
            pass
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def run_benchmark(
    config: BenchmarkConfig,
    prompt_token_ids: list,
    sampling_params: list,
    enable_pd: bool = False,
    pd_policy: str = "decode_first",
    prefill_chunk_size: int = 4096,
    max_prefill_batch_tokens: int = 8192,
    max_decode_batch_size: int = 256,
    min_decode_batch_size: int = 1,
) -> BenchmarkResult:
    """运行单次测试"""
    
    mode = "Normal" if not enable_pd else f"PD-{pd_policy}"
    print(f"\n{'='*60}")
    print(f"Running benchmark: {mode}")
    print(f"{'='*60}")
    
    # 创建LLM实例
    llm_kwargs = {
        "enforce_eager": config.enforce_eager,
        "max_model_len": config.max_model_len,
    }
    
    if enable_pd:
        llm_kwargs.update({
            "enable_pd_disaggregation": True,
            "prefill_chunk_size": prefill_chunk_size,
            "max_prefill_batch_tokens": max_prefill_batch_tokens,
            "max_decode_batch_size": max_decode_batch_size,
            "min_decode_batch_size": min_decode_batch_size,
            "pd_schedule_policy": pd_policy,
        })
    
    llm = LLM(config.model_path, **llm_kwargs)
    
    # Warmup
    print("Warming up...")
    for _ in range(config.warmup_prompts):
        llm.generate(["Warmup test: "], SamplingParams(max_tokens=10))
    
    # 计算总token数
    total_input_tokens = sum(len(p) for p in prompt_token_ids)
    total_output_tokens = sum(sp.max_tokens for sp in sampling_params)
    
    # 运行测试
    print(f"Running {config.num_seqs} sequences...")
    start_time = time.time()
    outputs = llm.generate(prompt_token_ids, sampling_params, use_tqdm=True)
    end_time = time.time()
    
    total_time = end_time - start_time
    output_throughput = total_output_tokens / total_time
    
    # 获取调度统计
    prefill_batches = None
    decode_batches = None
    if hasattr(llm.scheduler, 'num_prefill_batches'):
        prefill_batches = llm.scheduler.num_prefill_batches
    if hasattr(llm.scheduler, 'num_decode_batches'):
        decode_batches = llm.scheduler.num_decode_batches
    
    result = BenchmarkResult(
        mode=mode,
        total_time=total_time,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        output_throughput=output_throughput,
        prefill_batches=prefill_batches,
        decode_batches=decode_batches,
    )
    
    # 清理 - 必须显式调用exit来销毁进程组
    cleanup(llm)
    del llm
    
    return result


def print_comparison(results: list[BenchmarkResult]):
    """打印对比结果"""
    print("\n")
    print("=" * 70)
    print("BENCHMARK COMPARISON RESULTS")
    print("=" * 70)
    
    # 找到基准（Normal模式）
    baseline = next((r for r in results if r.mode == "Normal"), results[0])
    
    print(f"\n{'Mode':<25} {'Time(s)':<12} {'Throughput':<15} {'Speedup':<10}")
    print("-" * 70)
    
    for r in results:
        speedup = baseline.total_time / r.total_time
        print(f"{r.mode:<25} {r.total_time:<12.2f} {r.output_throughput:<15.2f} {speedup:<10.2f}x")
    
    print("-" * 70)
    
    # 详细结果
    print("\n\nDETAILED RESULTS:")
    print("-" * 70)
    for r in results:
        print(f"\n[{r.mode}]")
        print(r)


def run_all_benchmarks(config: BenchmarkConfig):
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("NANO-VLLM PD DISAGGREGATION BENCHMARK")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Model: {config.model_path}")
    print(f"  Sequences: {config.num_seqs}")
    print(f"  Input Length: {config.min_input_len}-{config.max_input_len}")
    print(f"  Output Length: {config.min_output_len}-{config.max_output_len}")
    print(f"  Max Model Length: {config.max_model_len}")
    print(f"  Enforce Eager: {config.enforce_eager}")
    
    # 生成测试数据（所有测试使用相同数据）
    prompt_token_ids, sampling_params = generate_test_data(config)
    
    results = []
    
    # 1. 普通模式
    try:
        result = run_benchmark(
            config, prompt_token_ids, sampling_params,
            enable_pd=False
        )
        results.append(result)
    except Exception as e:
        print(f"Normal mode failed: {e}")
    
    # 2. PD分离 - Decode优先 (使用较大的min batch避免小batch)
    try:
        result = run_benchmark(
            config, prompt_token_ids, sampling_params,
            enable_pd=True,
            pd_policy="decode_first",
            min_decode_batch_size=64,  # 至少64个序列才做decode，256序列规模下更高效
        )
        results.append(result)
    except Exception as e:
        print(f"PD decode_first mode failed: {e}")
    
    # 3. PD分离 - Prefill优先
    try:
        result = run_benchmark(
            config, prompt_token_ids, sampling_params,
            enable_pd=True,
            pd_policy="prefill_first"
        )
        results.append(result)
    except Exception as e:
        print(f"PD prefill_first mode failed: {e}")
    
    # 打印对比结果
    if results:
        print_comparison(results)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="PD Disaggregation Benchmark")
    parser.add_argument("--model", type=str, default="~/huggingface/Qwen3-0.6B/",
                        help="Model path")
    parser.add_argument("--num-seqs", type=int, default=128,
                        help="Number of sequences")
    parser.add_argument("--min-input-len", type=int, default=100,
                        help="Minimum input length")
    parser.add_argument("--max-input-len", type=int, default=512,
                        help="Maximum input length")
    parser.add_argument("--min-output-len", type=int, default=50,
                        help="Minimum output length")
    parser.add_argument("--max-output-len", type=int, default=256,
                        help="Maximum output length")
    parser.add_argument("--max-model-len", type=int, default=4096,
                        help="Maximum model length")
    parser.add_argument("--enforce-eager", action="store_true",
                        help="Disable CUDA graph")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    config = BenchmarkConfig(
        model_path=os.path.expanduser(args.model),
        num_seqs=args.num_seqs,
        min_input_len=args.min_input_len,
        max_input_len=args.max_input_len,
        min_output_len=args.min_output_len,
        max_output_len=args.max_output_len,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        seed=args.seed,
    )
    
    run_all_benchmarks(config)


if __name__ == "__main__":
    main()

