#!/usr/bin/env python3
"""
快速性能对比测试

使用较少的序列快速比较普通模式和PD分离模式的性能
"""
import os
import sys
import time
import gc
from random import randint, seed

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanovllm import LLM, SamplingParams


def cleanup(llm=None):
    """清理GPU内存和分布式进程组"""
    if llm is not None:
        try:
            llm.exit()
        except Exception:
            pass
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def run_test(model_path: str, enable_pd: bool, num_seqs: int = 32):
    """运行单次测试"""
    seed(42)
    
    # 生成测试数据
    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, 300))]
        for _ in range(num_seqs)
    ]
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(50, 150))
        for _ in range(num_seqs)
    ]
    
    total_output_tokens = sum(sp.max_tokens for sp in sampling_params)
    total_input_tokens = sum(len(p) for p in prompt_token_ids)
    
    # 创建LLM
    kwargs = {"max_model_len": 4096, "enforce_eager": False}
    if enable_pd:
        kwargs.update({
            "enable_pd_disaggregation": True,
            "pd_schedule_policy": "decode_first",
        })
    
    mode = "PD Mode" if enable_pd else "Normal Mode"
    print(f"\n[{mode}]")
    
    llm = LLM(model_path, **kwargs)
    
    # Warmup
    llm.generate(["Warmup: "], SamplingParams(max_tokens=5))
    
    # Benchmark
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    elapsed = time.time() - t
    
    throughput = total_output_tokens / elapsed
    
    print(f"  Input tokens:  {total_input_tokens}")
    print(f"  Output tokens: {total_output_tokens}")
    print(f"  Time:          {elapsed:.2f}s")
    print(f"  Throughput:    {throughput:.2f} tok/s")
    
    # 显示调度统计
    if hasattr(llm.scheduler, 'num_prefill_batches'):
        print(f"  Prefill batches: {llm.scheduler.num_prefill_batches}")
        print(f"  Decode batches:  {llm.scheduler.num_decode_batches}")
    
    # 清理 - 必须显式调用exit来销毁进程组
    cleanup(llm)
    del llm
    
    return elapsed, throughput


def main():
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    num_seqs = 32
    
    print("=" * 50)
    print("Quick PD Disaggregation Benchmark")
    print("=" * 50)
    print(f"Model: {model_path}")
    print(f"Sequences: {num_seqs}")
    
    # 普通模式
    normal_time, normal_throughput = run_test(model_path, enable_pd=False, num_seqs=num_seqs)
    
    # PD模式
    pd_time, pd_throughput = run_test(model_path, enable_pd=True, num_seqs=num_seqs)
    
    # 对比
    speedup = normal_time / pd_time
    print("\n" + "=" * 50)
    print("COMPARISON")
    print("=" * 50)
    print(f"Normal:  {normal_throughput:.2f} tok/s")
    print(f"PD Mode: {pd_throughput:.2f} tok/s")
    print(f"Speedup: {speedup:.2f}x")
    print("=" * 50)


if __name__ == "__main__":
    main()

