#!/usr/bin/env python3
"""
方案2: 首 Token 延迟测试 (TTFT - Time To First Token)

测试 decode_first 策略在降低首 token 延迟方面的优势

TTFT 定义: 从请求提交到收到第一个生成 token 的时间
"""
import os
import sys
import time
import gc
from random import randint, seed
from dataclasses import dataclass
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.pd_scheduler import PDScheduler
from nanovllm.engine.model_runner import ModelRunner
from transformers import AutoTokenizer
from dataclasses import fields
import torch.multiprocessing as mp
import atexit


def cleanup(engine=None):
    if engine is not None:
        try:
            engine.exit()
        except Exception:
            pass
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


class TTFTEngine:
    """
    支持 TTFT 测量的 LLM Engine
    
    记录每个序列的:
    - 提交时间
    - 首 token 时间 (TTFT)
    - 完成时间
    """
    
    def __init__(self, model, enable_pd=False, pd_policy="prefill_first", **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.config = config
        self.enable_pd = enable_pd
        
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        
        if enable_pd:
            config.enable_pd_disaggregation = True
            config.pd_schedule_policy = pd_policy
            self.scheduler = PDScheduler(config)
        else:
            self.scheduler = Scheduler(config)
        
        self._exited = False
        atexit.register(self.exit)
        
        # TTFT 追踪
        self.submit_times = {}      # seq_id -> submit_time
        self.first_token_times = {} # seq_id -> first_token_time
        self.finish_times = {}      # seq_id -> finish_time
        self.first_token_generated = set()  # 已生成首token的序列
    
    def exit(self):
        if self._exited:
            return
        self._exited = True
        try:
            atexit.unregister(self.exit)
        except Exception:
            pass
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()
    
    def add_request(self, prompt: list[int], sampling_params: SamplingParams):
        seq = Sequence(prompt, sampling_params)
        self.submit_times[seq.seq_id] = time.perf_counter()
        self.scheduler.add(seq)
        return seq.seq_id
    
    def step(self):
        if self.enable_pd:
            seqs, is_prefill, prefill_info = self.scheduler.schedule()
            if not seqs:
                return [], 0
            token_ids = self.model_runner.call("run", seqs, is_prefill, prefill_info)
            self.scheduler.postprocess(seqs, token_ids, is_prefill, prefill_info)
        else:
            seqs, is_prefill = self.scheduler.schedule()
            token_ids = self.model_runner.call("run", seqs, is_prefill)
            self.scheduler.postprocess(seqs, token_ids)
        
        # 记录首 token 时间
        now = time.perf_counter()
        for seq in seqs:
            if seq.seq_id not in self.first_token_generated:
                if seq.num_completion_tokens > 0:  # 已生成第一个 token
                    self.first_token_times[seq.seq_id] = now
                    self.first_token_generated.add(seq.seq_id)
        
        # 收集完成的序列
        outputs = []
        for seq in seqs:
            if seq.is_finished:
                self.finish_times[seq.seq_id] = now
                outputs.append((seq.seq_id, seq.completion_token_ids))
        
        return outputs, len(seqs)
    
    def is_finished(self):
        return self.scheduler.is_finished()
    
    def get_ttft_stats(self):
        """获取 TTFT 统计"""
        ttfts = []
        for seq_id in self.first_token_times:
            if seq_id in self.submit_times:
                ttft = self.first_token_times[seq_id] - self.submit_times[seq_id]
                ttfts.append(ttft)
        
        if not ttfts:
            return None
        
        ttfts.sort()
        return {
            "count": len(ttfts),
            "min": min(ttfts) * 1000,      # ms
            "max": max(ttfts) * 1000,      # ms
            "avg": sum(ttfts) / len(ttfts) * 1000,  # ms
            "p50": ttfts[len(ttfts) // 2] * 1000,   # ms
            "p90": ttfts[int(len(ttfts) * 0.9)] * 1000,  # ms
            "p99": ttfts[int(len(ttfts) * 0.99)] * 1000 if len(ttfts) >= 100 else ttfts[-1] * 1000,  # ms
        }


def run_ttft_test(
    model_path: str,
    num_seqs: int = 64,
    input_len: int = 256,
    output_len: int = 32,
    enable_pd: bool = False,
    pd_policy: str = "prefill_first",
):
    """运行 TTFT 测试"""
    seed(42)
    
    # 生成测试数据 - 固定长度便于对比
    prompt_token_ids = [
        [randint(0, 10000) for _ in range(input_len)]
        for _ in range(num_seqs)
    ]
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=output_len)
        for _ in range(num_seqs)
    ]
    
    mode = "Normal" if not enable_pd else f"PD-{pd_policy}"
    print(f"\n{'='*60}")
    print(f"[{mode}] TTFT Test")
    print(f"{'='*60}")
    print(f"  Sequences: {num_seqs}")
    print(f"  Input Length: {input_len}")
    print(f"  Output Length: {output_len}")
    
    # 创建引擎
    kwargs = {
        "max_model_len": 4096,
        "enforce_eager": False,
    }
    if enable_pd:
        kwargs.update({
            "prefill_chunk_size": 4096,
            "max_prefill_batch_tokens": 8192,
            "min_decode_batch_size": 1,  # 允许小 batch decode 以降低 TTFT
        })
    
    engine = TTFTEngine(model_path, enable_pd=enable_pd, pd_policy=pd_policy, **kwargs)
    
    # Warmup
    warmup_seq_id = engine.add_request([0] * 10, SamplingParams(max_tokens=5))
    while not engine.is_finished():
        engine.step()
    
    # 清空 warmup 统计
    engine.submit_times.clear()
    engine.first_token_times.clear()
    engine.finish_times.clear()
    engine.first_token_generated.clear()
    
    # 提交所有请求
    print(f"\n  Submitting {num_seqs} requests...")
    start_time = time.perf_counter()
    for prompt, sp in zip(prompt_token_ids, sampling_params):
        engine.add_request(prompt, sp)
    
    # 运行直到完成
    print(f"  Processing...")
    completed = 0
    while not engine.is_finished():
        outputs, _ = engine.step()
        completed += len(outputs)
    
    total_time = time.perf_counter() - start_time
    
    # 获取 TTFT 统计
    ttft_stats = engine.get_ttft_stats()
    
    print(f"\n  Results:")
    print(f"    Total Time: {total_time:.2f}s")
    if ttft_stats:
        print(f"    TTFT (Time To First Token):")
        print(f"      Min:  {ttft_stats['min']:.2f} ms")
        print(f"      Avg:  {ttft_stats['avg']:.2f} ms")
        print(f"      P50:  {ttft_stats['p50']:.2f} ms")
        print(f"      P90:  {ttft_stats['p90']:.2f} ms")
        print(f"      Max:  {ttft_stats['max']:.2f} ms")
    
    cleanup(engine)
    
    return {
        "mode": mode,
        "total_time": total_time,
        "ttft": ttft_stats,
    }


def main():
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    
    print("\n" + "=" * 70)
    print("TTFT BENCHMARK - Time To First Token Test")
    print("=" * 70)
    print(f"Model: {model_path}")
    print("\nThis test measures how quickly each request receives its first token.")
    print("decode_first should have lower TTFT for early requests.")
    
    results = []
    
    # Test 1: Normal mode
    r1 = run_ttft_test(
        model_path,
        num_seqs=64,
        input_len=256,
        output_len=32,
        enable_pd=False,
    )
    results.append(r1)
    
    # Test 2: PD prefill_first (baseline for PD)
    r2 = run_ttft_test(
        model_path,
        num_seqs=64,
        input_len=256,
        output_len=32,
        enable_pd=True,
        pd_policy="prefill_first",
    )
    results.append(r2)
    
    # Test 3: PD decode_first (optimized for TTFT)
    r3 = run_ttft_test(
        model_path,
        num_seqs=64,
        input_len=256,
        output_len=32,
        enable_pd=True,
        pd_policy="decode_first",
    )
    results.append(r3)
    
    # Comparison
    print("\n" + "=" * 70)
    print("TTFT COMPARISON")
    print("=" * 70)
    print(f"\n{'Mode':<25} {'Total(s)':<12} {'TTFT Avg(ms)':<15} {'TTFT P50(ms)':<15} {'TTFT P90(ms)':<15}")
    print("-" * 85)
    
    for r in results:
        ttft = r['ttft']
        if ttft:
            print(f"{r['mode']:<25} {r['total_time']:<12.2f} {ttft['avg']:<15.2f} {ttft['p50']:<15.2f} {ttft['p90']:<15.2f}")
    
    print("-" * 85)
    
    # 分析
    print("\n分析:")
    if len(results) >= 3:
        normal_avg = results[0]['ttft']['avg'] if results[0]['ttft'] else 0
        prefill_first_avg = results[1]['ttft']['avg'] if results[1]['ttft'] else 0
        decode_first_avg = results[2]['ttft']['avg'] if results[2]['ttft'] else 0
        
        if decode_first_avg < normal_avg:
            improvement = (1 - decode_first_avg / normal_avg) * 100
            print(f"  ✓ decode_first TTFT 比 Normal 降低 {improvement:.1f}%")
        else:
            print(f"  ✗ decode_first TTFT 没有优势")
        
        if decode_first_avg < prefill_first_avg:
            improvement = (1 - decode_first_avg / prefill_first_avg) * 100
            print(f"  ✓ decode_first TTFT 比 prefill_first 降低 {improvement:.1f}%")


if __name__ == "__main__":
    main()

