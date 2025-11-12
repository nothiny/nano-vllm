"""
LLM 推理引擎

这是 Nano-vLLM 的核心组件，负责协调整个推理流程：
1. 多进程/多GPU 管理（张量并行）
2. 请求调度（Scheduler）
3. 模型执行（ModelRunner）
4. Token 采样和解码
"""

import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:
    """
    LLM 推理引擎主类
    
    架构：
    - 单个 Scheduler：管理所有请求的调度
    - 多个 ModelRunner：每个 GPU 一个（张量并行）
    - Tokenizer：文本和 token ID 的转换
    
    工作流程：
    1. 用户提交请求 → add_request()
    2. Scheduler 调度请求 → schedule()
    3. ModelRunner 执行推理 → run()
    4. 更新序列状态 → postprocess()
    5. 返回完成的结果
    """

    def __init__(self, model, **kwargs):
        """
        初始化 LLM 引擎
        
        Args:
            model: 模型路径
            **kwargs: 配置参数（会传递给 Config）
        
        初始化流程：
        1. 解析配置参数
        2. 启动多进程（如果使用张量并行）
        3. 初始化 ModelRunner（模型加载和推理）
        4. 初始化 Tokenizer
        5. 初始化 Scheduler（请求调度）
        6. 注册退出清理函数
        """
        # ===== 1. 配置初始化 =====
        # 提取 Config 支持的参数（过滤掉不相关的 kwargs）
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        
        # ===== 2. 多进程初始化（张量并行） =====
        self.ps = []       # 子进程列表
        self.events = []   # 进程同步事件
        
        # 使用 "spawn" 上下文：每个进程完全独立（CUDA 兼容性最好）
        ctx = mp.get_context("spawn")
        
        # 启动 N-1 个子进程（主进程也会运行一个 ModelRunner）
        # 例如 tensor_parallel_size=4，则启动 3 个子进程 + 1 个主进程 = 4 个 GPU
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()  # 用于进程间同步
            # 每个子进程运行一个 ModelRunner，负责 GPU i
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        
        # ===== 3. 主进程的 ModelRunner =====
        # rank=0 的 ModelRunner 在主进程运行，负责协调和执行
        self.model_runner = ModelRunner(config, 0, self.events)
        
        # ===== 4. Tokenizer 初始化 =====
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id  # 设置结束符 ID
        
        # ===== 5. Scheduler 初始化 =====
        # 负责请求调度、KV 缓存管理、批处理策略
        self.scheduler = Scheduler(config)
        
        # ===== 6. 注册清理函数 =====
        # 程序退出时自动清理资源（关闭子进程等）
        atexit.register(self.exit)

    def exit(self):
        """
        清理资源，关闭所有进程
        
        清理顺序：
        1. 通知所有 ModelRunner 退出
        2. 删除主进程的 ModelRunner
        3. 等待所有子进程结束
        """
        self.model_runner.call("exit")  # 发送退出信号给所有 GPU 进程
        del self.model_runner            # 释放主进程的模型
        for p in self.ps:
            p.join()                     # 等待子进程结束

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        """
        添加一个新的推理请求
        
        Args:
            prompt: 输入提示（文本或 token IDs）
            sampling_params: 采样参数
        
        流程：
        1. 如果是文本，先 tokenize 转换为 token IDs
        2. 创建 Sequence 对象（封装请求信息）
        3. 添加到 Scheduler 的等待队列
        """
        # 统一转换为 token IDs
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        
        # 创建序列对象
        seq = Sequence(prompt, sampling_params)
        
        # 加入调度器的等待队列（状态：WAITING）
        self.scheduler.add(seq)

    def step(self):
        """
        执行一次推理步骤（核心循环）
        
        Returns:
            outputs: 完成的序列列表 [(seq_id, token_ids), ...]
            num_tokens: 本次处理的 token 数量
                       - 正数：Prefill 阶段处理的 token 数
                       - 负数：Decode 阶段处理的序列数（取反）
        
        执行流程：
        1. Scheduler 调度：选择要执行的序列
        2. ModelRunner 推理：前向传播 + 采样
        3. Scheduler 后处理：更新状态、管理 KV 缓存
        4. 收集完成的序列
        """
        # ===== 1. 调度阶段 =====
        # 决定本次执行哪些序列（考虑显存、批处理限制等）
        # is_prefill: True=处理新请求(Prefill), False=继续生成(Decode)
        seqs, is_prefill = self.scheduler.schedule()
        
        # ===== 2. 模型推理 =====
        # 前向传播 → 采样 → 生成新 token
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        
        # ===== 3. 后处理 =====
        # 更新序列状态、KV 缓存、检查是否完成
        self.scheduler.postprocess(seqs, token_ids)
        
        # ===== 4. 收集完成的序列 =====
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        
        # ===== 5. 计算吞吐量指标 =====
        # Prefill: 统计处理的 token 数（一次可能处理很多 tokens）
        # Decode: 统计序列数（每个序列生成 1 个 token），用负数表示
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        
        return outputs, num_tokens

    def is_finished(self):
        """
        检查是否所有请求都已完成
        
        Returns:
            bool: True 表示没有待处理的请求了
        """
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        """
        批量生成（主要的用户接口）
        
        Args:
            prompts: 提示词列表（文本或 token IDs）
            sampling_params: 采样参数（单个或列表）
            use_tqdm: 是否显示进度条
        
        Returns:
            list[dict]: 生成结果列表，每个元素包含：
                - "text": 生成的文本
                - "token_ids": 生成的 token IDs
        
        执行流程：
        1. 初始化进度条
        2. 添加所有请求到队列
        3. 循环执行 step() 直到全部完成
        4. 收集和解码结果
        5. 按原始顺序返回
        """
        # ===== 1. 初始化进度条 =====
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        
        # ===== 2. 统一采样参数格式 =====
        # 如果只提供一个 SamplingParams，则所有请求共享
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        
        # ===== 3. 添加所有请求 =====
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        
        # ===== 4. 执行主循环 =====
        outputs = {}  # {seq_id: token_ids}
        prefill_throughput = decode_throughput = 0.
        
        while not self.is_finished():
            # 记录开始时间（用于计算吞吐量）
            t = perf_counter()
            
            # 执行一步推理
            output, num_tokens = self.step()
            
            # 更新进度条的吞吐量显示
            if use_tqdm:
                if num_tokens > 0:
                    # Prefill 阶段：正数表示处理的 token 数
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    # Decode 阶段：负数表示序列数，取反计算吞吐量
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            
            # 收集完成的序列
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)  # 进度 +1
        
        # ===== 5. 整理输出（按原始顺序） =====
        # seq_id 是按提交顺序分配的，排序后就是原始顺序
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        
        # ===== 6. 解码为文本 =====
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        
        # ===== 7. 清理进度条 =====
        if use_tqdm:
            pbar.close()
        
        return outputs
