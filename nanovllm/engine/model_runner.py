import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:
    """
    模型执行器，负责模型推理的核心逻辑。
    
    主要功能：
    1. 模型加载和初始化
    2. KV Cache 分配和管理
    3. Prefill 和 Decode 阶段的输入准备
    4. CUDA Graph 捕获和重放 (优化 Decode 性能)
    5. 多 GPU 张量并行 (Tensor Parallelism) 支持
    """

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        """
        初始化 ModelRunner。
        
        Args:
            config: 全局配置对象
            rank: 当前进程在分布式环境中的 rank (GPU 编号)
            event: 用于多进程同步的事件对象 (单 GPU 时为 Event，多 GPU 时为 Event 列表)
        """
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager  # 是否强制使用 eager 模式 (不使用 CUDA Graph)
        self.world_size = config.tensor_parallel_size  # 张量并行的 GPU 数量
        self.rank = rank
        self.event = event

        # 初始化分布式进程组 (NCCL 后端)
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        
        # 设置默认数据类型和设备
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        
        # 加载模型和采样器
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        
        # 预热模型、分配 KV Cache、捕获 CUDA Graph
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        
        # 恢复默认设置
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # 多 GPU 场景：使用共享内存进行进程间通信
        if self.world_size > 1:
            if rank == 0:
                # Rank 0 创建共享内存
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                # 其他 rank 等待并连接到共享内存，然后进入事件循环
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        """清理资源并退出分布式环境。"""
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()  # Rank 0 负责删除共享内存
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        """
        Worker 进程 (rank > 0) 的主循环。
        不断从共享内存读取任务并执行，直到收到 "exit" 指令。
        """
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        """
        从共享内存读取方法名和参数 (仅 worker 进程调用)。
        使用事件进行同步，避免数据竞争。
        """
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()  # 等待 rank 0 写入数据
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        """
        将方法名和参数写入共享内存 (仅 rank 0 调用)。
        写入后触发所有 worker 的事件。
        """
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()  # 通知所有 worker 进程

    def call(self, method_name, *args):
        """
        统一的方法调用接口。
        在多 GPU 场景下，rank 0 会先将调用信息写入共享内存，然后所有进程执行该方法。
        """
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        """
        预热模型，用于测量模型推理所需的显存。
        使用最大批次大小和序列长度进行一次前向传播。
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """
        根据可用显存动态分配 KV Cache。
        
        计算逻辑：
        1. 通过 warmup 获取模型本身的显存占用
        2. 计算单个 KV 块的字节数
        3. 根据 gpu_memory_utilization 参数，将剩余显存分配给 KV Cache
        4. 将 KV Cache 张量绑定到模型的各个 Attention 层
        """
        config = self.config
        hf_config = config.hf_config
        
        # 获取显存使用情况
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        
        # 计算单个 KV 块的字节数
        num_kv_heads = hf_config.num_key_value_heads // self.world_size  # 张量并行：KV heads 分片
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        # 2 表示 K 和 V，num_hidden_layers 表示层数
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        
        # 计算可分配的 KV 块数量
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        
        # 分配 KV Cache 张量 [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        
        # 将 KV Cache 绑定到模型的各个 Attention 层
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        """
        准备块表 (Block Table) 张量。
        块表记录了每个序列的逻辑块到物理块的映射。
        使用 -1 填充以保证批次内所有序列的块表长度一致。
        """
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        """
        准备 Prefill 阶段的输入数据。
        
        Prefill 阶段处理 prompt 中的所有 token，需要：
        1. input_ids: 待处理的 token (跳过已缓存的部分)
        2. positions: 每个 token 在序列中的位置
        3. cu_seqlens_q/k: 累积序列长度 (用于 Flash Attention 的变长序列处理)
        4. slot_mapping: 每个 token 对应的 KV Cache 物理位置
        5. block_tables: 如果有前缀缓存，需要提供块表
        """
        input_ids = []
        positions = []
        cu_seqlens_q = [0]  # Query 的累积序列长度 (实际计算的部分)
        cu_seqlens_k = [0]  # Key 的累积序列长度 (包括缓存的部分)
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        
        for seq in seqs:
            seqlen = len(seq)
            # 只处理未缓存的 token
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            
            seqlen_q = seqlen - seq.num_cached_tokens  # 需要计算的长度
            seqlen_k = seqlen  # Attention 需要看到的总长度 (包括缓存)
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            
            if not seq.block_table:    # warmup 阶段没有块表
                continue
            
            # 构建 slot_mapping: 将每个 token 映射到 KV Cache 的物理位置
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    # 最后一个块可能未满
                    end = start + seq.last_block_num_tokens 
                slot_mapping.extend(list(range(start, end)))
        
        # 如果存在前缀缓存 (K 的长度大于 Q)，需要提供块表
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs)
        
        # 转换为 GPU 张量 (使用 pin_memory 和 non_blocking 加速传输)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        
        # 设置全局上下文 (供 Attention 层使用)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        """
        准备 Decode 阶段的输入数据。
        
        Decode 阶段每次只生成一个 token，需要：
        1. input_ids: 上一步生成的 token (批次大小 = 序列数)
        2. positions: 每个 token 在序列中的位置
        3. slot_mapping: 新 token 的 KV 存储位置
        4. context_lens: 每个序列的当前长度 (用于 Paged Attention)
        5. block_tables: 块表 (用于查找历史 KV)
        """
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            # 计算新 token 在 KV Cache 中的存储位置
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
        
        # 转换为 GPU 张量
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        
        # 设置全局上下文
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        """准备采样参数 (temperature)。"""
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        """
        执行模型前向传播。
        
        对于 Decode 阶段且批次大小 <= 512，使用 CUDA Graph 加速。
        CUDA Graph 通过预先录制计算图并重放，消除 kernel launch 开销。
        """
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            # Prefill 或大批次：使用 eager 模式
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            # Decode 小批次：使用 CUDA Graph
            bs = input_ids.size(0)
            context = get_context()
            
            # 找到最接近的 graph (batch size 向上取整)
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            
            # 更新 graph 的输入变量
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            
            # 重放 graph
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """
        执行一次推理迭代 (Prefill 或 Decode)。
        
        Returns:
            生成的 token IDs (仅 rank 0 返回，其他 rank 返回 None)
        """
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        """
        捕获 CUDA Graph 用于 Decode 阶段加速。
        
        为不同的批次大小 (1, 2, 4, 8, 16, 32, ...) 分别捕获 graph。
        使用时选择最接近的 graph 并重放，避免重复捕获。
        
        CUDA Graph 的优势：
        - 消除 Python 和 CUDA kernel launch 开销
        - 对于小批次 Decode (延迟敏感场景) 有显著加速
        """
        config = self.config
        hf_config = config.hf_config
        
        # 最大批次大小限制为 512 (超过则使用 eager 模式)
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        
        # 预分配所有输入输出张量 (graph 捕获时会记录这些张量的地址)
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        
        # 定义要捕获的批次大小列表
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        # 从大到小捕获 (共享 memory pool)
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            
            # Warmup: 确保所有 kernel 都已编译
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            
            # 捕获 graph
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            
            # 第一次捕获后获取 memory pool，后续 graph 共享该 pool
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        # 保存 graph 使用的变量引用 (重放时更新这些变量)
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
