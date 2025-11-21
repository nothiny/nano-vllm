from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:
    """
    代表 KV Cache 中的一个物理块 (Physical Block)。
    存储了实际的 Token 数据和哈希值。
    """

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0  # 引用计数，表示有多少个逻辑块指向该物理块
        self.hash = -1      # 块内容的哈希值，用于前缀缓存匹配
        self.token_ids = [] # 该块当前存储的 Token IDs

    def update(self, hash: int, token_ids: list[int]):
        """更新块的哈希值和 Token 数据。通常在块填满时调用。"""
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        """重置块状态，以便重新分配给新的序列或逻辑块。"""
        self.ref_count = 1  # 分配时初始化引用计数为 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    """
    块管理器，负责管理 KV Cache 的物理块分配、回收和复用。
    实现了基于哈希的前缀缓存 (Prefix Caching) 机制，允许不同序列共享相同的 KV 块。
    """

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        # 预先初始化所有物理块对象
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        # 哈希表：映射 (块内容哈希 -> 物理块 ID)，用于查找可复用的 cached block
        self.hash_to_block_id: dict[int, int] = dict()
        # 空闲块 ID 队列，用于分配新块
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        # 已使用的块 ID 集合
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """
        计算 Token 序列的哈希值。
        
        Args:
            token_ids: 当前块内的 Token ID 列表
            prefix: 前一个块的哈希值 (链式哈希)，用于保证前缀的一致性
        """
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        """内部辅助方法：从空闲列表中取出一个指定 ID 的块并标记为已使用。"""
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        """内部辅助方法：将指定 ID 的块标记为空闲。注意：不清除哈希表中的记录，以便后续复用 (Resurrection)。"""
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        """
        检查是否有足够的空闲块来满足新序列的 Prefill 需求。
        """
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        """
        为新序列分配块 (Prefill 阶段)。
        会尝试复用已有的块 (Prefix Caching)。
        """
        assert not seq.block_table
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            # 只有填满的块才计算哈希，用于缓存匹配。不满的块 (通常是最后一个) 不进缓存。
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            
            # 尝试在哈希表中查找对应的物理块 ID
            block_id = self.hash_to_block_id.get(h, -1)
            
            # 如果未找到，或者物理块内容不匹配 (哈希碰撞保护)，则视为 Cache Miss
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            
            if cache_miss:
                # Cache Miss: 分配一个新的空闲块
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                # Cache Hit: 复用现有块
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    # 如果块正在被其他序列使用，增加引用计数
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    # 如果块在空闲列表中 (ref_count=0 但仍在 hash_to_block_id 中)，将其“复活”
                    # 这利用了 _deallocate_block 不清除哈希映射的特性
                    block = self._allocate_block(block_id)
            
            # 如果计算了有效哈希 (完整块)，更新块信息并维护哈希表
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        """
        释放序列占用的所有块。减少引用计数，若计数归零则回收物理块。
        """
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """
        检查是否可以为序列追加一个 Token (Decode 阶段)。
        如果当前最后一个块已满，需要一个新的空闲块。
        逻辑：len(seq) % block_size == 1 表示刚刚进入一个新的块 (需要分配)。
        """
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """
        在 Decode 阶段调度前调用。
        根据序列长度，决定是分配新块，还是为刚填满的块计算哈希以供缓存。
        """
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        
        if len(seq) % self.block_size == 1:
            # 刚刚开启了一个新块 (上一个块已满并终结)
            # 上一个块应该已经计算过哈希了
            assert last_block.hash != -1
            # 分配新的物理块给这个新开始的逻辑块
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
            
        elif len(seq) % self.block_size == 0:
            # 刚刚填满了一个块，计算其哈希并加入缓存
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks-1)
            # 获取前缀哈希 (即倒数第二个块的哈希)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            
            # 更新块的哈希信息
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            # 块处于中间填充状态，无需操作
            assert last_block.hash == -1
