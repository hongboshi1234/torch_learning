import torch
class BlockTable:
    def __init__(self, blocks):
        self.block_list = blocks
        self.cur_index = 0
        self.offset = 0

class PagedKVCache:
    def __init__(self, block_size, nh, hd, available_memory, dtype=torch.bfloat16):
        self.block_size = block_size
        assert dtype is torch.bfloat16
        self.num_blocks = int(available_memory/2/2/block_size/nh/hd) # 2 byte * 2 (kv) * num_blocks * block_size * nh * hd 
        self.free_blocks = set([i for i in range(self.num_blocks)])
        print(f"total number of blocks allocated for cache is {self.num_blocks}")
        self.k = torch.empty(self.num_blocks, block_size, nh, hd)
        self.v = torch.empty(self.num_blocks, block_size, nh, hd)
        
    def allocate_cache(self, max_len):
        required_blocks = max_len//self.block_size + 1
        if required_blocks > len(self.free_blocks):
            raise Exception("not enough memory for max_len tokens")
        blocks = []
        for _ in range(required_blocks):
            blocks.append(self.free_blocks.pop())
        return BlockTable(blocks)
            
        
    def free_cache(self, block_table):
        for block_id in block_table.block_list: 
            self.free_blocks.add(block_id)

    def write_cache(self, k, v, block_table):
        L = k.shape[0]
        logical = block_table.cur_index # cur_index is logical, block_id is physical
        offset = block_table.offset
        for i in range(L):
            if offset == self.block_size:
                logical += 1
                offset = 0
            phy = block_table.block_list[logical]
            self.k[phy, offset] = k[i]
            self.v[phy, offset] = v[i]
            offset += 1
            print(phy, offset)
        block_table.cur_index = logical
        block_table.offset=offset

    def get_cache(self, block_table):
        _, _, nh, hd = self.k.shape
        logical = block_table.cur_index
        phy = block_table.block_list[logical]
        offset = block_table.offset
        phy_ids = torch.tensor(block_table.block_list[:logical], dtype=torch.long)
        k = self.k.index_select(0, phy_ids).contiguous().view(-1, nh, hd)
        v = self.v.index_select(0, phy_ids).contiguous().view(-1, nh, hd)
        if offset != 0:
            k_offset = self.k[phy, :offset].view(-1, nh, hd)
            v_offset = self.v[phy, :offset].view(-1, nh, hd)
            k = torch.cat([k, k_offset], dim=0)
            v = torch.cat([v, v_offset], dim=0)
        return k, v


L=100
nh = 8
hd = 256
block_size=16
prefill_k = torch.randn(L, nh, hd)
prefill_v = torch.randn(L, nh, hd)

k1 = torch.randn(1, nh, hd)
v1 = torch.randn(1, nh, hd)

k2 = torch.randn(1, nh, hd)
v2 = torch.randn(1, nh, hd)

cache = PagedKVCache(block_size, nh, hd, 6000000000)
block_table = cache.allocate_cache(max_len=100)
block_table2 = cache.allocate_cache(max_len=100)
cache.write_cache(prefill_k, prefill_v, block_table)

prefill_k2 = torch.randn(L, nh, hd)
prefill_v2 = torch.randn(L, nh, hd)
cache.write_cache(prefill_k2, prefill_v2, block_table2)

cache.write_cache(k1, v1, block_table)
cache.write_cache(k2, v2, block_table2)
old_k, oldv = cache.get_cache(block_table)
old_k2, oldv2 = cache.get_cache(block_table2)
print(f"prefill_k2[1]", prefill_k2[1])
print(f"old_k2[1]", old_k2[1])
print(f"k2", k2)
print(f"old_k2[-1]", old_k2[-1])
