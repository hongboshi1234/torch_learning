import torch
import math
class PagedKVCache:
    def __init__(self, num_blocks, block_size, num_heads, head_size, max_len):
        self.k = torch.empty(num_blocks, block_size, num_heads, head_size)
        self.v = torch.like(self.k)
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_size = head_size
        self.cur_len = 0
        self.max_len = max_len
    
    def write_cache(self, k, v): # k, v must be in shape of n_head, head_size
        assert self.cur_len <= self.max_len
        row = self.cur_len // self.block_size
        col = self.cur_len % self.block_size
        self.k[row, col, :, :] = k
        self.v[row, col, :, :] = v
        self.cur_len += 1

    def get_cache(self):
        row = self.cur_len // self.block_size
        col = self.cur_len % self.cur_len
        k_list = [self.k[i] for i in range(row-1)]
        v_list = [self.v[i] for i in range(row-1)]
 
        if col:
            k_list.append(row+1, :col, :, :)
            v_list.append(row+1, :col, :, :)
        k = torch.stack(k_list, dim=0)
        v = torch.stack(v_list, dim=0)
        return k, v

class Attention(torch.nn.Module):
    def __init__(self, hidden_size, max_len, num_heads, head_dim, max_len, block_size):
        self.hidden_size = hidden_size
        self.num_heads = num_head
        self.head_dim = head_dim
        self.q_proj = torch.ModuleList([torch.nn.Linear(hidden_size, head_size) for _ in range(num_heads)])
        self.k_proj = torch.ModuleList([torch.nn.Linear(hidden_size, head_size) for _ in range(num_heads)])
        self.v_proj = torch.ModuleList([torch.nn.Linear(hidden_size, head_size) for _ in range(num_heads)])
        num_blocks = self.max_len // block_size + 1
        self.cache = PagedKVcache(num_blocks, block_size, num_heads, head_size)
    @torch.no_grad()
    def forward(h):  # L, nh, hd
        L, nh, hd = h.shape
        for i in range(L):
            q = self.q_proj(h[i])
            k = self.q_proj(h[i])
            v = self.q_proj(h[i])
            self.cache.write_cache(k, v) 
        k, v = self.cache.get_cache()  # k in shape of L, num_head, head_size

    def forward_decode(h):
        L, head_dim = h.shape # 1, head_dim
        q_list = []
        k_list= []
        v_list = []
        for i in range(nh):
            q = self.q_proj[i](h) # 1, head_size
            k = self.k_proj[i](h)
            v = self.v_proj[i](h)
            q_list.append(q)
            k_list.append(k)
            v_list.append(v)
        q = torch.cat(q_list) # nh, hd
        k = torch.cat(k_list)
        v = torch.cat(v_list)
        self.cache.write_cache(k, v)
        k_all, v_all = self.cache.get_cache() # L, nh, hd
        k_all = k_all.permute(1,0,2).contagious() # nh, L, hd
        v_all = v_all.permute(1,0,2).contagious()
        q_all=q
        
        for i in range(nh):
            q = q_all[i:i+1] # 1, hd
            k = k_all[i] # L, hd
            v = v_all[i]
            s = q @ K.T # 1, hd @ hd, L -> 1, L
            s = s / math.sqrt(head_dim)
            s = torch.softmax(s)
            out = s @ v # 1,L @ L, hd -> 1, hd
            out_list.append(out)
        output = torch.cat(out_list, dim=0)
        return output