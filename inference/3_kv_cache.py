
import torch
import math

# kv cache with batch = 1, with kv cache re use
# better keep kv as a class inside attention! no need to handle layer id
# better to use cur_len, no need to pass position id, for interview purpose
# better to assume no padding and all batch has the same input len? need to improve as sedond iteration

# prefill and decode should be seperated ! because mask operation is very different! 
# there are still quite bit issues with this.

@dataclass
class KVCache:
    k: torch.Tensor # B, L, num_head, head_size
    v: torch.Tensor
    cur_len: int = 0
    
class Attention(torch.nn.Module):
    def __init__(self,hidden_size, head_dim, num_head, max_len):
        super().__init__()
        self.q_proj = torch.nn.ModuleList([torch.nn.Linear(hidden_size, head_dim) for _ in range(num_head)])
        self.k_proj = torch.nn.ModuleList([torch.nn.Linear(hidden_size, head_dim) for _ in range(num_head)])
        self.v_proj = torch.nn.ModuleList([torch.nn.Linear(hidden_size, head_dim) for _ in range(num_head)])
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_head = num_head
        self.mask = torch.tril(torch.ones(max_len,max_len, dtype=torch.bool))
        k = torch.empty(batch_size, max_len, num_head, head_size)
        v = torch.empty(batch_size, max_len, num_head, head_size)
        self.kv = KVCache(k=k, v=v, cur_len=0) # B, L, head_id, head_size 
        # mask = mask.view(max_len,max_len)

    @torch.no_grad() 
    def forward(self, h):
            outs = []
            l = h.shape[1]
            mask = self.mask[:l, :l]
            for i in range(self.num_head):
                    # start prefillilng
                q = self.q_proj[i](h) 
                k = self.k_proj[i](h)
                v = self.v_proj[i](h)
                b,l,d = q.shape
                if self.kv.cur_len==0:
                    self.kv.k[:,:l+1,:,:] = k 
                    self.kv.v[:,:l+1,:,:] = v
                    self.cur_len = l
                else:
                    l = self.kv.cur_len + 1
                    self.kv.cur_len += 1
                    self.kv.k[:,l+1,:,:] = k
                    self.kv.v[:,l+1,:,:] = v
                    k = self.kv.k[:,:l+1]
                    v = self.kv.v[:,:l+1]
                k_T = k.permute(0,2,1)
                s = torch.matmul(q, k_T) # (b,l,h) @ (b,h,l) -> (b, l, l)
                s = s/math.sqrt(self.head_dim)  
                s = s.masked_fill(~mask, float('-inf'))
                s = s - s.max(dim=-1,keepdim=True).values
                s = torch.softmax(s, dim=-1)
                out = torch.matmul(s, v) # (b, l, l) @ (b, l, d) -> (b, l, d)
                outs.append(out)
            output = torch.cat(outs, dim=-1) # b,l,h
            return output
    # def forward_decoding(self, h): # 2, B, L, head_id, head_size
    #     with torch.no_grad():
    #         # TO imporve, we do not need to save the large outs. we can stack out with previous outs
    #         outs = []
    #         l = h.shape[1]
    #         mask = self.mask[:l, :l]
    #         for i in range(self.num_head):
    #                 # start prefillilng
    #             q = self.q_proj[i](h) # (b,L, h) -> (b, L, d)
    #             k = self.k_proj[i](h)
    #             v = self.v_proj[i](h)
    #             b,l,d = q.shape
    #             l = self.kv.cur_len + 1
    #             self.kv.cur_len += 1
    #             self.kv.k[:,l+1,:,:] = k
    #             self.kv.v[:,l+1,:,:] = v
    #             k = self.kv.k[:,:l+1]
    #             v = self.kv.v[:,:l+1]
    #             k_T = k.permute(0,2,1)
    #             s = torch.matmul(q, k_T) # (b,l,h) @ (b,h,l) -> (b, l, l)
    #             s = s/math.sqrt(self.head_dim)  
    #             s = s.masked_fill(~mask, float('-inf'))
    #             s = s - s.max(dim=-1,keepdim=True).values
    #             out = torch.matmul(s, v) # (b, l, l) @ (b, l, d) -> (b, l, d)
    #             outs.append(out)
    #         output = torch.cat(outs, dim=-1) # b,l,h
        return output

