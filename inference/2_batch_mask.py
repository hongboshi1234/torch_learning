
import torch
import math

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
        # mask = mask.view(max_len,max_len)

    
    def forward(self, h):
        with torch.no_grad():
            # TO imporve, we do not need to save the large outs. we can stack out with previous outs
            outs = []
            l = h.shape[1]
            mask = self.mask[:l, :l]
            for i in range(self.num_head):
                q = self.q_proj[i](h) # (b,L, h) -> (b, L, d)
                k = self.k_proj[i](h)
                v = self.v_proj[i](h)
                b,l,d = q.shape
                k_T = k.permute(0,2,1)
                s = torch.matmul(q, k_T) # (b,l,h) @ (b,h,l) -> (b, l, l)
                s = s/math.sqrt(self.head_dim)  
                s = s.masked_fill(~mask, float('-inf'))
                s = s - s.max(dim=-1,keepdim=True).values
                out = torch.matmul(s, v) # (b, l, l) @ (b, l, d) -> (b, l, d)
                outs.append(out)
            output = torch.cat(outs, dim=-1) # b,l,h
        return output

max_len=8000
hidden_size = 4096
head_dim = 1024
assert hidden_size % head_dim == 0
num_head = int(hidden_size / head_dim)
input_len = 1000
batch = 10
attention = Attention(hidden_size, head_dim, num_head, max_len)
h = torch.randn([batch, input_len, hidden_size])
h_out = attention(h)
assert h_out.shape == (batch, input_len, hidden_size)