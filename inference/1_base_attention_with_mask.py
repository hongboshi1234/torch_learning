# without page attetnion
import torch
import math
class Attention(torch.nn.Module):
    def __init__(self, hidden_size, head_dim, max_len):
        super().__init__()
        assert hidden_size % head_dim == 0
        self.num_head =int(hidden_size/head_dim)
        self.head_dim = head_dim
        self.q_proj = torch.nn.ModuleList([torch.nn.Linear(hidden_size, head_dim) for _ in range(self.num_head)])
        self.k_proj = torch.nn.ModuleList([torch.nn.Linear(hidden_size, head_dim) for _ in range(self.num_head)])
        self.v_proj = torch.nn.ModuleList([torch.nn.Linear(hidden_size, head_dim) for _ in range(self.num_head)])
        self.mask = torch.tril(torch.ones(max_len, max_len, dtype=torch.bool))
    def forward(self, h):
        outputs = []
        with torch.no_grad():
            for i in range(self.num_head):
                q = self.q_proj[i](h) # input_len, head_size
                k = self.k_proj[i](h) 
                v = self.v_proj[i](h)
                l = h.shape[0]
                s = q @ k.T # input_len, head_size, head_size, iput len -> input len, input_len
                s = s /math.sqrt(self.head_dim)
                mask = s.masked_fill(~self.mask[:l, :l], float('-inf'))
                s = s-s.max(dim=-1, keepdim=True).values
                s = torch.softmax(s, dim=-1)
                output = s @ v # input_len, input_len   input_len head_size
                outputs.append(output)
            outputs = torch.cat(outputs, dim=-1) #input_len, hidden_size
        return outputs
max_len = 4000
input_len = 1000
hidden_size = 4096
head_dim = 1024
attention = Attention(4096, 1024, max_len)
h = torch.randn(input_len, hidden_size)
h_out = attention(h)
assert h_out.shape == (input_len, hidden_size)
