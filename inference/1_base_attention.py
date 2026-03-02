# without page attetnion
import torch
import math
class Attention(torch.nn.Module):
    def __init__(self, hidden_size, head_dim):
        super().__init__()
        assert hidden_size % head_dim == 0
        self.num_head =int(hidden_size/head_dim)
        self.head_dim = head_dim
        #self.q_proj = [torch.nn.Linear(hidden_size, head_dim) for i in range(self.num_head)]
        #self.k_proj = [torch.nn.Linear(hidden_size, head_dim) for i in range(self.num_head)]
        #self.v_proj = [torch.nn.Linear(hidden_size, head_dim) for i in range(self.num_head)]
        self.q_proj = torch.nn.ModuleList([torch.nn.Linear(hidden_size, head_dim) for _ in range(self.num_head)])
        self.k_proj = torch.nn.ModuleList([torch.nn.Linear(hidden_size, head_dim) for _ in range(self.num_head)])
        self.v_proj = torch.nn.ModuleList([torch.nn.Linear(hidden_size, head_dim) for _ in range(self.num_head)])
    def forward(self, h):
        outputs = []
        with torch.no_grad():
            for i in range(self.num_head):
                q = self.q_proj[i](h) # input_len, head_size
                k = self.k_proj[i](h) 
                v = self.v_proj[i](h)
                s = q @ k.T # input_len, head_size, head_size, iput len -> input len, input_len
                s = s /math.sqrt(self.head_dim)
                s = torch.softmax(s, dim=-1)
                output = s @ v # input_len, input_len   input_len head_size
                outputs.append(output)
            outputs = torch.cat(outputs, dim=-1) #input_len, hidden_size
        return outputs

input_len = 1000
hidden_size = 4096
head_dim = 1024
attention = Attention(4096, 1024)
h = torch.randn(input_len, hidden_size)
h_out = attention(h)
assert h_out.shape == (input_len, hidden_size)
