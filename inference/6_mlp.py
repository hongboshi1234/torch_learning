import torch
class MLP(torch.nn.Module):
    def __init__(self, hd, inter_size):
        super().__init__()
        self.wu = torch.nn.Linear(hd, inter_size)
        self.wg = torch.nn.Linear(hd, inter_size)
        self.wd = torch.nn.Linear(inter_size, hd)
    def forward(self, x):# x: L, hd
        u = self.wu(x) # L: 2hd
        g = self.wg(x) # L: 2hd
        import pdb; pdb.set_trace()
        x = u * torch.nn.functional.silu(g) # L,2hd @ L
        return self.wd(x)
    
hd = 1024
inter_size = 2048
input_len = 1000
x = torch.randn(input_len, hd)
mlp = MLP(hd, inter_size)
x_out = mlp(x)
print(f'x_out.shape is {x_out.shape}') 