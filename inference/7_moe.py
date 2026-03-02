import torch
class Expert(torch.nn.Module):
    def __init__(self, hd, inter_size):
        super().__init__()
        self.wu = torch.nn.Linear(hd, inter_size)
        self.wg = torch.nn.Linear(hd, inter_size)
        self.wd = torch.nn.Linear(inter_size, hd)
    
    def forward(self, x):
        u = self.wu(x)
        g = self.wg(x)
        a = u * torch.nn.functional.silu(g)
        return self.wd(a)
    
class MOE(torch.nn.Module):
    def __init__(self, hd, inter_size, ne, ne_per_token):
        super().__init__()
        self.ne=ne
        self.ne_per_token = ne_per_token
        self.hd = hd
        self.experts = torch.nn.ModuleList([Expert(hd, inter_size) for _ in range(ne)])
        self.gate = torch.nn.Linear(hd, ne)
    

    def router(self, x, ne_per_token):
        logits = self.gate(x)
        probs = torch.softmax(logits, dim=-1)
        topk = torch.topk(probs, k =ne_per_token)
        return topk.indices, topk.values
    def forward(self, x):
        experts_ids, weights = self.router(x, self.ne_per_token)
        L, hd = x.shape
        token_ids = torch.arange(L, dtype=int).view(1, -1)
        token_ids = token_ids.repeat( ne_per_token, 1)
        
        token_ids_flat = torch.flatten(token_ids.T)
        experts_flat = torch.flatten(experts_ids)
        weights_flat= torch.flatten(weights)

        permute = torch.argsort(experts_flat) # index of token ids, corresponding to flatten sorted expert [e0, e0, e1, e1, e2, e3]
        expert_permute = experts_flat.index_select(0, permute)
        counts = torch.bincount(expert_permute, minlength=self.ne)
        token_ids_permute = token_ids_flat.index_select(0, permute)
        w_permute = weights_flat.index_select(0, permute)
        x_permute = x.index_select(0,token_ids_permute)  # token0
                                             # token2
                                             # token0
                                             # token1
                                             # token1
                                             # token2 
        # token0 [e0, e1]  [0.8, 0,2] 
        # token1 [e1, e2]  [0.7, 0.3]
        # token2 [e0, e3]  [0.6. 0.4]
        low_index = 0
        # x_out = torch.empty(L, hd)
        x_out = torch.zeros(L, hd)
        for i in range(ne):
            expert = self.experts[i]
            high_index = low_index + counts[i]
            x_slice = x_permute[low_index:high_index]
            x_out_slice = expert(x_slice)  # l, hd
            x_out_slice = x_out_slice * w_permute[low_index:high_index].view(-1,1) # l, hd
            # scatter_index = token_ids_permute[low_index:high_index].view(1,-1)
            scatter_index = token_ids_permute[low_index:high_index]
            # scatter_index = scatter_index.T.repeat(1, hd) 
            x_out.index_add_(0, scatter_index, x_out_slice)
            low_index = high_index
        return x_out

# x_out_slice 0.1 0.2 0.3 0.4
#             0.5 0.6 0.7 0.8 
#  how to update those into token 1 and token 2?
# we need to user [10, 11, 12, 13]
# .               [20, 21, 22, 23]
# remove the columns, it should be [1,1,1,1]
#                                  [2,2,2,2]
input_len =3
hd=16
inter_size=8
ne = 8
ne_per_token=2
torch.manual_seed(0)
moe = MOE(hd, inter_size,ne, ne_per_token)
x = torch.randn(input_len, hd)
x_out = moe(x)
print(f"x_out.shape: {x_out.shape}")