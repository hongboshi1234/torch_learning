import torch
import math
L = 20
d = 4

q = torch.randn(L, d)
k = torch.randn(L, d)
v = torch.randn(L, d)
s1 = q @ k.T # (L,L) # Q @ K^T
s2 = s1 / math.sqrt(d)
s3 = torch.softmax(s2, dim=-1)
s4 = s3 @ v # (L,d)
print(s4.shape)
print(f"s4.sum() = {s4.sum()}")
print(f's4:\n{s4}')

# chunked
tile_size = 4
num_tiles = L//4
s1c_list_all=[]
s2c_list_all=[]
s3c_list_all=[]
s4c_list_all=[]

for i in range(num_tiles):
    qc = q[tile_size*i:tile_size*(i+1)]# tile_size, d
    s1c_list=[]
    s2c_list=[]
    s3c_list=[]
    s4c_list=[]
    for j in range(num_tiles):
        kc = k[tile_size*j:tile_size*(j+1)]# tile_size, d
        vc = v[tile_size*j:tile_size*(j+1)]# tile_size, d
        s1c = qc @ kc.T # tile_size, tile_size
        s2c = s1c / math.sqrt(d)
        s1c_list.append(s1c)
        s2c_list.append(s2c)
    s1c_list_all.append(torch.cat(s1c_list, dim=-1))
    s2c_list_all.append(torch.cat(s2c_list, dim=-1))

s1c_all = torch.cat(s1c_list_all, dim=0)
s2c_all = torch.cat(s2c_list_all, dim=0)


import pdb; pdb.set_trace()
print(f's1: {s1-s1c_all}')
# print(f's1c_list: {s1c_list}')
print(f's2: {s2-s2c_all}')
# print(f's2c_list: {s2c_list}')