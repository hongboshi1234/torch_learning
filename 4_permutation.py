# transpose(dim0, dim1)
# permute(*dims) 任意维度重排， 大多数时候返回非连续试图， non contigous, Stride chagned.
# contiguous() copy 成 连续内存布局, both vitual memory ptr  x.untyped_storage().data_ptr()and physical HBM address will change
"""
# Run: python3 4_permutation.py
# Output:
# before transpose: torch.Size([2, 3, 4]), x.is_contiguous: True, torch.cuda.memory_allocated: 512
# x.storage address: 0x7f4903e00000
# ------------------------------------------------
# after transpose: torch.Size([3, 2, 4]), xt.is_contiguous: False, torch.cuda.memory_allocated: 512
# x.storage address: 0x7f4903e00000, xt.storage address: 0x7f4903e00000
# ------------------------------------------------
# after permute: torch.Size([3, 2, 4]), x_permute.is_contiguous: False, torch.cuda.memory_allocated: 512
# ------------------------------------------------
# after x_permute.contiguous(): torch.Size([3, 2, 4]), x_permute.is_contiguous: False, torch.cuda.memory_allocated: 1024
# x.storage address: 0x7f4903e00000, x_permute.storage address: 0x7f4903e00000
# ------------------------------------------------
# ------------------------------------------------
# after reshape: torch.Size([2, 3, 4]), z.is_contiguous: True, torch.cuda.memory_allocated: 1536
"""

import torch

x = torch.randn(2,3,4).to("cuda")
print(f"before transpose: {x.shape}, x.is_contiguous: {x.is_contiguous()}, torch.cuda.memory_allocated: {torch.cuda.memory_allocated()}")
print(f"x.storage address: {hex(x.untyped_storage().data_ptr())}")
print("\n--------------------------------\n")
xt = x.transpose(0,1)
print(f"after transpose: {xt.shape}, xt.is_contiguous: {xt.is_contiguous()}, torch.cuda.memory_allocated: {torch.cuda.memory_allocated()}")
print(f"x.storage address: {hex(x.untyped_storage().data_ptr())}, xt.storage address: {hex(xt.untyped_storage().data_ptr())}")

print("\n--------------------------------\n")
x_permute= x.permute(1,0,2)
print(f"after permute: {x_permute.shape}, x_permute.is_contiguous: {x_permute.is_contiguous()}, torch.cuda.memory_allocated: {torch.cuda.memory_allocated()}")

print("\n--------------------------------\n")
y = x_permute.contiguous()

print(f"after x_permute.contiguous(): {x_permute.shape}, x_permute.is_contiguous: {x_permute.is_contiguous()}, torch.cuda.memory_allocated: {torch.cuda.memory_allocated()}")
print(f"x.storage address: {hex(x.untyped_storage().data_ptr())}, x_permute.storage address: {hex(x_permute.untyped_storage().data_ptr())}")

print("\n--------------------------------\n")
try:
    x_permute_view = x_permute.view(3,2,4)
except Exception as e:
    print(f"Error with x_permute_view = x_permute.view(3,2,4) : {e}")

print("\n--------------------------------\n")
x_permute = x_permute.reshape(2, 3, 4)
z=x_permute
print(f"after reshape: {z.shape}, z.is_contiguous: {z.is_contiguous()}, torch.cuda.memory_allocated: {torch.cuda.memory_allocated()}")

