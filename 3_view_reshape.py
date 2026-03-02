"""
# Run: python3 3_view_reshape.py
# Output:
# before view: torch.Size([2, 3, 4]), x.is_contiguous: True, torch.cuda.memory_allocated: 512
# ------------------------------------------------
# after view: torch.Size([2, 12]), y.is_contiguous: True, torch.cuda.memory_allocated: 512
# x.storage address: 0x7f852fe00000, y.storage address: 0x7f852fe00000
# ------------------------------------------------
# after reshape: torch.Size([2, 12]), z.is_contiguous: True, torch.cuda.memory_allocated: 512
# x.storage address: 0x7f852fe00000, z.storage address: 0x7f852fe00000
# ------------------------------------------------
# after transpose: torch.Size([3, 2, 4]), xt.is_contiguous: False, torch.cuda.memory_allocated: 512
# x.storage address: 0x7f852fe00000, xt.storage address: 0x7f852fe00000
# ------------------------------------------------
# Error with xt_view = xt.view(3,8) : view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
# ------------------------------------------------
# after reshape: torch.Size([3, 8]), xt_reshape.is_contiguous: True, torch.cuda.memory_allocated: 1024
# x.storage address: 0x7f852fe00000, xt_reshape.storage address: 0x7f852fe00200
"""
import torch
x = torch.randn(2,3,4).to("cuda")
print(f"before view: {x.shape}, x.is_contiguous: {x.is_contiguous()}, torch.cuda.memory_allocated: {torch.cuda.memory_allocated()}")
print("\n--------------------------------\n")
y = x.view(2,12)
print(f"after view: {y.shape}, y.is_contiguous: {y.is_contiguous()}, torch.cuda.memory_allocated: {torch.cuda.memory_allocated()}")
print(f"x.storage address: {hex(x.untyped_storage().data_ptr())}, y.storage address: {hex(y.untyped_storage().data_ptr())}")
print("\n--------------------------------\n")

z = x.reshape(2,12)
print(f"after reshape: {z.shape}, z.is_contiguous: {z.is_contiguous()}, torch.cuda.memory_allocated: {torch.cuda.memory_allocated()}")
print(f"x.storage address: {hex(x.untyped_storage().data_ptr())}, z.storage address: {hex(z.untyped_storage().data_ptr())}")

print("\n--------------------------------\n")
xt = x.transpose(0,1)
print(f"after transpose: {xt.shape}, xt.is_contiguous: {xt.is_contiguous()}, torch.cuda.memory_allocated: {torch.cuda.memory_allocated()}")
print(f"x.storage address: {hex(x.untyped_storage().data_ptr())}, xt.storage address: {hex(xt.untyped_storage().data_ptr())}")

print("\n--------------------------------\n")
try:
    xt_view = xt.view(3,8)
except Exception as e:
    print(f"Error with xt_view = xt.view(3,8) : {e}")

print("\n--------------------------------\n")
xt_reshape = xt.reshape(3,8)
print(f"after reshape: {xt_reshape.shape}, xt_reshape.is_contiguous: {xt_reshape.is_contiguous()}, torch.cuda.memory_allocated: {torch.cuda.memory_allocated()}")
print(f"x.storage address: {hex(x.untyped_storage().data_ptr())}, xt_reshape.storage address: {hex(xt_reshape.untyped_storage().data_ptr())}")
"""
Same values, two copies, different memory order. That’s why reshape on a non-contiguous tensor increases memory use.
"""


