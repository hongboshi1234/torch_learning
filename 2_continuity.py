"""
# Run: python3 2_continuity.py
# Output:
# before x.T, x.continous: True
# total allocated memory: 512
# after y=x.T, y.is_contiguous: False
# x.storage address: 0x7f8d7fe00000, y.storage address: 0x7f8d7fe00000
# total allocated memory: 512
"""
import torch
x = torch.randn(2,3,4).to("cuda")
print(f"before x.T, x.continous: {x.is_contiguous()}")
print(f"total allocated memory: {torch.cuda.memory_allocated()}")
y = x.T
print(f"after y=x.T, y.is_contiguous: {y.is_contiguous()}")
print(f"x.storage address: {hex(x.untyped_storage().data_ptr())}, y.storage address: {hex(y.untyped_storage().data_ptr())}")
print(f"total allocated memory: {torch.cuda.memory_allocated()}")



"""
I am trying to understand what's going on here. x has address 0x7f063be00000, this address mapped to a physical memory on GPU, let's call that p1

then y =x.T, then y and x shares the same virtual addresss, which also point to p1.

then why x is conguous but y is not. I thoght congous means continous on physical memory?
Contiguous in PyTorch does not mean “one contiguous block of physical memory.” Both x and y use the same block of physical memory (same storage).
x (2, 3, 4): x[i,j,k] is at offset i*12 + j*4 + k → sequential in memory.
y = x.T (4, 3, 2): y[i,j,k] corresponds to x[k,j,i], so its strides are reversed. Iterating y in logical order does not walk memory sequentially.


Physical memory (one block, shared by x and y):
[ x[0,0,0] x[0,0,1] x[0,0,2] x[0,0,3] | x[0,1,0] ... | ... | x[1,2,3] ]
   ↑           ↑
   y[0,0,0]   y[1,0,0]  ← y's "first" and "second" elements are NOT adjacent!
"""