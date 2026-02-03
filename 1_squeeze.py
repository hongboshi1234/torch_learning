"""
# Run: python3 1_squeeze.py
# Output:
# before unsqueeze: torch.Size([2, 3, 4]), memory: 96 bytes
#   [GPU total allocated: 512 bytes]
# after unsqueeze(0): torch.Size([1, 2, 3, 4]), memory: 96 bytes
#   [GPU total allocated: 512 bytes]
# after unsqueeze(-1): torch.Size([1, 2, 3, 4, 1]), memory: 96 bytes
#   [GPU total allocated: 512 bytes]
# after squeeze(0): torch.Size([2, 3, 4, 1]), memory: 96 bytes
#   [GPU total allocated: 512 bytes]
# after squeeze(-1): torch.Size([2, 3, 4]), memory: 96 bytes
#   [GPU total allocated: 512 bytes]
#
# final shape of x: torch.Size([1, 2, 3, 4, 1]), y: torch.Size([2, 3, 4])
# x.storage address: 0x7f1bd3e00000
# y.storage address: 0x7f1bd3e00000
# before modifying x[0,0,0,0,0]: -0.31804001331329346,  y[0,0,0]: -0.31804001331329346
# after modifying x[0,0,0,0,0]: 100.0,  y[0,0,0]: 100.0
"""
import torch

def tensor_memory(t):
    """Return tensor size in bytes."""
    return t.element_size() * t.nelement()

def storage_ptr(t):
    """Return the underlying storage memory address."""
    return t.untyped_storage().data_ptr()

def total_allocated():
    """Return total allocated memory (GPU if available, else N/A for CPU)."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated()
    return None

x = torch.randn(2, 3, 4)
x=x.to("cuda")
print(f"before unsqueeze: {x.shape}, memory: {tensor_memory(x):,} bytes")
if total_allocated() is not None:
    print(f"  [GPU total allocated: {total_allocated():,} bytes]")

x = x.unsqueeze(0)
print(f"after unsqueeze(0): {x.shape}, memory: {tensor_memory(x):,} bytes")
if total_allocated() is not None:
    print(f"  [GPU total allocated: {total_allocated():,} bytes]")

x = x.unsqueeze(-1)
print(f"after unsqueeze(-1): {x.shape}, memory: {tensor_memory(x):,} bytes")
if total_allocated() is not None:
    print(f"  [GPU total allocated: {total_allocated():,} bytes]")

y = x.squeeze(0)
print(f"after squeeze(0): {y.shape}, memory: {tensor_memory(y):,} bytes")
if total_allocated() is not None:
    print(f"  [GPU total allocated: {total_allocated():,} bytes]")

y = y.squeeze(-1)
print(f"after squeeze(-1): {y.shape}, memory: {tensor_memory(y):,} bytes")
if total_allocated() is not None:
    print(f"  [GPU total allocated: {total_allocated():,} bytes]")

print(f"\nfinal shape of x: {x.shape}, y: {y.shape}")
print(f"x.storage address: {hex(storage_ptr(x))}")
print(f"y.storage address: {hex(storage_ptr(y))}")

# check if a and y share storage:
print(f"before modifying x[0,0,0,0,0]: {x[0,0,0,0,0]},  y[0,0,0]: {y[0,0,0]}")   
x[0,0,0] = 100
print(f"after modifying x[0,0,0,0,0]: {x[0,0,0,0,0]},  y[0,0,0]: {y[0,0,0]}")

"""
No extra memory is allocated for the tensor data in either case. Both unsqueeze and squeeze return views that share the same underlying storage as the original tensor.
x = x.unsqueeze(0)
unsqueeze adds a dimension of size 1; it does not copy data.
The new tensor is a view of the same storage.
Element count stays the same (e.g. 2×3×4 → 1×2×3×4, still 24 elements).
Only a small amount of metadata (shape, strides) is created; the data buffer is reused.
y = x.squeeze(0)
squeeze removes dimensions of size 1; it also does not copy data.
The result is again a view of the same storage.
Element count stays the same (e.g. 1×2×3×4 → 2×3×4, still 24 elements).
Same story: no new data allocation, only metadata.


Direction	Mapping
the address printed is the virtual cpu address
Virtual → Physical	One-to-one (at a given time)
Physical → Virtual	One-to-many (possible)

"""