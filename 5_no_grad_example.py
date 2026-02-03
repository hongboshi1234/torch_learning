"""
Example: torch.no_grad() vs without - memory and behavior difference

# Run: python3 5_no_grad_example.py
# Output:
# === WITHOUT torch.no_grad() (builds computation graph) ===
#   Memory before: 3.81 MB
#   Memory after forward: 39.63 MB
#   Peak memory: 43.44 MB
#   y.requires_grad: True
#   y.grad_fn: <AddBackward0 object at 0x7f7734bcf6a0>
#
# === WITH torch.no_grad() (no graph) ===
#   Memory before: 39.63 MB
#   Memory after forward: 43.44 MB
#   Peak memory: 47.26 MB
#   y2.requires_grad: False
#   y2.grad_fn: None
#
# === Key difference: backward() ===
#   Without no_grad: y.backward() works (we have the graph)
#   With no_grad:    y2.backward() would FAIL - no graph to backprop through!
#   y.backward() succeeded, x.grad shape: torch.Size([1000, 1000])
#   y2.backward() failed: RuntimeError - no computation graph
"""
import torch

# Create input that requires gradients (e.g., during training)
x = torch.randn(1000, 1000, device="cuda", requires_grad=True)

print("=== WITHOUT torch.no_grad() (builds computation graph) ===")
torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()
mem_before = torch.cuda.memory_allocated()

y = x @ x.T + x  # Matrix multiply + add - builds graph for backprop
mem_after = torch.cuda.memory_allocated()
peak = torch.cuda.max_memory_allocated()

print(f"  Memory before: {mem_before / 1024**2:.2f} MB")
print(f"  Memory after forward: {mem_after / 1024**2:.2f} MB")
print(f"  Peak memory: {peak / 1024**2:.2f} MB")
print(f"  y.requires_grad: {y.requires_grad}")
print(f"  y.grad_fn: {y.grad_fn}")  # Has grad_fn - graph was built

print("\n=== WITH torch.no_grad() (no graph) ===")
torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()
mem_before = torch.cuda.memory_allocated()

with torch.no_grad():
    y2 = x @ x.T + x  # Same computation, but no graph

mem_after = torch.cuda.memory_allocated()
peak = torch.cuda.max_memory_allocated()

print(f"  Memory before: {mem_before / 1024**2:.2f} MB")
print(f"  Memory after forward: {mem_after / 1024**2:.2f} MB")
print(f"  Peak memory: {peak / 1024**2:.2f} MB")
print(f"  y2.requires_grad: {y2.requires_grad}")
print(f"  y2.grad_fn: {y2.grad_fn}")  # None - no graph

print("\n=== Key difference: backward() ===")
print("  Without no_grad: y.backward() works (we have the graph)")
print("  With no_grad:    y2.backward() would FAIL - no graph to backprop through!")

# Demonstrate backward works only for y
try:
    y.sum().backward()
    print(f"  y.backward() succeeded, x.grad shape: {x.grad.shape}")
except Exception as e:
    print(f"  y.backward() failed: {e}")

try:
    y2.sum().backward()
    print("  y2.backward() succeeded")
except Exception as e:
    print(f"  y2.backward() failed: {type(e).__name__} - no computation graph")
