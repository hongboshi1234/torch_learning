import os
import torch

assert torch.cuda.is_available()
torch.manual_seed(0)

device = "cuda"

# -----------------------------
# 关键参数：把 copy 和 compute 都弄得够“明显”
# -----------------------------
B = 16144  # batch-like
K = 16144
N = 16144
iters = 12

# pinned host buffers (双缓冲)
h0 = torch.empty((B, K), device="cpu", pin_memory=True, dtype=torch.float16)
h1 = torch.empty((B, K), device="cpu", pin_memory=True, dtype=torch.float16)

# device buffers (双缓冲)
d0 = torch.empty((B, K), device=device, dtype=torch.float16)
d1 = torch.empty((B, K), device=device, dtype=torch.float16)

# 一个固定的权重矩阵，让 compute 是 GEMM（会用 cuBLAS）
W = torch.randn((K, N), device=device, dtype=torch.float16)

copy_stream = torch.cuda.Stream()
compute_stream = torch.cuda.Stream()

# 每个 buffer 对应一个 “copy finished” event
copy_done = [torch.cuda.Event(enable_timing=False), torch.cuda.Event(enable_timing=False)]

def fill_host(x: torch.Tensor, seed: int):
    # 在 CPU 上填充点随机数（开销很小，主要让数据变化）
    g = torch.Generator(device="cpu").manual_seed(seed)
    x.uniform_(-1, 1, generator=g)

# warmup（让 cuBLAS/cuda runtime 初始化完，避免第一次很奇怪）
for _ in range(3):
    y = torch.randn((B, K), device=device, dtype=torch.float16) @ W
torch.cuda.synchronize()

# -----------------------------
# 用 profiler 看 overlap
# -----------------------------
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=False,
    profile_memory=False,
) as prof:
    for i in range(iters):
        buf = i % 2
        h = h0 if buf == 0 else h1
        d = d0 if buf == 0 else d1

        # 1) 准备 host 数据
        with torch.profiler.record_function("fill_host"):
            fill_host(h, seed=1234 + i)

        # 2) 在 copy stream 上做 H2D async copy，并 record event
        with torch.profiler.record_function("copy"):
            with torch.cuda.stream(copy_stream):
                d.copy_(h, non_blocking=True)  # pinned + non_blocking => cudaMemcpyAsync
                copy_done[buf].record(copy_stream)
        with torch.profiler.record_function("compute"):
        # 3) 在 compute stream 上等待“这个 buffer 的 copy 完成”，然后做计算
            with torch.cuda.stream(compute_stream):
                compute_stream.wait_event(copy_done[buf])

                # 让 compute 明显：GEMM + 一个小 reduction
                out = d @ W                         # cublasGemmEx 通常会出现
                s = out.float().sum()               # 防止被优化掉
                # 轻微使用一下结果，避免死代码消除（虽然后面也同步了）
                if i == iters - 1:
                    last = s

    # 等所有 stream 做完
    torch.cuda.synchronize()

# 导出 trace，打开 chrome://tracing 查看时间线
trace_path = "overlap_trace.json"
prof.export_chrome_trace(trace_path)
print(f"Trace exported to: {os.path.abspath(trace_path)}")

# 也顺手打印一个小 summary（可选）
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
