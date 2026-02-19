import os, torch
assert torch.cuda.is_available()
device="cuda"
torch.manual_seed(0)

B=6144; K=6144; N=6144
iters=10

# 2 个 pinned host buffer（提前填好，避免 CPU uniform_ 抢占时间线）
h = [torch.empty((B,K), pin_memory=True, dtype=torch.float16) for _ in range(2)]
for i in range(2):
    h[i].uniform_(-1, 1)

# 2 个 device buffer
d = [torch.empty((B,K), device=device, dtype=torch.float16) for _ in range(2)]

W = torch.randn((K,N), device=device, dtype=torch.float16)

copy_stream = torch.cuda.Stream()
compute_stream = torch.cuda.Stream()
copy_done = [torch.cuda.Event(False), torch.cuda.Event(False)]

# warmup
for _ in range(3):
    _ = (torch.randn((B,K), device=device, dtype=torch.float16) @ W).sum()
torch.cuda.synchronize()

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
) as prof:

    # 先把第 0 批 copy 进去（prefetch）
    with torch.profiler.record_function("test record copy 0"):
        with torch.cuda.stream(copy_stream):
            d[0].copy_(h[0], non_blocking=True)
            copy_done[0].record(copy_stream)

    for i in range(1, iters):
        cur = i % 2
        prev = (i - 1) % 2

        # 1) 先启动下一批 copy（cur）
        with torch.profiler.record_function("test record iter %d" % iter):
            with torch.cuda.stream(copy_stream):
                d[cur].copy_(h[cur], non_blocking=True)
                copy_done[cur].record(copy_stream)

        # 2) 同时在另一个 stream 里算上一批（prev）
        with torch.profiler.record_function("test record compute %d" % iter):
            with torch.cuda.stream(compute_stream):
                compute_stream.wait_event(copy_done[prev])
                out = d[prev] @ W
                s = out.float().sum()
                if i == iters - 1:
                    last = s

    torch.cuda.synchronize()

trace_path="overlap_trace.json"
prof.export_chrome_trace(trace_path)
print("trace:", os.path.abspath(trace_path))
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))

