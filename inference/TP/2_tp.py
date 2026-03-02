import torch
torch.distributed.init_process_group(backend='nccl')
rank = torch.distributed.get_rank()
world_size=torch.distributed.get_world_size() 
device=f'cuda:{rank}'
b = 200
k = 4000
n = 200
assert k%2==0
x = torch.randn(b,k, dtype=torch.float16).to(device) # b, k
w = torch.randn(n,k, dtype=torch.float16).to(device)# n, k
torch.distributed.broadcast(x, src=0)
torch.distributed.broadcast(w, src=0)
y = torch.empty(b,n, dtype=torch.float16).to(device) # b, n
y_baseline = x @ w.T # b,k @ k, n -> b,n

chunk = k//world_size
x_chunk = x[:,chunk*rank:chunk*(rank+1)] # m, k/tp
w_chunk = w[:,chunk*rank:chunk*(rank+1)]  # n, k/tp 
y_chunk =   x_chunk @ w_chunk.transpose(1,0) # b, k/tp @ k.tp, n -> b, n
# print(f'{'_'*100}')
# print(f'rank {rank}, y_chunk.sum():{y_chunk.sum()}')
# print(f"x:{x}")
# print(f"w:{w}")
# print(f"x_chunk:{x_chunk}")
# print(f"w_chunk:{w_chunk}")
# print(f"y_chunk:{y_chunk}")
# print(f"y: {y}")
torch.distributed.all_reduce(y_chunk, op=torch.distributed.ReduceOp.SUM)
print(f'{'_'*100}')
print(f'y_baseline - y_chunk')
# print(f'rank {rank}, y_chunk.sum():{y_chunk.sum()}')
print(f'{y_baseline - y_chunk}')
# print(f'{(y_baseline - y_chunk).sum()}')

chunk = n//world_size
w_chunk = w[chunk*rank:chunk*(rank+1),:] # n/tp, k
y_chunk2 = x @ w_chunk.T # b, k @ k, n/tp -> b, n/tp
y_list = [torch.empty(b, chunk, dtype=torch.float16).to(device) for _ in range(world_size)]
torch.distributed.all_gather(y_list, y_chunk2) 
y_gather = torch.cat(y_list, dim=-1) # b, n
print(f'{'_'*100}')
print(f'y_baseline - y_gather')
print(f'{y_baseline - y_gather}')

torch.distributed.destroy_process_group()

