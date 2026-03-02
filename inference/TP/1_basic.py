import torch
torch.distributed.init_process_group(backend='nccl')
rank = torch.distributed.get_rank()
world = torch.distributed.get_world_size()
device = f"cuda:{rank}"
print(f"{'*'*100}\n my rank is: {rank}")

m=10
n=20
x = torch.randn(m,n).to(device)
z = [torch.empty(m, n).to(device) for _ in range(world)]
torch.distributed.all_gather(z, x)
z=torch.cat(z)
print(f"{'_'*100}\n z.shape is : {z.shape}")
print(f"x.sum:{x.sum()}, z.sum(): {z.sum()}")

torch.distributed.destroy_process_group()