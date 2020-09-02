


import os
import torch
from transformers import BertForPreTraining

import os
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForSeq2SeqLM

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()

BART_TINY = "sshleifer/bart-tiny-random"
from transformers import AutoModelForSeq2SeqLM

def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    model = AutoModelForSeq2SeqLM.from_pretrained(BART_TINY).to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()


import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def evaluate(gpu, args):
    rank = gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    torch.cuda.set_device(gpu)
    model = DDP(model)




    pass


def main():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.distributed.init_process_group(backend="nccl", rank=0, world_size=2)

    mp.spawn(evaluate, nprocs=2, args=(args,))

model = BertForPreTraining.from_pretrained('bert-base-uncased').cuda()
model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
model = torch.nn.parallel.DistributedDataParallel()
outputs = model(torch.tensor([[1, 2, 3]]).cuda())
outputs[0].sum().backward()

# check parameters with no grad
for n, p in model.named_parameters():
    if p.grad is None and p.requires_grad is True:
        print('Parameter not used:', n, p.shape)  # prints unused parameters. Remove them from your model

outputs = model(torch.tensor([[1, 2, 3]]).cuda())
outputs[0].sum().backward()  # if any unused parameters remain in your model, this will break.


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)
