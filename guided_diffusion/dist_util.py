"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
import torch as th
import torch.distributed as dist
from . import logger
# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3


def setup_dist(devices=""):
    """
    Setup a distributed process group (torch.distributed only).
    This function is compatible with both single-process (single GPU/CPU)
    and multi-process (DDP launched via torchrun) execution.
    """
    if dist.is_initialized():
        return

    # Optional: honor a user-specified visible devices list if provided.
    # We DO NOT slice per-rank here; torchrun sets LOCAL_RANK and we bind later.
    if devices:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", devices)

    backend = "nccl" if th.cuda.is_available() else "gloo"

    # If launched by torchrun, these envs are already set.
    # If running single-process, initialize a local default group.
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = "1"
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = "0"

    # Initialize process group via env://
    dist.init_process_group(backend=backend, init_method="env://")

    # Bind CUDA device to LOCAL_RANK if available
    try:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    except ValueError:
        local_rank = 0
    if th.cuda.is_available():
        th.cuda.set_device(local_rank)


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file while avoiding redundant remote fetches in multi-process setups.
    Rank 0 reads the bytes via blobfile; other ranks receive the bytes via torch.distributed broadcast.
    Works with NCCL (CUDA tensors) and Gloo (CPU tensors).
    """
    # If no distrib has been initialized, just read locally.
    if not dist.is_available() or not dist.is_initialized():
        return th.load(path, **kwargs)

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    rank = dist.get_rank()

    if rank == 0:
        with bf.BlobFile(path, "rb") as f:
            data_bytes = f.read()
        n = th.tensor([len(data_bytes)], dtype=th.long, device=device)
    else:
        data_bytes = None
        n = th.zeros(1, dtype=th.long, device=device)

    # Broadcast length first
    dist.broadcast(n, src=0)
    n_int = int(n.item())

    # Prepare buffer and broadcast the raw bytes
    buf = th.empty(n_int, dtype=th.uint8, device=device)
    if rank == 0 and n_int > 0:
        # Move raw bytes to tensor
        buf.copy_(th.frombuffer(memoryview(data_bytes), dtype=th.uint8).to(device))
    dist.broadcast(buf, src=0)

    # Reconstruct bytes and load state_dict
    data_bytes_out = buf.cpu().numpy().tobytes()
    return th.load(io.BytesIO(data_bytes_out), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
