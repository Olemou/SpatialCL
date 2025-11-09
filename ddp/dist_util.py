import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def destroy_dist():
    """Destroy the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()
        print("Distributed process group destroyed.")

def is_dist() -> bool:
    """Check if distributed training is initialized."""
    return dist.is_available() and dist.is_initialized()

def is_master() -> bool:
    """True only for rank 0 or single GPU."""
    return (not is_dist()) or (dist.get_rank() == 0)

def get_world_size() -> int:
    """Return total number of processes."""
    return dist.get_world_size() if is_dist() else 1

def get_rank() -> int:
    """Return current process rank."""
    return dist.get_rank() if is_dist() else 0

# =============================================================================
# COMMUNICATION HELPERS
# =============================================================================
def dist_gather(obj):
    """Gather arbitrary Python object from all ranks."""
    if not is_dist():
        return [obj]
    obj_all = [None for _ in range(get_world_size())]
    dist.all_gather_object(obj_all, obj)
    return obj_all

# =============================================================================
# MODEL WRAPPING
# =============================================================================
def wrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Wrap model in DDP only if distributed is initialized."""
    if not is_dist():
        return model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    ddp_model = DDP(
        model.to(torch.cuda.current_device()),
        device_ids=[torch.cuda.current_device()],
        output_device=torch.cuda.current_device(),
        find_unused_parameters=False
    )
    return ddp_model

def get_model_device(model: torch.nn.Module) -> torch.device:
    """Get the current device of a model, even if wrapped in DDP."""
    if isinstance(model, DDP):
        return next(model.module.parameters()).device
    else:
        return next(model.parameters()).device


