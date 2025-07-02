# Description: GPU utility functions.

import torch

def is_rocm_system():
    """Check if the system is using ROCm."""
    return torch.version.hip is not None

def get_gpu_device_name():
    """Get the name of the GPU device."""
    if is_rocm_system():
        return torch.cuda.get_device_name(0)
    else:
        return torch.cuda.get_device_name(0)
