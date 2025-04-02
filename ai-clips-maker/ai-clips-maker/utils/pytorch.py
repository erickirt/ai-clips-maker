"""
Utility functions for PyTorch computations and device handling.
"""

# built-in imports
import logging
import random

# 3rd party imports
import torch
import psutil
import pynvml

# local package imports
from .exceptions import InvalidComputeDeviceError


def get_valid_torch_devices() -> list[str]:
    """
    Returns the valid torch device types supported.

    Returns
    -------
    list[str]
        List of valid torch device strings: ['cpu', 'cuda', 'mps']
    """
    return ["cpu", "cuda", "mps"]


def check_valid_torch_device(device: str) -> str | None:
    """
    Checks whether the given device string is valid.

    Parameters
    ----------
    device: str
        Device string to validate.

    Returns
    -------
    str | None
        None if valid, otherwise a descriptive error message.
    """
    if device not in get_valid_torch_devices():
        return f"Invalid device '{device}'. Valid options: {get_valid_torch_devices()}"
    return None


def is_valid_torch_device(device: str) -> bool:
    """
    Returns whether the given device is valid.

    Returns
    -------
    bool
    """
    return check_valid_torch_device(device) is None


def assert_valid_torch_device(device: str) -> None:
    """
    Raises if device is not valid.

    Raises
    ------
    InvalidComputeDeviceError
    """
    msg = check_valid_torch_device(device)
    if msg:
        raise InvalidComputeDeviceError(msg)


def get_compute_device() -> str:
    """
    Returns the best available compute device.

    Returns
    -------
    str
        'cuda' if available, otherwise 'cpu'
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def check_compute_device_available(device: str) -> str | None:
    """
    Checks if the compute device is both valid and available.

    Parameters
    ----------
    device: str

    Returns
    -------
    str | None
        None if available, otherwise error message.
    """
    msg = check_valid_torch_device(device)
    if msg:
        return msg

    if device == "cuda" and not torch.cuda.is_available():
        return "CUDA device not available."
    if device == "mps" and not torch.backends.mps.is_available():
        return "MPS device not available."
    return None


def is_compute_device_available(device: str) -> bool:
    """
    Returns whether a device is valid and available.

    Returns
    -------
    bool
    """
    return check_compute_device_available(device) is None


def assert_compute_device_available(device: str) -> None:
    """
    Raises if compute device is not valid or not available.

    Raises
    ------
    InvalidComputeDeviceError
    """
    msg = check_compute_device_available(device)
    if msg:
        raise InvalidComputeDeviceError(msg)


def max_magnitude_2d(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Returns the max magnitude values across rows or columns of a 2D tensor.

    Parameters
    ----------
    tensor: torch.Tensor
        A 2D tensor.
    dim: int
        Axis to compute max along. Must be 0 or 1.

    Returns
    -------
    torch.Tensor
        Tensor of max values along the specified axis.
    """
    if not torch.is_tensor(tensor):
        raise TypeError(f"'tensor' must be a torch.Tensor, got {type(tensor)}")
    if dim not in (0, 1):
        raise ValueError(f"'dim' must be 0 or 1, got {dim}")

    abs_tensor = torch.abs(tensor)
    _, indices = torch.max(abs_tensor, dim=dim)

    if dim == 0:
        return tensor[indices, torch.arange(tensor.shape[1])]
    else:
        return tensor[torch.arange(tensor.shape[0]), indices]


def reset_seed(seed: int) -> None:
    """
    Sets the global random seed.

    Parameters
    ----------
    seed: int
    """
    random.seed(seed)
    torch.manual_seed(seed)


def mem_stats() -> dict:
    """
    Retrieves current CPU and GPU memory stats.

    Returns
    -------
    dict
        {
            "gpu": {"total": ..., "free": ...},
            "cpu": {"total": ..., "free": ...}
        }
    """
    gpu_stats = {"total": 0, "free": 0}
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_stats["total"] = info.total
        gpu_stats["free"] = info.free

    cpu_mem = psutil.virtual_memory()
    cpu_stats = {
        "total": cpu_mem.total,
        "free": cpu_mem.available,
    }

    return {"gpu": gpu_stats, "cpu": cpu_stats}


def get_free_cpu_memory() -> int:
    """
    Returns available CPU memory in bytes.

    Returns
    -------
    int
    """
    return psutil.virtual_memory().available
