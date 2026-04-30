"""Device and dtype selection for local inference."""

from __future__ import annotations

from enum import Enum

import torch


class Device(str, Enum):
    auto = "auto"
    cuda = "cuda"
    mps = "mps"
    cpu = "cpu"


class Dtype(str, Enum):
    auto = "auto"
    bf16 = "bf16"
    fp16 = "fp16"
    fp32 = "fp32"


_DTYPE_MAP = {
    Dtype.bf16: torch.bfloat16,
    Dtype.fp16: torch.float16,
    Dtype.fp32: torch.float32,
}


def resolve(device_opt: Device, dtype_opt: Dtype) -> tuple[str, torch.dtype]:
    """Resolve `auto` device/dtype to concrete values based on hardware."""
    if device_opt is Device.auto:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = device_opt.value

    if dtype_opt is Dtype.auto:
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
    else:
        dtype = _DTYPE_MAP[dtype_opt]
    return device, dtype


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")
