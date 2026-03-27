from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import logging
from pathlib import Path
import platform

import torch

from app.core.config import get_settings

logger = logging.getLogger(__name__)
X86_ARCHS = {"x86_64", "amd64", "i386", "i686"}


@dataclass(frozen=True)
class RuntimeDevice:
    backend: str
    torch_device: str
    pipeline_device: int | str
    sentence_transformers_device: str
    torch_dtype: torch.dtype


def _build_runtime_device_for_backend(backend: str) -> RuntimeDevice:
    if backend == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return RuntimeDevice(
            backend="cuda",
            torch_device="cuda",
            pipeline_device=0,
            sentence_transformers_device="cuda",
            torch_dtype=dtype,
        )

    if backend == "rocm":
        return RuntimeDevice(
            backend="rocm",
            torch_device="cuda",
            pipeline_device=0,
            sentence_transformers_device="cuda",
            torch_dtype=torch.float16,
        )

    if backend == "xpu":
        return RuntimeDevice(
            backend="xpu",
            torch_device="xpu",
            pipeline_device="xpu",
            sentence_transformers_device="xpu",
            torch_dtype=torch.float16,
        )

    if backend == "mps":
        return RuntimeDevice(
            backend="mps",
            torch_device="mps",
            pipeline_device="mps",
            sentence_transformers_device="mps",
            torch_dtype=torch.float16,
        )

    return RuntimeDevice(
        backend="cpu",
        torch_device="cpu",
        pipeline_device=-1,
        sentence_transformers_device="cpu",
        torch_dtype=torch.float32,
    )


def _is_backend_available(backend: str) -> bool:
    if backend == "cpu":
        return True
    if backend == "cuda":
        return torch.cuda.is_available() and not bool(getattr(torch.version, "hip", None))
    if backend == "rocm":
        return torch.cuda.is_available() and bool(getattr(torch.version, "hip", None))
    if backend == "xpu":
        xpu = getattr(torch, "xpu", None)
        return xpu is not None and xpu.is_available()
    if backend == "mps":
        mps = getattr(torch.backends, "mps", None)
        return mps is not None and mps.is_available()
    return False


@lru_cache(maxsize=1)
def get_runtime_device() -> RuntimeDevice:
    settings = get_settings()
    forced_device = settings.model_device.strip().lower()

    # Optional manual override from .env (MODEL_DEVICE=cpu|cuda|xpu|mps|auto).
    if forced_device and forced_device != "auto":
        if _is_backend_available(forced_device):
            return _build_runtime_device_for_backend(forced_device)
        logger.warning("Forced MODEL_DEVICE=%s is unavailable, falling back to auto detection", forced_device)

    machine = platform.machine().lower()
    system = platform.system().lower()
    is_x86 = machine in X86_ARCHS

    # x86 GPU branch:
    # - NVIDIA via CUDA
    # - AMD via ROCm (HIP API)
    # - Intel GPU via XPU
    if is_x86:
        if torch.cuda.is_available():
            if getattr(torch.version, "hip", None):
                return _build_runtime_device_for_backend("rocm")
            return _build_runtime_device_for_backend("cuda")

        xpu = getattr(torch, "xpu", None)
        if xpu is not None and xpu.is_available():
            return _build_runtime_device_for_backend("xpu")

    # Apple Silicon branch (includes Mac mini M4 Pro).
    if system == "darwin" and machine in {"arm64", "aarch64"}:
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return _build_runtime_device_for_backend("mps")

    if system == "linux" and machine in {"arm64", "aarch64"} and Path("/.dockerenv").exists():
        logger.info("MPS is unavailable in Linux containers. Run API/worker natively on macOS to use Apple GPU.")

    return _build_runtime_device_for_backend("cpu")
