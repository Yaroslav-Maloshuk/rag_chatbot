from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("pydantic_settings")

from app.core import runtime_device


@pytest.fixture(autouse=True)
def _clear_runtime_cache() -> None:
    runtime_device.get_runtime_device.cache_clear()
    yield
    runtime_device.get_runtime_device.cache_clear()


def test_runtime_device_selects_nvidia_cuda_on_x86(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_device, "get_settings", lambda: SimpleNamespace(model_device="auto"))
    monkeypatch.setattr(runtime_device.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(runtime_device.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runtime_device.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runtime_device.torch.cuda, "is_bf16_supported", lambda: False)
    monkeypatch.setattr(runtime_device.torch.version, "hip", None, raising=False)

    device = runtime_device.get_runtime_device()

    assert device.backend == "cuda"
    assert device.torch_device == "cuda"
    assert device.pipeline_device == 0


def test_runtime_device_selects_amd_rocm_on_x86(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_device, "get_settings", lambda: SimpleNamespace(model_device="auto"))
    monkeypatch.setattr(runtime_device.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(runtime_device.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runtime_device.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runtime_device.torch.version, "hip", "6.0", raising=False)

    device = runtime_device.get_runtime_device()

    assert device.backend == "rocm"
    assert device.torch_device == "cuda"


def test_runtime_device_selects_intel_xpu_on_x86(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_device, "get_settings", lambda: SimpleNamespace(model_device="auto"))
    monkeypatch.setattr(runtime_device.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(runtime_device.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runtime_device.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        runtime_device.torch,
        "xpu",
        SimpleNamespace(is_available=lambda: True),
        raising=False,
    )

    device = runtime_device.get_runtime_device()

    assert device.backend == "xpu"
    assert device.torch_device == "xpu"


def test_runtime_device_selects_mps_on_apple_silicon(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_device, "get_settings", lambda: SimpleNamespace(model_device="auto"))
    monkeypatch.setattr(runtime_device.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(runtime_device.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(runtime_device.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        runtime_device.torch,
        "backends",
        SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
        raising=False,
    )

    device = runtime_device.get_runtime_device()

    assert device.backend == "mps"
    assert device.torch_device == "mps"


def test_runtime_device_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_device, "get_settings", lambda: SimpleNamespace(model_device="auto"))
    monkeypatch.setattr(runtime_device.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(runtime_device.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runtime_device.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        runtime_device.torch,
        "xpu",
        SimpleNamespace(is_available=lambda: False),
        raising=False,
    )

    device = runtime_device.get_runtime_device()

    assert device.backend == "cpu"
    assert device.torch_device == "cpu"


def test_forced_unavailable_device_falls_back_to_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_device, "get_settings", lambda: SimpleNamespace(model_device="cuda"))
    monkeypatch.setattr(runtime_device.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(runtime_device.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runtime_device.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        runtime_device.torch,
        "xpu",
        SimpleNamespace(is_available=lambda: False),
        raising=False,
    )

    device = runtime_device.get_runtime_device()

    assert device.backend == "cpu"
    assert device.torch_device == "cpu"
