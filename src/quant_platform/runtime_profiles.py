from __future__ import annotations

import platform
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None

try:
    import torch_directml
except Exception:  # pragma: no cover - optional dependency
    torch_directml = None

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover - optional dependency
    lgb = None

try:
    import transformers
except Exception:  # pragma: no cover - optional dependency
    transformers = None


SUPPORTED_COMPUTE_TARGETS = {"auto", "cpu", "cuda", "directml"}
SUPPORTED_PRECISION_MODES = {"auto", "fp32", "amp"}
SUPPORTED_SELF_CHECK_MODEL_KINDS = {"pytorch_mlp", "gru", "temporal_cnn"}


@dataclass
class ResolvedRuntime:
    requested_compute_target: str
    resolved_compute_target: str
    provider: str
    backend: str
    device: Any
    requested_precision_mode: str
    precision_mode: str
    amp_enabled: bool
    autocast_device_type: str | None
    notes: list[str]

    def to_summary(self) -> dict[str, Any]:
        return {
            "requested_compute_target": self.requested_compute_target,
            "resolved_compute_target": self.resolved_compute_target,
            "provider": self.provider,
            "backend": self.backend,
            "requested_precision_mode": self.requested_precision_mode,
            "precision_mode": self.precision_mode,
            "amp_enabled": self.amp_enabled,
            "notes": self.notes,
        }


def runtime_capabilities() -> dict[str, Any]:
    devices = [
        {
            "kind": "cpu",
            "available": True,
            "provider": "native",
            "backend": "cpu",
            "label": platform.processor() or "CPU",
            "memory_gb": None,
        }
    ]
    notes: list[str] = []

    if torch is None:
        notes.append("PyTorch is not installed; torch-backed models will fall back to sklearn baselines.")
    else:
        if torch.cuda.is_available():
            backend = "rocm" if getattr(torch.version, "hip", None) else "cuda"
            memory_gb = None
            try:
                memory_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
            except Exception:  # pragma: no cover - best effort only
                memory_gb = None
            devices.append(
                {
                    "kind": "cuda",
                    "available": True,
                    "provider": "torch",
                    "backend": backend,
                    "label": torch.cuda.get_device_name(0),
                    "memory_gb": memory_gb,
                }
            )
        else:
            notes.append("Torch CUDA/ROCm device not available in the current environment.")

    if torch is not None and torch_directml is not None:
        try:
            torch_directml.device()
            devices.append(
                {
                    "kind": "directml",
                    "available": True,
                    "provider": "torch_directml",
                    "backend": "directml",
                    "label": "DirectML device",
                    "memory_gb": None,
                }
            )
        except Exception as exc:  # pragma: no cover - provider specific
            notes.append(f"DirectML detected but failed to initialize: {exc}")
    elif platform.system() == "Windows":
        notes.append("DirectML is not installed; Windows AMD GPU acceleration requires torch-directml.")

    recommended_compute_target = "cpu"
    for preferred in ("cuda", "directml"):
        if any(device["kind"] == preferred and device["available"] for device in devices):
            recommended_compute_target = preferred
            break

    return {
        "python_version": sys.version.split()[0],
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": {
            "torch": {
                "installed": torch is not None,
                "version": getattr(torch, "__version__", None),
                "cuda_version": getattr(getattr(torch, "version", None), "cuda", None) if torch is not None else None,
                "hip_version": getattr(getattr(torch, "version", None), "hip", None) if torch is not None else None,
            },
            "torch_directml": {
                "installed": torch_directml is not None,
                "version": getattr(torch_directml, "__version__", None),
            },
            "lightgbm": {
                "installed": lgb is not None,
                "version": getattr(lgb, "__version__", None),
            },
            "transformers": {
                "installed": transformers is not None,
                "version": getattr(transformers, "__version__", None),
            },
        },
        "devices": devices,
        "recommended_compute_target": recommended_compute_target,
        "supported_compute_targets": sorted(SUPPORTED_COMPUTE_TARGETS),
        "supported_precision_modes": sorted(SUPPORTED_PRECISION_MODES),
        "notes": notes,
    }


def normalize_runtime_settings(settings: dict[str, Any] | None, defaults: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = dict(defaults or {})
    incoming = settings if isinstance(settings, dict) else {}
    resolved.update({key: value for key, value in incoming.items() if value is not None})
    resolved["compute_target"] = _normalize_compute_target(resolved.get("compute_target", "auto"))
    resolved["precision_mode"] = _normalize_precision_mode(resolved.get("precision_mode", "auto"))
    resolved["batch_size"] = _coerce_int(resolved.get("batch_size", 128), minimum=8, maximum=4096, fallback=128)
    resolved["sequence_length"] = _coerce_int(resolved.get("sequence_length", 20), minimum=5, maximum=252, fallback=20)
    resolved["gradient_clip_norm"] = _coerce_float(
        resolved.get("gradient_clip_norm", 1.0),
        minimum=0.1,
        maximum=20.0,
        fallback=1.0,
    )
    return resolved


def resolve_runtime(settings: dict[str, Any] | None, defaults: dict[str, Any] | None = None) -> ResolvedRuntime:
    normalized = normalize_runtime_settings(settings, defaults)
    requested_compute_target = str(normalized["compute_target"])
    requested_precision_mode = str(normalized["precision_mode"])
    notes: list[str] = []
    device = None
    provider = "native"
    backend = "cpu"
    resolved_compute_target = "cpu"

    if torch is not None and requested_compute_target in {"auto", "cuda"} and torch.cuda.is_available():
        resolved_compute_target = "cuda"
        backend = "rocm" if getattr(torch.version, "hip", None) else "cuda"
        provider = "torch"
        device = torch.device("cuda")
    elif torch is not None and requested_compute_target in {"auto", "directml"} and torch_directml is not None:
        try:
            device = torch_directml.device()
            resolved_compute_target = "directml"
            backend = "directml"
            provider = "torch_directml"
        except Exception as exc:  # pragma: no cover - provider specific
            notes.append(f"DirectML initialization failed: {exc}")
    elif requested_compute_target not in {"auto", "cpu"}:
        notes.append(f"Requested compute target `{requested_compute_target}` is unavailable; using CPU.")

    if device is None and torch is not None:
        device = torch.device("cpu")

    precision_mode = "fp32"
    amp_enabled = False
    autocast_device_type: str | None = None
    if requested_precision_mode == "amp":
        if resolved_compute_target == "cuda":
            precision_mode = "amp"
            amp_enabled = True
            autocast_device_type = "cuda"
        else:
            notes.append("AMP was requested but is only enabled on CUDA/ROCm torch backends in the current implementation.")
    elif requested_precision_mode == "auto" and resolved_compute_target == "cuda":
        precision_mode = "amp"
        amp_enabled = True
        autocast_device_type = "cuda"

    return ResolvedRuntime(
        requested_compute_target=requested_compute_target,
        resolved_compute_target=resolved_compute_target,
        provider=provider,
        backend=backend,
        device=device,
        requested_precision_mode=requested_precision_mode,
        precision_mode=precision_mode,
        amp_enabled=amp_enabled,
        autocast_device_type=autocast_device_type,
        notes=notes,
    )


def autocast_context(runtime: ResolvedRuntime):
    if torch is None or not runtime.amp_enabled or not runtime.autocast_device_type:
        return nullcontext()
    return torch.autocast(device_type=runtime.autocast_device_type, dtype=torch.float16)


def runtime_self_check(
    settings: dict[str, Any] | None = None,
    *,
    model_kind: str = "pytorch_mlp",
    input_dim: int = 8,
) -> dict[str, Any]:
    normalized_settings = normalize_runtime_settings(settings, None)
    runtime = resolve_runtime(normalized_settings)
    started_at = time.perf_counter()
    result = {
        "success": False,
        "requested_target_satisfied": False,
        "requested_settings": normalized_settings,
        "resolved_runtime": runtime.to_summary(),
        "checks": {
            "tensor_allocation": False,
            "forward_pass": False,
            "backward_pass": False,
            "optimizer_step": False,
        },
        "errors": [],
        "metrics": {},
        "device_summary": None,
    }

    if torch is None or nn is None:
        result["errors"].append("PyTorch is not installed in the current environment.")
        result["metrics"]["elapsed_ms"] = round((time.perf_counter() - started_at) * 1000.0, 3)
        return result

    requested_model_kind = str(model_kind or "pytorch_mlp")
    if requested_model_kind not in SUPPORTED_SELF_CHECK_MODEL_KINDS:
        requested_model_kind = "pytorch_mlp"
        runtime.notes.append("Unsupported self-check model kind requested; using pytorch_mlp.")
    result["model_kind"] = requested_model_kind

    try:
        device = runtime.device if runtime.device is not None else torch.device("cpu")
        batch_size = int(normalized_settings.get("batch_size", 32))
        sequence_length = int(normalized_settings.get("sequence_length", 20))
        model = _build_self_check_model(requested_model_kind, input_dim=input_dim, hidden_dim=32)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        loss_fn = nn.MSELoss()
        scaler = torch.amp.GradScaler("cuda", enabled=runtime.amp_enabled and runtime.autocast_device_type == "cuda")

        if requested_model_kind in {"gru", "temporal_cnn"}:
            inputs = torch.randn(batch_size, sequence_length, input_dim, device=device)
        else:
            inputs = torch.randn(batch_size, input_dim, device=device)
        targets = torch.randn(batch_size, 1, device=device)
        result["checks"]["tensor_allocation"] = True

        optimizer.zero_grad()
        with autocast_context(runtime):
            predictions = model(inputs)
            result["checks"]["forward_pass"] = True
            loss = loss_fn(predictions, targets)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            result["checks"]["backward_pass"] = True
            scaler.step(optimizer)
            scaler.update()
            result["checks"]["optimizer_step"] = True
        else:
            loss.backward()
            result["checks"]["backward_pass"] = True
            optimizer.step()
            result["checks"]["optimizer_step"] = True

        result["success"] = True
        result["requested_target_satisfied"] = (
            runtime.requested_compute_target == "auto"
            or runtime.requested_compute_target == runtime.resolved_compute_target
        )
        result["metrics"] = {
            "elapsed_ms": round((time.perf_counter() - started_at) * 1000.0, 3),
            "loss": round(float(loss.detach().cpu().item()), 6),
            "output_mean": round(float(predictions.detach().cpu().mean().item()), 6),
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        }
        result["device_summary"] = {
            "device": str(device),
            "provider": runtime.provider,
            "backend": runtime.backend,
            "amp_enabled": runtime.amp_enabled,
        }
        return result
    except Exception as exc:  # pragma: no cover - depends on local hardware/runtime
        result["errors"].append(str(exc))
        result["metrics"]["elapsed_ms"] = round((time.perf_counter() - started_at) * 1000.0, 3)
        return result


def _normalize_compute_target(value: Any) -> str:
    normalized = str(value or "auto").strip().lower()
    if normalized not in SUPPORTED_COMPUTE_TARGETS:
        return "auto"
    return normalized


def _normalize_precision_mode(value: Any) -> str:
    normalized = str(value or "auto").strip().lower()
    if normalized not in SUPPORTED_PRECISION_MODES:
        return "auto"
    return normalized


def _coerce_int(value: Any, minimum: int, maximum: int, fallback: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    return max(minimum, min(maximum, parsed))


def _coerce_float(value: Any, minimum: float, maximum: float, fallback: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    return max(minimum, min(maximum, parsed))


if nn is not None:
    class _SelfCheckMLP(nn.Module):  # type: ignore[misc]
        def __init__(self, input_dim: int, hidden_dim: int):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, inputs):
            return self.network(inputs)


    class _SelfCheckGRU(nn.Module):  # type: ignore[misc]
        def __init__(self, input_dim: int, hidden_dim: int):
            super().__init__()
            self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
            self.head = nn.Linear(hidden_dim, 1)

        def forward(self, inputs):
            outputs, _ = self.gru(inputs)
            return self.head(outputs[:, -1, :])


    class _SelfCheckTemporalCNN(nn.Module):  # type: ignore[misc]
        def __init__(self, input_dim: int, hidden_dim: int):
            super().__init__()
            reduced_dim = max(8, hidden_dim // 2)
            self.encoder = nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(hidden_dim, reduced_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Linear(reduced_dim, 1)

        def forward(self, inputs):
            encoded = self.encoder(inputs.transpose(1, 2)).squeeze(-1)
            return self.head(encoded)
else:
    class _SelfCheckMLP:
        pass


    class _SelfCheckGRU:
        pass


    class _SelfCheckTemporalCNN:
        pass


def _build_self_check_model(model_kind: str, input_dim: int, hidden_dim: int):
    if model_kind == "gru":
        return _SelfCheckGRU(input_dim=input_dim, hidden_dim=hidden_dim)
    if model_kind == "temporal_cnn":
        return _SelfCheckTemporalCNN(input_dim=input_dim, hidden_dim=hidden_dim)
    return _SelfCheckMLP(input_dim=input_dim, hidden_dim=hidden_dim)
