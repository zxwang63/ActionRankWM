from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from actionrankwm.io import atomic_write_json, digest_file, read_json


DEFAULT_TARGET_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z",
    "linear_attn.out_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        factory = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.lora_A = nn.Parameter(torch.empty((rank, base.in_features), **factory))
        self.lora_B = nn.Parameter(torch.zeros((base.out_features, rank), **factory))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_output = self.base(inputs)
        update = F.linear(F.linear(self.dropout(inputs), self.lora_A), self.lora_B)
        return base_output + update.to(base_output.dtype) * self.scaling


@dataclass(frozen=True)
class LoRAModuleSpec:
    name: str
    in_features: int
    out_features: int
    rank: int
    alpha: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "rank": self.rank,
            "alpha": self.alpha,
        }


def _set_submodule(root: nn.Module, name: str, module: nn.Module) -> None:
    parent_name, _, child_name = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    if child_name not in parent._modules:
        raise ValueError(f"Cannot replace missing submodule {name}")
    parent._modules[child_name] = module


def _matches_qwen_language_target(
    name: str,
    *,
    target_prefix: str,
    target_suffixes: Sequence[str],
) -> bool:
    if not name.startswith(target_prefix):
        return False
    return any(name.endswith(suffix) for suffix in target_suffixes)


def inject_lora(
    model: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    target_prefix: str = "model.layers.",
    target_suffixes: Sequence[str] = DEFAULT_TARGET_SUFFIXES,
    exact_module_names: Iterable[str] | None = None,
) -> list[LoRAModuleSpec]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    exact = set(exact_module_names) if exact_module_names is not None else None
    candidates = list(model.named_modules())
    specs: list[LoRAModuleSpec] = []
    for name, module in candidates:
        selected = (
            name in exact
            if exact is not None
            else _matches_qwen_language_target(
                name,
                target_prefix=target_prefix,
                target_suffixes=target_suffixes,
            )
        )
        if not selected:
            continue
        if isinstance(module, LoRALinear):
            raise ValueError(f"LoRA already injected at {name}")
        if not isinstance(module, nn.Linear):
            raise TypeError(f"LoRA target is not nn.Linear: {name}/{type(module).__name__}")
        wrapped = LoRALinear(
            module,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        _set_submodule(model, name, wrapped)
        specs.append(
            LoRAModuleSpec(
                name=name,
                in_features=module.in_features,
                out_features=module.out_features,
                rank=rank,
                alpha=alpha,
            )
        )
    if exact is not None and {spec.name for spec in specs} != exact:
        missing = sorted(exact - {spec.name for spec in specs})
        raise ValueError(f"Missing exact LoRA modules: {missing}")
    if not specs:
        raise ValueError("No LoRA target modules matched")
    return sorted(specs, key=lambda item: item.name)


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    parameters = [
        parameter
        for module in model.modules()
        if isinstance(module, LoRALinear)
        for parameter in (module.lora_A, module.lora_B)
    ]
    if not parameters:
        raise ValueError("The model contains no LoRA parameters")
    return parameters


def trainable_parameter_summary(model: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def _adapter_state(model: nn.Module) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        state[f"{name}.lora_A"] = module.lora_A.detach().cpu().contiguous()
        state[f"{name}.lora_B"] = module.lora_B.detach().cpu().contiguous()
    if not state:
        raise ValueError("Cannot save an empty LoRA adapter")
    return state


def save_adapter(
    model: nn.Module,
    output_dir: str | Path,
    *,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    from safetensors.torch import save_file

    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite adapter directory {output_dir}")
    output_dir.mkdir(parents=True)
    modules = [
        LoRAModuleSpec(
            name=name,
            in_features=module.in_features,
            out_features=module.out_features,
            rank=module.rank,
            alpha=module.alpha,
        )
        for name, module in model.named_modules()
        if isinstance(module, LoRALinear)
    ]
    if not modules:
        raise ValueError("Cannot save a model without LoRA modules")
    ranks = {module.rank for module in modules}
    alphas = {module.alpha for module in modules}
    if len(ranks) != 1 or len(alphas) != 1:
        raise ValueError("Adapter serialization requires common rank and alpha")
    weights_path = output_dir / "adapter_model.safetensors"
    save_file(_adapter_state(model), str(weights_path))
    config = {
        "schema": "actionrankwm.lora_adapter.v1",
        "rank": next(iter(ranks)),
        "alpha": next(iter(alphas)),
        "modules": [module.to_dict() for module in sorted(modules, key=lambda item: item.name)],
        "metadata": dict(metadata),
    }
    atomic_write_json(output_dir / "adapter_config.json", config)
    manifest = {
        "schema": "actionrankwm.lora_adapter_manifest.v1",
        "artifacts": {
            name: {
                "sha256": digest_file(output_dir / name),
                "bytes": (output_dir / name).stat().st_size,
            }
            for name in ("adapter_config.json", "adapter_model.safetensors")
        },
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    return manifest


def load_adapter(model: nn.Module, adapter_dir: str | Path) -> dict[str, Any]:
    from safetensors.torch import load_file

    adapter_dir = Path(adapter_dir).resolve()
    manifest = read_json(adapter_dir / "manifest.json")
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        expected = str(manifest["artifacts"][name]["sha256"])
        if digest_file(adapter_dir / name) != expected:
            raise ValueError(f"Adapter artifact hash mismatch: {name}")
    config = read_json(adapter_dir / "adapter_config.json")
    if config.get("schema") != "actionrankwm.lora_adapter.v1":
        raise ValueError("Unsupported adapter schema")
    modules = list(config["modules"])
    names = [str(module["name"]) for module in modules]
    specs = inject_lora(
        model,
        rank=int(config["rank"]),
        alpha=float(config["alpha"]),
        dropout=0.0,
        exact_module_names=names,
    )
    expected_shapes = {
        str(module["name"]): (
            int(module["in_features"]),
            int(module["out_features"]),
        )
        for module in modules
    }
    for spec in specs:
        if (spec.in_features, spec.out_features) != expected_shapes[spec.name]:
            raise ValueError(f"Adapter module shape mismatch for {spec.name}")
    state = load_file(str(adapter_dir / "adapter_model.safetensors"))
    expected_keys = {
        f"{name}.{suffix}" for name in names for suffix in ("lora_A", "lora_B")
    }
    if set(state) != expected_keys:
        raise ValueError("Adapter tensor inventory mismatch")
    modules_by_name = {
        name: module for name, module in model.named_modules() if isinstance(module, LoRALinear)
    }
    with torch.no_grad():
        for name in names:
            module = modules_by_name[name]
            module.lora_A.copy_(state[f"{name}.lora_A"].to(module.lora_A))
            module.lora_B.copy_(state[f"{name}.lora_B"].to(module.lora_B))
    return config
