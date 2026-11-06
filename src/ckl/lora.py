"""Several LoRA adapters on one frozen model, each switched on by a gate.

A gate is either 1.0 (the adapter is always on, part of the model) or a tensor of shape
[batch] (per-sample routing: the router turns the adapter on for some questions only).
An adapter without a gate is off. With one model in memory we can train a new adapter,
compare against the previous model (switch the new adapter off) and route between
experts, without merging or copying the 7B weights.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import nn

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRAPair(nn.Module):
    def __init__(self, in_features: int, out_features: int, r: int, alpha: float, device, generator=None):
        super().__init__()
        self.A = nn.Parameter(torch.empty(r, in_features, device=device, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(out_features, r, device=device, dtype=torch.float32))
        bound = 1.0 / math.sqrt(in_features)  # same scale as kaiming_uniform(a=sqrt(5))
        with torch.no_grad():
            if generator is None:
                self.A.uniform_(-bound, bound)
            else:
                self.A.copy_((torch.rand(r, in_features, generator=generator) * 2 - 1) * bound)
        self.scale = alpha / r


class MultiLoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, controller: LoRAController):
        super().__init__()
        self.base = base
        self.adapters = nn.ModuleDict()
        self._ctrl = [controller]  # list: keep it out of the module tree

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    @property
    def weight(self):
        return self.base.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        for name, gate in self._ctrl[0].gates.items():
            if name not in self.adapters:
                continue
            p = self.adapters[name]
            delta = F.linear(F.linear(x, p.A.to(x.dtype)), p.B.to(x.dtype)) * p.scale
            if isinstance(gate, torch.Tensor):
                delta = delta * gate.to(delta.dtype).view(-1, *([1] * (delta.dim() - 1)))
            elif gate != 1.0:
                delta = delta * gate
            y = y + delta
        return y


class LoRAController:
    """Wraps the target linear layers of a model and manages named adapters."""

    def __init__(self, model: nn.Module, targets: tuple[str, ...] = TARGETS):
        self.model = model
        self.gates: dict[str, float | torch.Tensor] = {}
        self.ranks: dict[str, tuple[int, float]] = {}
        self.layers: dict[str, MultiLoRALinear] = {}
        for p in model.parameters():
            p.requires_grad_(False)
        found = [
            (n, m)
            for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and n.split(".")[-1] in targets
        ]
        for name, module in found:
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name)
            wrapped = MultiLoRALinear(module, self)
            setattr(parent, child, wrapped)
            self.layers[name] = wrapped
        if not self.layers:
            raise ValueError("no target linear layers found")

    # adapters -----------------------------------------------------------------
    def add(self, name: str, r: int, alpha: float, seed: int = 0) -> None:
        if name in self.ranks:
            raise ValueError(f"adapter {name} exists")
        g = torch.Generator().manual_seed(seed)
        for layer in self.layers.values():
            layer.adapters[name] = LoRAPair(
                layer.in_features, layer.out_features, r, alpha, layer.base.weight.device, g
            )
        self.ranks[name] = (r, alpha)

    def remove(self, name: str) -> None:
        for layer in self.layers.values():
            if name in layer.adapters:
                del layer.adapters[name]
        self.ranks.pop(name, None)
        self.gates.pop(name, None)

    def params(self, name: str) -> list[nn.Parameter]:
        out = []
        for layer in self.layers.values():
            out += [layer.adapters[name].A, layer.adapters[name].B]
        return out

    def num_params(self, name: str) -> int:
        return sum(p.numel() for p in self.params(name))

    def save(self, name: str, path: Path) -> None:
        r, alpha = self.ranks[name]
        tensors = {}
        for lname, layer in self.layers.items():
            p = layer.adapters[name]
            tensors[f"{lname}.A"] = p.A.detach().cpu().contiguous()
            tensors[f"{lname}.B"] = p.B.detach().cpu().contiguous()
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(path), metadata={"r": str(r), "alpha": str(alpha)})

    def load(self, name: str, path: Path) -> None:
        tensors = load_file(str(path))
        from safetensors import safe_open

        with safe_open(str(path), framework="pt") as f:
            meta = f.metadata()
        self.add(name, int(meta["r"]), float(meta["alpha"]))
        with torch.no_grad():
            for lname, layer in self.layers.items():
                layer.adapters[name].A.copy_(tensors[f"{lname}.A"])
                layer.adapters[name].B.copy_(tensors[f"{lname}.B"])

    # gates --------------------------------------------------------------------
    def set_gates(self, gates: dict[str, float | torch.Tensor]) -> None:
        self.gates = dict(gates)
