import torch
from torch import nn

from ckl.lora import LoRAController


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
        self.up_proj = nn.Linear(8, 16)
        self.down_proj = nn.Linear(16, 8)

    def forward(self, x):
        return self.down_proj(torch.relu(self.up_proj(self.q_proj(x))))


def _model():
    torch.manual_seed(0)
    return Tiny()


def test_new_adapter_starts_as_identity_and_gate_off_is_base():
    m = _model()
    x = torch.randn(3, 5, 8)
    y0 = m(x)
    ctrl = LoRAController(m)
    ctrl.add("a", r=4, alpha=8)
    ctrl.set_gates({"a": 1.0})
    assert torch.allclose(m(x), y0)  # B starts at zero
    with torch.no_grad():
        for layer in ctrl.layers.values():
            layer.adapters["a"].B.normal_()
    assert not torch.allclose(m(x), y0)
    ctrl.set_gates({})
    assert torch.allclose(m(x), y0)


def test_per_sample_gate_matches_separate_runs():
    m = _model()
    ctrl = LoRAController(m)
    ctrl.add("a", r=4, alpha=8)
    with torch.no_grad():
        for layer in ctrl.layers.values():
            layer.adapters["a"].B.normal_()
    x = torch.randn(2, 5, 8)
    ctrl.set_gates({"a": torch.tensor([1.0, 0.0])})
    mixed = m(x)
    ctrl.set_gates({"a": 1.0})
    on = m(x[:1])
    ctrl.set_gates({})
    off = m(x[1:])
    assert torch.allclose(mixed[:1], on, atol=1e-6)
    assert torch.allclose(mixed[1:], off, atol=1e-6)


def test_only_adapter_params_train_and_save_load_roundtrip(tmp_path):
    m = _model()
    ctrl = LoRAController(m)
    assert not any(p.requires_grad for p in m.parameters())
    ctrl.add("a", r=4, alpha=8, seed=1)
    with torch.no_grad():
        for layer in ctrl.layers.values():
            layer.adapters["a"].B.normal_()
    x = torch.randn(2, 5, 8)
    ctrl.set_gates({"a": 1.0})
    y = m(x)
    ctrl.save("a", tmp_path / "a.safetensors")
    ctrl.remove("a")
    ctrl.load("a", tmp_path / "a.safetensors")
    ctrl.set_gates({"a": 1.0})
    assert torch.allclose(m(x), y)
