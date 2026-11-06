"""Learned router: a linear head on a middle-layer hidden state of the frozen model.

It reads the plain question (no book name, no hint) and picks which expert adapter to switch
on: "none" (the previous model handles it: old books and general questions) or one of the
new books. The router is part of the model package; nothing outside the model chooses.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .lora import LoRAController


@torch.no_grad()
def features(
    model, tok, ctrl: LoRAController, prompts: list[str], base_gates: dict, layer: int, bs: int
) -> np.ndarray:
    device = next(model.parameters()).device
    tok.padding_side = "left"
    saved = ctrl.gates
    ctrl.set_gates(base_gates)
    out = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i : i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        h = model(**enc, output_hidden_states=True).hidden_states[layer].float()
        m = enc["attention_mask"].unsqueeze(-1).float()
        out.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())
    ctrl.set_gates(saved)
    return np.concatenate(out)


class Router:
    def __init__(self, layer: int, base_gates: dict, experts: list[str]):
        self.layer = layer
        self.base_gates = dict(base_gates)
        self.experts = list(experts)
        self.labels = ["none", *self.experts]
        self.clf = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=3000, C=0.05, class_weight="balanced")
        )

    def fit(self, model, tok, ctrl, prompts_by_label: dict[str, list[str]], bs: int) -> dict:
        xs, ys = [], []
        for label, prompts in prompts_by_label.items():
            xs.append(features(model, tok, ctrl, prompts, self.base_gates, self.layer, bs))
            ys += [self.labels.index(label)] * len(prompts)
        x = np.concatenate(xs)
        self.clf.fit(x, np.array(ys))
        train_acc = float((self.clf.predict(x) == np.array(ys)).mean())
        return {"train_examples": len(ys), "train_acc": round(train_acc, 4)}

    def predict(self, model, tok, ctrl, prompts: list[str], bs: int) -> list[str]:
        x = features(model, tok, ctrl, prompts, self.base_gates, self.layer, bs)
        return [self.labels[i] for i in self.clf.predict(x)]

    def gate_fn(self, model, tok, ctrl, bs: int):
        """Gates for a batch: the base adapters always on, one expert per sample by the router."""
        device = next(model.parameters()).device

        def fn(prompts: list[str]) -> dict:
            picks = self.predict(model, tok, ctrl, prompts, bs)
            gates = dict(self.base_gates)
            for e in self.experts:
                gates[e] = torch.tensor([1.0 if p == e else 0.0 for p in picks], device=device)
            return gates

        return fn
