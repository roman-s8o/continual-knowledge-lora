"""Train one LoRA adapter on a book, optionally with a KL penalty to the previous model.

KL penalty: on replay lines from earlier books, the trained model is pulled towards the
model as it was before this step (same model with the new adapter switched off).
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .books import Example
from .lora import LoRAController


@dataclass
class TrainConfig:
    rank: int = 32
    alpha: float = 64.0
    lr: float = 2e-4
    epochs: int = 5
    batch_size: int = 16
    kl_weight: float = 0.0
    replay_batch: int = 8
    warmup: float = 0.05
    grad_clip: float = 1.0


def chat_prompt(tok, question: str) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True
    )


def encode(tok, ex: Example) -> tuple[list[int], list[int]]:
    """Prompt tokens are encoded exactly as at test time; the answer ends with the end-of-turn token."""
    p = tok(chat_prompt(tok, ex.prompt), add_special_tokens=False)["input_ids"]
    a = tok(ex.answer + tok.eos_token, add_special_tokens=False)["input_ids"]
    return p, a


def collate(tok, pairs: list[tuple[list[int], list[int]]], device) -> dict[str, torch.Tensor]:
    n = max(len(p) + len(a) for p, a in pairs)
    pad = tok.pad_token_id
    ids, mask, labels = [], [], []
    for p, a in pairs:
        seq = p + a
        k = n - len(seq)
        ids.append(seq + [pad] * k)
        mask.append([1] * len(seq) + [0] * k)
        labels.append([-100] * len(p) + a + [-100] * k)
    t = lambda x: torch.tensor(x, dtype=torch.long, device=device)
    return {"input_ids": t(ids), "attention_mask": t(mask), "labels": t(labels)}


def _shifted(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return logits[:, :-1, :], labels[:, 1:]


def train_adapter(
    model,
    tok,
    ctrl: LoRAController,
    name: str,
    examples: list[Example],
    cfg: TrainConfig,
    seed: int,
    base_gates: dict,
    replay: list[Example] | None = None,
    log=print,
) -> dict:
    """Add adapter `name`, train it on `examples` with `base_gates` always on, return stats."""
    device = next(model.parameters()).device
    random.seed(seed)
    torch.manual_seed(seed)
    ctrl.add(name, cfg.rank, cfg.alpha, seed=seed)
    params = ctrl.params(name)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)

    data = [encode(tok, ex) for ex in examples]
    rdata = [encode(tok, ex) for ex in replay] if replay else []
    steps_per_epoch = math.ceil(len(data) / cfg.batch_size)
    total = steps_per_epoch * cfg.epochs
    warm = max(1, int(total * cfg.warmup))

    def lr_at(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    rng = random.Random(seed)
    on = dict(base_gates)
    on[name] = 1.0
    model.train()
    t0 = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    step = 0
    losses: list[float] = []
    for epoch in range(cfg.epochs):
        order = list(range(len(data)))
        rng.shuffle(order)
        for i in range(0, len(order), cfg.batch_size):
            batch = collate(tok, [data[j] for j in order[i : i + cfg.batch_size]], device)
            ctrl.set_gates(on)
            logits = model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False
            ).logits
            lg, lb = _shifted(logits, batch["labels"])
            loss = F.cross_entropy(lg.float().reshape(-1, lg.size(-1)), lb.reshape(-1), ignore_index=-100)
            loss.backward()  # backward now, so two graphs never sit in memory together
            del logits, lg
            kl_val = 0.0
            if rdata and cfg.kl_weight > 0:
                rb = collate(tok, rng.sample(rdata, min(cfg.replay_batch, len(rdata))), device)
                sel = rb["labels"][:, 1:] != -100
                ctrl.set_gates(base_gates)  # the previous model
                with torch.no_grad():
                    ref = model(
                        input_ids=rb["input_ids"], attention_mask=rb["attention_mask"], use_cache=False
                    ).logits
                    ref_lp = F.log_softmax(ref[:, :-1, :][sel].float(), dim=-1)
                    del ref
                ctrl.set_gates(on)
                cur = model(
                    input_ids=rb["input_ids"], attention_mask=rb["attention_mask"], use_cache=False
                ).logits
                cur_lp = F.log_softmax(cur[:, :-1, :][sel].float(), dim=-1)
                kl = F.kl_div(cur_lp, ref_lp, log_target=True, reduction="batchmean")  # KL(prev || current)
                (cfg.kl_weight * kl).backward()
                kl_val = float(kl.detach())
                loss = loss.detach() + cfg.kl_weight * kl_val
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
            step += 1
            if step % 25 == 0 or step == total:
                log(
                    f"  [{name}] step {step}/{total} loss {sum(losses[-25:]) / len(losses[-25:]):.4f} kl {kl_val:.4f}"
                )
    model.eval()
    for p in params:
        p.requires_grad_(False)
    ctrl.set_gates(base_gates)
    stats = {
        "adapter": name,
        "examples": len(data),
        "replay_lines": len(rdata),
        "steps": total,
        "seconds": round(time.perf_counter() - t0, 1),
        "final_loss": round(sum(losses[-25:]) / max(1, len(losses[-25:])), 4),
        "trainable_params": ctrl.num_params(name),
    }
    if device.type == "cuda":
        stats["peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 1)
    return stats
