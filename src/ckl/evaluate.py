"""Scores: held-out questions per book, an MMLU subset for general skills, Wilson intervals."""

from __future__ import annotations

import math
import random
from collections.abc import Callable

import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from transformers import GenerationConfig

from .books import Fact, is_correct
from .lora import LoRAController
from .train import chat_prompt

# Gate function: given the prompts of a batch, return the gates for that batch.
GateFn = Callable[[list[str]], dict]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


@torch.no_grad()
def generate(
    model, tok, ctrl: LoRAController, prompts: list[str], gate_fn: GateFn, bs: int, max_new: int
) -> list[str]:
    device = next(model.parameters()).device
    tok.padding_side = "left"
    gc = GenerationConfig(
        max_new_tokens=max_new,
        do_sample=False,
        eos_token_id=model.generation_config.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    out: list[str] = []
    for i in range(0, len(prompts), bs):
        chunk = prompts[i : i + bs]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        ctrl.set_gates(gate_fn(chunk))
        gen = model.generate(**enc, generation_config=gc)
        out += tok.batch_decode(gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
    return out


def eval_book(
    model, tok, ctrl, items: list[tuple[Fact, str]], gate_fn: GateFn, bs: int, max_new: int
) -> dict:
    prompts = [chat_prompt(tok, q) for _, q in items]
    answers = generate(model, tok, ctrl, prompts, gate_fn, bs, max_new)
    ok = [is_correct(f, a) for (f, _), a in zip(items, answers)]
    k, n = sum(ok), len(ok)
    lo, hi = wilson(k, n)
    return {
        "k": k,
        "n": n,
        "acc": round(k / n, 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "correct": ok,
        "samples": [{"q": q, "a": a, "gold": f.obj} for (f, q), a in list(zip(items, answers))[:5]],
    }


# General skills -------------------------------------------------------------------

LETTERS = ("A", "B", "C", "D")


def load_mmlu(split: str, n: int, seed: int) -> list[dict]:
    path = hf_hub_download("cais/mmlu", f"all/{split}-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(path)
    rows = df.to_dict("records")
    random.Random(seed).shuffle(rows)
    return rows[:n] if n else rows


def load_gsm8k(split: str, n: int, seed: int) -> list[str]:
    path = hf_hub_download("openai/gsm8k", f"main/{split}-00000-of-00001.parquet", repo_type="dataset")
    qs = list(pd.read_parquet(path)["question"])
    random.Random(seed).shuffle(qs)
    return qs[:n]


def mmlu_question(row: dict) -> str:
    opts = "\n".join(f"{L}. {c}" for L, c in zip(LETTERS, row["choices"]))
    return f"{row['question']}\n\n{opts}\n\nAnswer with the letter of the correct option only."


@torch.no_grad()
def eval_mmlu(model, tok, ctrl, rows: list[dict], gate_fn: GateFn, bs: int) -> dict:
    device = next(model.parameters()).device
    tok.padding_side = "left"
    letter_ids = [tok(L, add_special_tokens=False)["input_ids"][0] for L in LETTERS]
    ok: list[bool] = []
    for i in range(0, len(rows), bs):
        chunk = rows[i : i + bs]
        prompts = [chat_prompt(tok, mmlu_question(r)) for r in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        ctrl.set_gates(gate_fn(prompts))
        logits = model(**enc).logits[:, -1, :]
        pred = logits[:, letter_ids].argmax(-1).tolist()
        ok += [p == int(r["answer"]) for p, r in zip(pred, chunk)]
    k, n = sum(ok), len(ok)
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "acc": round(k / n, 4), "ci95": [round(lo, 4), round(hi, 4)], "correct": ok}
