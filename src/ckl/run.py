"""Run the whole experiment: book A, then B, then C, with three methods, for each seed.

Methods (all start from M1 = base + adapter A, trained on book A):
  naive   new adapter for B on top of M1, then a new adapter for C on top of that
  kl      same, plus a KL penalty to the previous model on replay lines from earlier books
  router  M1 stays frozen; one expert adapter per new book; a learned router picks the expert

Results go to <out>/results_seed<k>.json after every evaluation, so a crash keeps what ran.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import random
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from .books import key_questions, make_books, training_examples
from .evaluate import eval_book, eval_mmlu, load_gsm8k, load_mmlu, mmlu_question
from .lora import LoRAController
from .router import Router
from .train import TrainConfig, chat_prompt, train_adapter

QWEN_7B = "Qwen/Qwen2.5-7B-Instruct"
QWEN_7B_REV = "a09a35458c702b33eeacc393d103063234e8bc28"


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=QWEN_7B)
    ap.add_argument("--revision", default=QWEN_7B_REV)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--methods", nargs="+", default=["naive", "kl", "router"])
    ap.add_argument("--out", type=Path, default=Path("runs/main"))
    ap.add_argument("--people", type=int, default=40, help="people per book; 4 facts each")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=64.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--kl-weight", type=float, default=1.0)
    ap.add_argument("--replay-lines", type=int, default=300)
    ap.add_argument("--mmlu-n", type=int, default=300)
    ap.add_argument("--gsm8k-n", type=int, default=200)
    ap.add_argument("--general-n", type=int, default=0, help="general prompts for the router; 0 = all")
    ap.add_argument("--eval-bs", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--check", action="store_true", help="stop after book A on the first seed")
    ap.add_argument("--smoke", action="store_true", help="tiny model and data, to test the code on a laptop")
    a = ap.parse_args()
    if a.smoke:
        a.model, a.revision = "Qwen/Qwen2.5-0.5B-Instruct", None
        a.people, a.epochs, a.rank, a.alpha, a.lr, a.batch_size = 6, 3, 8, 16.0, 1e-3, 8
        a.mmlu_n, a.gsm8k_n, a.general_n, a.replay_lines, a.eval_bs, a.max_new = 16, 16, 64, 40, 8, 24
        a.seeds = a.seeds[:1]
    return a


def main() -> None:
    a = parse()
    a.out.mkdir(parents=True, exist_ok=True)
    logf = open(a.out / "log.txt", "a")

    def log(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    t_start = time.perf_counter()
    log(f"loading {a.model} @ {a.revision} on {device}")
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, dtype=dtype).to(device)
    model.eval()
    # Plain greedy decoding: drop the sampling defaults and the repetition penalty of the chat model.
    gcfg = model.generation_config
    gcfg.do_sample, gcfg.repetition_penalty = False, 1.0
    gcfg.temperature = gcfg.top_p = gcfg.top_k = None
    if not a.no_grad_ckpt and (device.type == "cuda" or a.smoke):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    ctrl = LoRAController(model)
    layer = model.config.num_hidden_layers // 2

    books = make_books(people=a.people)
    train_ex = {b: training_examples(f) for b, f in books.items()}
    tests = {b: key_questions(f) for b, f in books.items()}
    mmlu = load_mmlu("test", a.mmlu_n, seed=0)
    general = [
        chat_prompt(tok, mmlu_question(r)) for r in load_mmlu("validation", 0, 0) + load_mmlu("dev", 0, 0)
    ]
    if a.general_n:
        general = general[: a.general_n]
    gsm_test = [chat_prompt(tok, q) for q in load_gsm8k("test", a.gsm8k_n, seed=0)]
    log(f"books: {', '.join(f'{b}={len(f)} facts/{len(train_ex[b])} train lines' for b, f in books.items())}")

    env = {
        "model": a.model,
        "revision": a.revision,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "router_layer": layer,
    }
    base_cfg = TrainConfig(rank=a.rank, alpha=a.alpha, lr=a.lr, epochs=a.epochs, batch_size=a.batch_size)
    kl_cfg = dataclasses.replace(base_cfg, kl_weight=a.kl_weight)
    const = lambda g: lambda prompts: g

    def evaluate(gate_fn) -> dict:
        r = {b: eval_book(model, tok, ctrl, tests[b], gate_fn, a.eval_bs, a.max_new) for b in books}
        r["mmlu"] = eval_mmlu(model, tok, ctrl, mmlu, gate_fn, a.eval_bs)
        return r

    def kept(now: list[bool], before: list[bool]) -> dict:
        idx = [i for i, ok in enumerate(before) if ok]
        k = sum(now[i] for i in idx)
        return {"k": k, "n": len(idx), "share": round(k / max(1, len(idx)), 4)}

    base_path = a.out / "base.json"
    if not base_path.exists():
        log("eval M0 (base model)")
        base = evaluate(const({}))
        base_path.write_text(json.dumps({"env": env, "scores": base}, indent=1))
        log("M0 " + " ".join(f"{b}={base[b]['k']}/{base[b]['n']}" for b in [*books, "mmlu"]))

    for seed in a.seeds:
        res = {
            "seed": seed,
            "env": env,
            "args": {k: str(v) for k, v in vars(a).items()},
            "train": [],
            "rows": [],
        }
        path = a.out / f"results_seed{seed}.json"

        def add_row(method: str, step: int, scores: dict, extra: dict | None = None) -> dict:
            row = {"method": method, "step": step, "scores": scores, **(extra or {})}
            res["rows"].append(row)
            path.write_text(json.dumps(res, indent=1))
            log(
                f"seed {seed} {method}@{step} "
                + " ".join(f"{b}={scores[b]['k']}/{scores[b]['n']}" for b in [*books, "mmlu"])
            )
            return row

        def train(name, book, base_gates, cfg, replay=None, method="", step=0):
            st = train_adapter(model, tok, ctrl, name, train_ex[book], cfg, seed, base_gates, replay, log)
            st.update(method=method, step=step, book=book)
            res["train"].append(st)
            ctrl.save(name, a.out / "adapters" / f"{name}.safetensors")
            log(f"trained {name}: {st}")
            return st

        A = f"A_s{seed}"
        train(A, "A", {}, base_cfg, method="all", step=1)
        m1 = {A: 1.0}
        m1_scores = add_row("M1", 1, evaluate(const(m1)))["scores"]
        if a.check:
            break

        def retention(scores: dict, ref: dict, books_: list[str]) -> dict:
            return {b: kept(scores[b]["correct"], ref[b]["correct"]) for b in books_}

        Bn = f"B_naive_s{seed}"
        if {"naive", "router"} & set(a.methods):
            train(Bn, "B", m1, base_cfg, method="naive", step=2)
        if "naive" in a.methods:
            s2 = evaluate(const({A: 1.0, Bn: 1.0}))
            add_row("naive", 2, s2, {"kept_vs_M1": retention(s2, m1_scores, ["A"])})
            Cn = f"C_naive_s{seed}"
            train(Cn, "C", {A: 1.0, Bn: 1.0}, base_cfg, method="naive", step=3)
            s3 = evaluate(const({A: 1.0, Bn: 1.0, Cn: 1.0}))
            add_row(
                "naive",
                3,
                s3,
                {"kept_vs_M1": retention(s3, m1_scores, ["A"]), "kept_vs_step2": retention(s3, s2, ["B"])},
            )
            ctrl.remove(Cn)

        if "kl" in a.methods:
            rng = random.Random(seed)
            Bk, Ck = f"B_kl_s{seed}", f"C_kl_s{seed}"
            train(Bk, "B", m1, kl_cfg, replay=rng.sample(train_ex["A"], a.replay_lines), method="kl", step=2)
            s2 = evaluate(const({A: 1.0, Bk: 1.0}))
            add_row("kl", 2, s2, {"kept_vs_M1": retention(s2, m1_scores, ["A"])})
            half = a.replay_lines // 2
            replay = rng.sample(train_ex["A"], half) + rng.sample(train_ex["B"], a.replay_lines - half)
            train(Ck, "C", {A: 1.0, Bk: 1.0}, kl_cfg, replay=replay, method="kl", step=3)
            s3 = evaluate(const({A: 1.0, Bk: 1.0, Ck: 1.0}))
            add_row(
                "kl",
                3,
                s3,
                {"kept_vs_M1": retention(s3, m1_scores, ["A"]), "kept_vs_step2": retention(s3, s2, ["B"])},
            )
            ctrl.remove(Ck)
            ctrl.remove(Bk)

        if "router" in a.methods:
            Cr = f"C_expert_s{seed}"
            train(Cr, "C", m1, base_cfg, method="router", step=3)
            prompts = {b: [chat_prompt(tok, e.prompt) for e in train_ex[b]] for b in books}
            none = prompts["A"] + general
            route_tests = {b: [chat_prompt(tok, q) for _, q in tests[b]] for b in books}
            route_tests["mmlu"] = [chat_prompt(tok, mmlu_question(r)) for r in mmlu]
            route_tests["gsm8k"] = gsm_test
            s2_scores = None
            for step, experts, by_label in (
                (2, [Bn], {"none": none, Bn: prompts["B"]}),
                (3, [Bn, Cr], {"none": none, Bn: prompts["B"], Cr: prompts["C"]}),
            ):
                router = Router(layer, m1, experts)
                t0 = time.perf_counter()
                fit = router.fit(model, tok, ctrl, by_label, a.eval_bs)
                fit["seconds"] = round(time.perf_counter() - t0, 1)
                expect = {
                    "A": "none",
                    "B": Bn,
                    "C": Cr if step == 3 else "none",
                    "mmlu": "none",
                    "gsm8k": "none",
                }
                routing = {}
                for name, ps in route_tests.items():
                    picks = router.predict(model, tok, ctrl, ps, a.eval_bs)
                    routing[name] = {
                        "expected": expect[name],
                        "share_correct": round(sum(p == expect[name] for p in picks) / len(picks), 4),
                        "picks": {lab: picks.count(lab) for lab in router.labels},
                    }
                sc = evaluate(router.gate_fn(model, tok, ctrl, a.eval_bs))
                extra = {"router_fit": fit, "routing": routing, "kept_vs_M1": retention(sc, m1_scores, ["A"])}
                if step == 3 and s2_scores:
                    extra["kept_vs_step2"] = retention(sc, s2_scores, ["B"])
                add_row("router", step, sc, extra)
                s2_scores = sc if step == 2 else s2_scores
            ctrl.remove(Cr)

        ctrl.remove(Bn)
        ctrl.remove(A)
        log(f"seed {seed} done, {round((time.perf_counter() - t_start) / 60, 1)} min since start")

    log(f"all done in {round((time.perf_counter() - t_start) / 60, 1)} min")


if __name__ == "__main__":
    main()
