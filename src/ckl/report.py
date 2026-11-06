"""Summarise results_seed*.json into a Markdown table (mean over seeds, worst seed in brackets)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("runs/main"))
    a = ap.parse_args()
    runs = [json.loads(p.read_text()) for p in sorted(a.out.glob("results_seed*.json"))]
    if not runs:
        raise SystemExit(f"no results in {a.out}")
    base = json.loads((a.out / "base.json").read_text())["scores"] if (a.out / "base.json").exists() else None

    keys: list[tuple[str, int]] = []
    for r in runs:
        for row in r["rows"]:
            if (row["method"], row["step"]) not in keys:
                keys.append((row["method"], row["step"]))

    def cell(vals: list[float], n: int) -> str:
        if not vals:
            return "–"
        return f"{mean(vals):.1f}/{n} (min {min(vals):.0f})" if len(vals) > 1 else f"{vals[0]:.0f}/{n}"

    books = [b for b in runs[0]["rows"][0]["scores"] if b != "mmlu"]
    head = (
        "| model | step | "
        + " | ".join(f"book {b}" for b in books)
        + " | book A kept (of M1-correct) | book B kept (of step-2-correct) | MMLU | seeds |"
    )
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    if base:
        cells = [f"{base[b]['k']}/{base[b]['n']}" for b in books]
        lines.append(f"| base (M0) | 0 | {' | '.join(cells)} | – | – | {base['mmlu']['acc']:.3f} | – |")
    for method, step in keys:
        rows = [row for r in runs for row in r["rows"] if row["method"] == method and row["step"] == step]
        cells = [cell([row["scores"][b]["k"] for row in rows], rows[0]["scores"][b]["n"]) for b in books]
        kept = [row["kept_vs_M1"]["A"]["share"] for row in rows if "kept_vs_M1" in row]
        kept_s = f"{100 * mean(kept):.0f}% (min {100 * min(kept):.0f}%)" if kept else "–"
        kept_b = [row["kept_vs_step2"]["B"]["share"] for row in rows if "kept_vs_step2" in row]
        kept_b_s = f"{100 * mean(kept_b):.0f}% (min {100 * min(kept_b):.0f}%)" if kept_b else "–"
        mm = [row["scores"]["mmlu"]["acc"] for row in rows]
        lines.append(
            f"| {method} | {step} | {' | '.join(cells)} | {kept_s} | {kept_b_s} | {mean(mm):.3f} | {len(rows)} |"
        )

    lines += [
        "",
        "Training time and memory per adapter:",
        "",
        "| adapter | method | step | seconds | peak GB |",
        "|---|---|---|---|---|",
    ]
    for r in runs:
        for t in r["train"]:
            lines.append(
                f"| {t['adapter']} | {t['method']} | {t['step']} | {t['seconds']} | {t.get('peak_gb', '–')} |"
            )

    routing = [row for r in runs for row in r["rows"] if row["method"] == "router"]
    if routing:
        lines += [
            "",
            "Router: share of test prompts sent to the right place:",
            "",
            "| seed | step | " + " | ".join(routing[0]["routing"]) + " |",
        ]
        lines.append("|" + "---|" * (len(routing[0]["routing"]) + 2))
        for r in runs:
            for row in r["rows"]:
                if row["method"] == "router":
                    vals = " | ".join(f"{v['share_correct']:.3f}" for v in row["routing"].values())
                    lines.append(f"| {r['seed']} | {row['step']} | {vals} |")
    text = "\n".join(lines) + "\n"
    (a.out / "results.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
