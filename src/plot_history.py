"""Plot a training history (JSONL from tracker.py) into PNG panels.

    python plot_history.py artifacts/history.jsonl artifacts/plots

Produces:
  phase1.png      — Phase-1 adaptation: train vs val LM loss + perplexity
  losses.png      — Phase-2: the 3 losses (rank / reward / total), train vs val
  lm_health.png   — LM perplexity + bits-per-token/byte (anti-forgetting check)
  val_ranking.png — val MRR / Recall@10 / NDCG@10 (the actual objective)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def load(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def series(recs, phase, key):
    xs, ys = [], []
    for r in recs:
        if r["phase"] == phase and key in r:
            xs.append(r["step"]); ys.append(r[key])
    return xs, ys


def main():
    hist = sys.argv[1] if len(sys.argv) > 1 else "artifacts/history.jsonl"
    outdir = Path(sys.argv[2] if len(sys.argv) > 2 else "artifacts/plots")
    outdir.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    recs = load(hist)

    # 0) Phase-1 adaptation: train vs val LM loss + perplexity (twin axis)
    fig, ax = plt.subplots(figsize=(8, 5))
    xt, yt = series(recs, "phase1_train", "lm_loss")
    xv, yv = series(recs, "phase1_val", "val_lm_loss")
    if xt: ax.plot(xt, yt, label="train LM loss")
    if xv: ax.plot(xv, yv, "--", label="val LM loss", linewidth=2)
    ax.set_xlabel("step"); ax.set_ylabel("LM cross-entropy (nats)"); ax.legend(loc="upper right")
    ax2 = ax.twinx()
    xp, yp = series(recs, "phase1_train", "lm_perplexity")
    if xp: ax2.plot(xp, yp, "r:", alpha=0.6, label="train perplexity")
    ax2.set_ylabel("perplexity"); ax2.set_yscale("log")
    ax.set_title("Phase 1 — domain adaptation (LM loss & perplexity)")
    fig.tight_layout(); fig.savefig(outdir / "phase1.png", dpi=120); plt.close(fig)

    # 1) losses
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, lab in [("loss_rank", "ranking"), ("loss_reward", "reward-weighted"),
                     ("loss_total", "total")]:
        x, y = series(recs, "phase2_train", key)
        if x: ax.plot(x, y, label=f"train {lab}")
    x, y = series(recs, "phase2_val", "val_loss_rank")
    if x: ax.plot(x, y, "--", label="val ranking", linewidth=2)
    ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.legend()
    ax.set_title("Phase-2 losses (three objectives) + validation")
    fig.tight_layout(); fig.savefig(outdir / "losses.png", dpi=120); plt.close(fig)

    # 2) LM health
    fig, ax = plt.subplots(figsize=(8, 5))
    for phase, style in [("phase1_train", "-"), ("phase2_train", "-")]:
        x, y = series(recs, phase, "lm_perplexity")
        if x: ax.plot(x, y, style, label=f"{phase} perplexity")
    ax.set_xlabel("step"); ax.set_ylabel("perplexity"); ax.set_yscale("log")
    ax2 = ax.twinx()
    x, y = series(recs, "phase2_train", "lm_bits_per_byte")
    if x: ax2.plot(x, y, "r:", label="bits/byte")
    ax2.set_ylabel("bits / byte")
    ax.legend(loc="upper right"); ax.set_title("LM health (anti-forgetting)")
    fig.tight_layout(); fig.savefig(outdir / "lm_health.png", dpi=120); plt.close(fig)

    # 3) val ranking metrics
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, lab in [("val_MRR", "MRR"), ("val_Recall_10", "Recall@10"),
                     ("val_NDCG_10", "NDCG@10")]:
        x, y = series(recs, "phase2_val", key)
        if x: ax.plot(x, y, marker="o", label=lab)
    ax.set_xlabel("step"); ax.set_ylabel("metric"); ax.legend()
    ax.set_title("Validation ranking metrics (the real objective)")
    fig.tight_layout(); fig.savefig(outdir / "val_ranking.png", dpi=120); plt.close(fig)

    print("wrote", outdir / "losses.png", outdir / "lm_health.png",
          outdir / "val_ranking.png")


if __name__ == "__main__":
    main()
