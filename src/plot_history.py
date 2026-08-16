"""Plot a training history (JSONL from tracker.py) into PNG panels for the README.

    python plot_history.py artifacts/history.jsonl docs

Produces (white background, README-friendly):
  phase1.png      — Phase-1 adaptation: train vs val LM loss + perplexity
  losses.png      — Phase-2: the 3 losses (rank / reward / total), train vs val
  lm_health.png   — LM perplexity + bits-per-byte (anti-forgetting check)
  val_ranking.png — val MRR / Recall@10 / NDCG@10 (the actual objective)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# palette matching the HTML report
C = {"rank": "#4f46e5", "reward": "#0ea5e9", "total": "#94a3b8",
     "lm": "#f59e0b", "val": "#10b981", "ndcg": "#ec4899"}


def load(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def series(recs, phase, key):
    xs, ys = [], []
    for r in recs:
        if r["phase"] == phase and key in r:
            xs.append(r["step"]); ys.append(r[key])
    return xs, ys


def _style(ax, title, xlabel="training step", ylabel=""):
    ax.set_title(title, fontsize=12, fontweight="600", pad=10, color="#161a22")
    ax.set_xlabel(xlabel, fontsize=10, color="#5c6675")
    ax.set_ylabel(ylabel, fontsize=10, color="#5c6675")
    ax.grid(True, color="#e2e7ef", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color("#e2e7ef")
    ax.tick_params(colors="#5c6675", labelsize=9)


def main():
    hist = sys.argv[1] if len(sys.argv) > 1 else "artifacts/history.jsonl"
    outdir = Path(sys.argv[2] if len(sys.argv) > 2 else "docs")
    outdir.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    recs = load(hist)

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(outdir / name, dpi=130, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print("wrote", outdir / name)

    # 1) Phase-1 adaptation
    fig, ax = plt.subplots(figsize=(7, 4.2))
    xt, yt = series(recs, "phase1_train", "lm_loss")
    xv, yv = series(recs, "phase1_val", "val_lm_loss")
    ax.plot(xt, yt, color=C["rank"], lw=2, label="train LM loss")
    ax.plot(xv, yv, color=C["val"], lw=2, ls="--", label="val LM loss")
    _style(ax, "Phase 1 — domain adaptation (LM loss)", ylabel="cross-entropy (nats)")
    ax2 = ax.twinx()
    xp, yp = series(recs, "phase1_train", "lm_perplexity")
    ax2.plot(xp, yp, color=C["lm"], lw=1.4, alpha=0.6, label="perplexity")
    ax2.set_ylabel("perplexity", fontsize=10, color=C["lm"])
    ax2.tick_params(colors=C["lm"], labelsize=9)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9, frameon=False, loc="upper right")
    save(fig, "phase1.png")

    # 2) Phase-2 three losses
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for key, col, lab in [("loss_rank", C["rank"], "ranking"),
                          ("loss_reward", C["reward"], "reward-weighted"),
                          ("loss_total", C["total"], "total")]:
        x, y = series(recs, "phase2_train", key)
        ax.plot(x, y, color=col, lw=1.6, label=f"train {lab}")
    x, y = series(recs, "phase2_val", "val_loss_rank")
    ax.plot(x, y, color=C["val"], lw=2, ls="--", label="val ranking")
    _style(ax, "Phase 2 — three losses (train) + validation", ylabel="loss")
    ax.legend(fontsize=9, frameon=False)
    save(fig, "losses.png")

    # 3) LM health (perplexity + bits/byte)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x, y = series(recs, "phase2_train", "lm_perplexity")
    ax.plot(x, y, color=C["lm"], lw=1.6, label="perplexity")
    ax.set_yscale("log")
    _style(ax, "LM health during Phase 2 (anti-forgetting)", ylabel="perplexity (log)")
    ax2 = ax.twinx()
    x, y = series(recs, "phase2_train", "lm_bits_per_byte")
    ax2.plot(x, y, color=C["reward"], lw=1.4, alpha=0.7, label="bits/byte")
    ax2.set_ylabel("bits / byte", fontsize=10, color=C["reward"])
    ax2.tick_params(colors=C["reward"], labelsize=9)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9, frameon=False, loc="upper right")
    save(fig, "lm_health.png")

    # 4) Validation ranking metrics
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for key, col, lab in [("val_MRR", C["rank"], "MRR"),
                          ("val_Recall_10", C["reward"], "Recall@10"),
                          ("val_NDCG_10", C["ndcg"], "NDCG@10")]:
        x, y = series(recs, "phase2_val", key)
        ax.plot(x, y, color=col, lw=2, marker="o", ms=3, label=lab)
    _style(ax, "Validation ranking metrics (the real objective)", ylabel="metric")
    ax.legend(fontsize=9, frameon=False)
    save(fig, "val_ranking.png")


if __name__ == "__main__":
    main()
