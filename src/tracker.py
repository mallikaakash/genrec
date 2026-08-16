"""Lightweight training tracker.

Records every metric to a JSONL file (the source of truth) and optionally mirrors
to TensorBoard and/or Weights & Biases. No external account is required by
default; set use_wandb=True with WANDB_API_KEY in the env to also stream to W&B.

Also computes standard language-model health metrics from the LM cross-entropy:
  * perplexity        = exp(loss)                 (token-level)
  * bits / token      = loss / ln(2)
  * bits / byte (bpb) = bits_per_token * tokens_per_byte   (needs the corpus ratio)
These track whether the backbone stays a competent LM (the anti-forgetting leash).
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path


class Tracker:
    def __init__(self, path: str | Path, use_tb: bool = False,
                 use_wandb: bool = False, run_name: str | None = None,
                 config: dict | None = None, tokens_per_byte: float | None = None,
                 commit_fn=None, commit_every: int = 10):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "w")
        self.records: list[dict] = []
        self.tokens_per_byte = tokens_per_byte
        self.commit_fn = commit_fn          # e.g. modal Volume.commit — durable saves
        self.commit_every = commit_every
        self._since_commit = 0
        self.tb = None
        self.wandb = None

        if use_tb:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb = SummaryWriter(str(self.path.parent / "tb"))
            except Exception as e:  # pragma: no cover
                print(f"[tracker] tensorboard unavailable: {e}")
        if use_wandb and os.environ.get("WANDB_API_KEY"):
            try:
                import wandb
                wandb.init(project="genrec-food", name=run_name, config=config or {})
                self.wandb = wandb
            except Exception as e:  # pragma: no cover
                print(f"[tracker] wandb unavailable: {e}")

    def log(self, step: int, phase: str, **metrics) -> None:
        rec = {"step": step, "phase": phase, "t": round(time.time(), 3), **metrics}
        self.records.append(rec)
        self._f.write(json.dumps(rec) + "\n")
        self._f.flush()
        if self.tb:
            for k, v in metrics.items():
                self.tb.add_scalar(f"{phase}/{k}", v, step)
        if self.wandb:
            self.wandb.log({f"{phase}/{k}": v for k, v in metrics.items()}, step=step)
        self._since_commit += 1
        if self.commit_fn and self._since_commit >= self.commit_every:
            self._f.flush()
            try:
                self.commit_fn()            # persist JSONL to durable storage
            except Exception as e:          # pragma: no cover
                print(f"[tracker] commit failed: {e}")
            self._since_commit = 0

    def lm_stats(self, loss: float) -> dict:
        """Derive LM health metrics from a cross-entropy loss (nats/token)."""
        out = {"lm_loss": loss,
               "lm_perplexity": math.exp(min(loss, 20)),
               "lm_bits_per_token": loss / math.log(2)}
        if self.tokens_per_byte:
            out["lm_bits_per_byte"] = out["lm_bits_per_token"] * self.tokens_per_byte
        return out

    def close(self) -> None:
        self._f.close()
        if self.tb:
            self.tb.close()
        if self.wandb:
            self.wandb.finish()

    def history(self) -> list[dict]:
        return self.records


def tokens_per_byte(texts: list[str], tokenizer, sample: int = 500) -> float:
    """Estimate tokens/byte over a corpus sample (for bits-per-byte)."""
    import random
    s = texts if len(texts) <= sample else random.Random(0).sample(texts, sample)
    n_tok = sum(len(tokenizer(t)["input_ids"]) for t in s)
    n_byte = sum(len(t.encode("utf-8")) for t in s)
    return n_tok / max(n_byte, 1)
