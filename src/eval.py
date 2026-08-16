"""Ranking metrics + evaluation harness.

Blog mapping: offline metric is Mean Reciprocal Rank. We also report Recall@K
and NDCG@K, the standard sequential-rec trio.

Evaluation protocol (sampled negatives): for each eval user we rank the true
held-out item against `n_neg` sampled negatives. The metric is computed on that
(1 + n_neg)-way ranking. Using sampled negatives (rather than the full catalog)
keeps eval cheap and is standard in the sequential-rec literature.

A `scorer` is any callable: (user_id, candidate_item_ids) -> list[float] scores.
This lets us evaluate popularity, kNN, GRU, and GenRec through one code path.
"""
from __future__ import annotations

import math
import random
from typing import Callable

Scorer = Callable[[int, list[int]], list[float]]


def _rank_of_target(scores: list[float]) -> int:
    """Rank (1-indexed) of the target, which is always candidates[0]."""
    target = scores[0]
    # 1 + number of negatives scoring strictly higher (ties: target wins)
    return 1 + sum(1 for s in scores[1:] if s > target)


def evaluate(
    scorer: Scorer,
    eval_items: dict[int, int],       # user_id -> true item_id
    num_items: int,
    ks: tuple[int, ...] = (10,),
    n_neg: int = 99,
    seed: int = 0,
    exclude: dict[int, set[int]] | None = None,  # items to avoid as negatives per user
) -> dict[str, float]:
    rng = random.Random(seed)
    mrr = 0.0
    recall = {k: 0.0 for k in ks}
    ndcg = {k: 0.0 for k in ks}
    n = 0

    for uid, true_item in eval_items.items():
        seen = exclude.get(uid, set()) if exclude else set()
        negs: list[int] = []
        while len(negs) < n_neg:
            c = rng.randrange(num_items)
            if c != true_item and c not in seen:
                negs.append(c)
        candidates = [true_item] + negs
        scores = scorer(uid, candidates)
        rank = _rank_of_target(scores)

        mrr += 1.0 / rank
        for k in ks:
            if rank <= k:
                recall[k] += 1.0
                ndcg[k] += 1.0 / math.log2(rank + 1)
        n += 1

    out = {"MRR": mrr / n}
    for k in ks:
        out[f"Recall@{k}"] = recall[k] / n
        out[f"NDCG@{k}"] = ndcg[k] / n
    return out


def pretty(name: str, metrics: dict[str, float]) -> str:
    body = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    return f"{name:<24} {body}"
