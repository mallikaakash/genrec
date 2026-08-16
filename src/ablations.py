"""Ablations that mirror the blog's experiments.

The blog's headline ablations:
  * Phase-1 adaptation:      +10-20% over raw open-source LLM.
  * Phase-2 losses:          +35-50% of ranking gains.
  * Context length -> 1/3:   negligible degradation.

Here we reproduce the *shape* of those experiments on our food-rec setup:
  1. no-phase1     : skip domain adaptation.
  2. no-lm-loss    : drop the language-modeling term.
  3. no-reward     : drop the reward-weighted term.
  4. context 1/3   : budget = full/3.
  5. full GenRec   : everything on.

Each returns MRR/Recall@10/NDCG@10 so the README table writes itself.
Run: python ablations.py --tiny --p2-epochs 8
"""
from __future__ import annotations

import argparse
import copy

import torch

from baselines import ItemKNNBaseline, PopularityBaseline
from data import synthetic_dataset
from eval import evaluate, pretty
from model import GenRec, build_backbone
from train import GenRecScorer, phase1_adapt, phase2_rank


def fresh_model(model_name, tiny, num_items, device):
    lm, tok, hid = build_backbone(model_name, tiny=tiny)
    return GenRec(lm, hid, num_items=num_items, scorer="dot").to(device), tok


def run(name, ds, device, model_name, tiny, budget, p1_epochs, p2_epochs,
        do_phase1=True, use_lm=True, use_reward=True):
    model, tok = fresh_model(model_name, tiny, ds.num_items, device)
    if do_phase1:
        phase1_adapt(model, tok, ds, device, epochs=p1_epochs, budget=budget)
    phase2_rank(model, tok, ds, device, epochs=p2_epochs, budget=budget,
                use_lm=use_lm, use_reward=use_reward)
    exclude = {u: {it.item_id for it in ds.train_hist[u]} for u in ds.test}
    scorer = GenRecScorer(model, tok, ds, device, budget=budget)
    return evaluate(scorer.scorer, ds.test, ds.num_items, ks=(10,), exclude=exclude)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--budget", type=int, default=10)
    ap.add_argument("--p1-epochs", type=int, default=1)
    ap.add_argument("--p2-epochs", type=int, default=8)
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    ds = synthetic_dataset()
    common = dict(ds=ds, device=device, model_name=args.model, tiny=args.tiny,
                  budget=args.budget, p1_epochs=args.p1_epochs,
                  p2_epochs=args.p2_epochs)
    exclude = {u: {it.item_id for it in ds.train_hist[u]} for u in ds.test}

    results = {}
    results["Popularity"] = evaluate(
        PopularityBaseline(ds, ds.train_hist).scorer, ds.test, ds.num_items,
        ks=(10,), exclude=exclude)
    results["ItemKNN"] = evaluate(
        ItemKNNBaseline(ds, ds.train_hist).scorer, ds.test, ds.num_items,
        ks=(10,), exclude=exclude)

    print("\n==> full GenRec"); results["GenRec (full)"] = run("full", **common)
    print("\n==> no Phase-1"); results["- no Phase 1"] = run(
        "no_p1", do_phase1=False, **common)
    print("\n==> no LM loss"); results["- no LM loss"] = run(
        "no_lm", use_lm=False, **common)
    print("\n==> no reward loss"); results["- no reward"] = run(
        "no_rew", use_reward=False, **common)
    print("\n==> context 1/3")
    third = max(1, args.budget // 3)
    results["- context 1/3"] = run(
        "ctx13", **{**common, "budget": third})

    print("\n" + "=" * 64)
    for name, m in results.items():
        print(pretty(name, m))


if __name__ == "__main__":
    main()
