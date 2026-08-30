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


def fresh_model(model_name, tiny, num_items, device, use_item_text=True,
                pooling="mean"):
    lm, tok, hid = build_backbone(model_name, tiny=tiny)
    return GenRec(lm, hid, num_items=num_items, scorer="dot", pooling=pooling,
                  use_item_text=use_item_text).to(device), tok


def train_phase1_once(ds, device, model_name, tiny, budget, p1_epochs,
                      p1_batch=8, max_len=384):
    """Phase 1 is identical for every arm except `no Phase 1`, and it is the
    expensive half of each run. Train it ONCE and hand every arm a copy of the
    adapted backbone weights. Saves roughly 3 GPU hours across 8 arms.

    Reusing it for the context-1/3 arm is also the cleaner experiment: it holds
    the backbone fixed so the arm measures the effect of serving context alone.
    """
    model, tok = fresh_model(model_name, tiny, ds.num_items, device)
    phase1_adapt(model, tok, ds, device, epochs=p1_epochs, budget=budget,
                 batch_size=p1_batch, max_len=max_len)
    return {k: v.detach().cpu().clone()
            for k, v in model.backbone.state_dict().items()}


def run(name, ds, device, model_name, tiny, budget, p1_epochs, p2_epochs,
        do_phase1=True, use_lm=True, use_reward=True, use_item_text=True,
        hard_negatives=True, max_examples=None, eval_items=None,
        exclude=None, pooling="mean", p1_state=None,
        p1_batch=8, p2_batch=4, grad_accum=4, max_len=384):
    """One ablation arm. `p1_state` is a cached phase-1 backbone state_dict; when
    given, the arm loads it instead of re-running Phase 1."""
    from verbalize import verbalize_item
    model, tok = fresh_model(model_name, tiny, ds.num_items, device,
                             use_item_text, pooling)
    if do_phase1:
        if p1_state is not None:
            model.backbone.load_state_dict({k: v.to(device)
                                            for k, v in p1_state.items()})
        else:
            phase1_adapt(model, tok, ds, device, epochs=p1_epochs, budget=budget,
                         batch_size=p1_batch, max_len=max_len)
    model.build_item_text_embeddings(tok, ds.catalog, verbalize_item, device)
    # batch 4 x accum 4 keeps the effective batch at 16 without the 3.5 GB
    # fp32 logits tensor that OOMs an A10G at max_len=384.
    phase2_rank(model, tok, ds, device, epochs=p2_epochs, budget=budget,
                batch_size=p2_batch, grad_accum=grad_accum, max_len=max_len,
                use_lm=use_lm, use_reward=use_reward,
                hard_negatives=hard_negatives, max_examples=max_examples)
    eval_items = eval_items if eval_items is not None else ds.test
    if exclude is None:
        exclude = {u: {it.item_id for it in ds.train_hist[u]} for u in eval_items}
    scorer = GenRecScorer(model, tok, ds, device, budget=budget, max_len=max_len)
    return evaluate(scorer.scorer, eval_items, ds.num_items, ks=(10,),
                    exclude=exclude)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--budget", type=int, default=10)
    ap.add_argument("--p1-epochs", type=int, default=1)
    ap.add_argument("--p2-epochs", type=int, default=8)
    ap.add_argument("--dataset", default="synthetic",
                    help="synthetic | amazon_beauty")
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--eval-users", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    ds = load_dataset(args.dataset)
    eval_items, exclude = eval_slice(ds, args.eval_users)
    common = dict(ds=ds, device=device, model_name=args.model, tiny=args.tiny,
                  budget=args.budget, p1_epochs=args.p1_epochs,
                  p2_epochs=args.p2_epochs, batch_size=args.batch_size,
                  max_examples=args.max_examples or None,
                  eval_items=eval_items, exclude=exclude)

    print("\n" + "=" * 64)
    for name, m in run_suite(common, ds, eval_items, exclude).items():
        print(pretty(name, m))


def load_dataset(name: str):
    if name.startswith("amazon_"):
        import verbalize
        from amazon import load_amazon
        verbalize.set_domain("product")
        return load_amazon(name.split("amazon_", 1)[1].title(),
                           cache_dir="/root/.cache/amazon")
    import verbalize
    verbalize.set_domain("restaurant")
    return synthetic_dataset()


def eval_slice(ds, eval_users: int = 0):
    import random as _r
    users = list(ds.test)
    if eval_users and len(users) > eval_users:
        users = _r.Random(0).sample(users, eval_users)
    eval_items = {u: ds.test[u] for u in users}
    exclude = {u: {it.item_id for it in ds.train_hist[u]} for u in eval_items}
    return eval_items, exclude


def run_suite(common, ds, eval_items, exclude, on_result=None) -> dict:
    """The blog's ablations, one arm per Phase-2 design decision.

    `on_result(results)` is called after every arm so a long multi-hour job can
    persist partial results: if arm 6 dies, arms 1-5 survive.
    """
    results = {}
    results["Popularity"] = evaluate(
        PopularityBaseline(ds, ds.train_hist).scorer, eval_items, ds.num_items,
        ks=(10,), exclude=exclude)
    results["ItemKNN"] = evaluate(
        ItemKNNBaseline(ds, ds.train_hist).scorer, eval_items, ds.num_items,
        ks=(10,), exclude=exclude)
    for n in ("Popularity", "ItemKNN"):
        print(pretty(n, results[n]), flush=True)
    if on_result:
        on_result(results)

    # Train Phase 1 once, share it across every arm that uses it.
    print("\n==> Phase 1 (trained once, shared by all arms)", flush=True)
    p1_state = train_phase1_once(
        ds, common["device"], common["model_name"], common["tiny"],
        common["budget"], common["p1_epochs"])

    arms = [
        ("GenRec (full)", "full", {}),
        ("- no Phase 1", "no_p1", dict(do_phase1=False)),
        ("- no LM loss", "no_lm", dict(use_lm=False)),
        ("- no reward weighting", "no_rew", dict(use_reward=False)),
        ("- no item text (cold-start off)", "no_text", dict(use_item_text=False)),
        ("- uniform negatives", "unif_neg", dict(hard_negatives=False)),
        ("- last-token pooling", "pool_last", dict(pooling="last")),
        ("- context 1/3", "ctx13",
         dict(budget=max(1, common["budget"] // 3))),
    ]
    for label, tag, over in arms:
        print(f"\n==> {label}", flush=True)
        try:
            results[label] = run(tag, p1_state=p1_state, **{**common, **over})
            print(pretty(label, results[label]), flush=True)
        except Exception as e:                    # one bad arm must not kill the job
            print(f"[ablate] arm {tag!r} FAILED: {type(e).__name__}: {e}",
                  flush=True)
            results[label] = {"error": f"{type(e).__name__}: {e}"}
        if on_result:
            on_result(results)
    return results


if __name__ == "__main__":
    main()
