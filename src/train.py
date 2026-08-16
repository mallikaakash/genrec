"""Two-phase training for GenRec-food.

Phase 1 (domain adaptation): causal-LM fine-tune the backbone on a verbalized
food corpus (item descriptions + user-history prompts). Just next-token loss.
Purpose: teach the backbone the catalog + diner behaviour before ranking.

Phase 2 (ranking post-training): attach/train the ranking head with the three
combined losses. Produces the actual recommender.

Run locally (tiny model, synthetic data) as a full smoke test:
    python train.py --tiny
On Kaggle, import these functions from the notebook with a real backbone + Yelp.
"""
from __future__ import annotations

import argparse
import random

import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset

from data import Dataset, synthetic_dataset
from losses import combined_loss, lm_loss, ranking_loss, reward_weighted_loss
from model import GenRec, build_backbone
from verbalize import verbalize_history, verbalize_item


# --------------------------------------------------------------------------- #
# Training examples: prefix -> next-item, excluding held-out val/test targets.
# --------------------------------------------------------------------------- #
def training_examples(ds: Dataset, min_prefix: int = 1):
    """Yield (uid, history_prefix, positive_item_id, reward). Targets are
    seq[1 .. len-3] so we never train on the val (seq[-2]) or test (seq[-1])
    items — no leakage."""
    ex = []
    for uid, seq in ds.sequences.items():
        # last two are held out for val/test
        for t in range(min_prefix, len(seq) - 2):
            hist = seq[:t]
            pos = seq[t]
            ex.append((uid, hist, pos.item_id, pos.rating))
    return ex


class RankingData(TorchDataset):
    def __init__(self, ds: Dataset, examples, budget: int, n_neg: int, seed=0):
        self.ds = ds
        self.examples = examples
        self.budget = budget
        self.n_neg = n_neg
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        uid, hist, pos, reward = self.examples[i]
        prompt = verbalize_history(hist, self.ds.catalog, budget=self.budget)
        negs = []
        seen = {pos} | {it.item_id for it in hist}
        while len(negs) < self.n_neg:
            c = self.rng.randrange(self.ds.num_items)
            if c not in seen:
                negs.append(c)
        return prompt, pos, negs, reward


def make_collate(tokenizer, max_len: int):
    def collate(batch):
        prompts, pos, negs, rewards = zip(*batch)
        enc = tokenizer(list(prompts), return_tensors="pt", padding=True,
                        truncation=True, max_length=max_len)
        cand = torch.tensor([[p] + n for p, n in zip(pos, negs)], dtype=torch.long)
        reward = torch.tensor(rewards, dtype=torch.float)
        return enc["input_ids"], enc["attention_mask"], cand, reward
    return collate


# --------------------------------------------------------------------------- #
# Phase 1 — domain adaptation (causal LM)
# --------------------------------------------------------------------------- #
def phase1_adapt(model: GenRec, tokenizer, ds: Dataset, device,
                 epochs=1, lr=5e-5, max_len=96, batch_size=8, budget=10):
    corpus = [verbalize_item(it) for it in ds.catalog.values()]
    corpus += [verbalize_history(ds.sequences[u][:-2], ds.catalog, budget=budget)
               for u in list(ds.sequences)[:2000]]
    random.shuffle(corpus)
    opt = torch.optim.AdamW(model.backbone.parameters(), lr=lr)
    model.train()
    for ep in range(epochs):
        total, n = 0.0, 0
        for i in range(0, len(corpus), batch_size):
            chunk = corpus[i:i + batch_size]
            enc = tokenizer(chunk, return_tensors="pt", padding=True,
                            truncation=True, max_length=max_len).to(device)
            logits = model.lm_logits(enc["input_ids"], enc["attention_mask"])
            loss = lm_loss(logits, enc["input_ids"], enc["attention_mask"])
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); n += 1
        print(f"[phase1] epoch {ep+1} lm_loss={total/max(n,1):.4f}")
    return model


# --------------------------------------------------------------------------- #
# Phase 2 — ranking post-training (three losses)
# --------------------------------------------------------------------------- #
def phase2_rank(model: GenRec, tokenizer, ds: Dataset, device,
                epochs=1, lr=1e-4, max_len=96, batch_size=8, budget=10,
                n_neg=8, weights=(1.0, 0.1, 0.5), use_lm=True, use_reward=True):
    a, b, c = weights
    examples = training_examples(ds)
    data = RankingData(ds, examples, budget=budget, n_neg=n_neg)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True,
                        collate_fn=make_collate(tokenizer, max_len))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for ep in range(epochs):
        agg = {"rank": 0.0, "lm": 0.0, "rew": 0.0}; n = 0
        for input_ids, attn, cand, reward in loader:
            input_ids, attn = input_ids.to(device), attn.to(device)
            cand, reward = cand.to(device), reward.to(device)

            uv = model.user_vector(input_ids, attn)
            scores = model.score_items(uv, cand)
            l_rank = ranking_loss(scores)
            l_lm = (lm_loss(model.lm_logits(input_ids, attn), input_ids, attn)
                    if use_lm else torch.zeros((), device=device))
            l_rew = (reward_weighted_loss(scores, reward)
                     if use_reward else torch.zeros((), device=device))
            loss = combined_loss(l_rank, l_lm, l_rew, a, b, c)

            opt.zero_grad(); loss.backward(); opt.step()
            agg["rank"] += l_rank.item(); agg["lm"] += l_lm.detach().item()
            agg["rew"] += l_rew.detach().item(); n += 1
        print(f"[phase2] epoch {ep+1} rank={agg['rank']/n:.4f} "
              f"lm={agg['lm']/n:.4f} reward={agg['rew']/n:.4f}")
    return model


# --------------------------------------------------------------------------- #
# GenRec as an eval scorer (prefill-only path)
# --------------------------------------------------------------------------- #
class GenRecScorer:
    def __init__(self, model, tokenizer, ds, device, budget=10, max_len=96,
                 which="test"):
        self.model, self.tok, self.ds, self.device = model, tokenizer, ds, device
        self.budget, self.max_len = budget, max_len
        self.hist = ds.train_hist if which == "test" else ds.val_hist
        model.eval()

    def scorer(self, uid, candidates):
        prompt = verbalize_history(self.hist[uid], self.ds.catalog, budget=self.budget)
        enc = self.tok([prompt], return_tensors="pt", padding=True,
                       truncation=True, max_length=self.max_len).to(self.device)
        cand = torch.tensor([candidates], dtype=torch.long, device=self.device)
        scores = self.model.rank(enc["input_ids"], enc["attention_mask"], cand)
        return scores[0].tolist()


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true", help="tiny GPT-2, no download")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--budget", type=int, default=10)
    ap.add_argument("--p1-epochs", type=int, default=1)
    ap.add_argument("--p2-epochs", type=int, default=2)
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device)

    ds = synthetic_dataset()
    lm, tok, hid = build_backbone(args.model, tiny=args.tiny)
    model = GenRec(lm, hid, num_items=ds.num_items, scorer="dot").to(device)

    from baselines import ItemKNNBaseline, PopularityBaseline
    from eval import evaluate, pretty
    exclude = {u: {it.item_id for it in ds.train_hist[u]} for u in ds.test}

    print("\n--- baselines ---")
    for name, m in [("Popularity", PopularityBaseline(ds, ds.train_hist)),
                    ("ItemKNN", ItemKNNBaseline(ds, ds.train_hist))]:
        print(pretty(name, evaluate(m.scorer, ds.test, ds.num_items,
                                    ks=(10,), exclude=exclude)))

    print("\n--- GenRec Phase 1 ---")
    phase1_adapt(model, tok, ds, device, epochs=args.p1_epochs, budget=args.budget)
    print("\n--- GenRec Phase 2 ---")
    phase2_rank(model, tok, ds, device, epochs=args.p2_epochs, budget=args.budget)

    print("\n--- GenRec eval ---")
    gr = GenRecScorer(model, tok, ds, device, budget=args.budget)
    print(pretty("GenRec", evaluate(gr.scorer, ds.test, ds.num_items,
                                    ks=(10,), exclude=exclude)))


if __name__ == "__main__":
    main()
