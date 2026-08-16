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

import contextlib

import numpy as np

from data import Dataset, synthetic_dataset
from losses import combined_loss, lm_loss, ranking_loss, reward_weighted_loss
from model import GenRec, build_backbone
from rewards import build_reward_fn, neutral_reward_fn
from verbalize import verbalize_history, verbalize_item, verbalize_target


# --------------------------------------------------------------------------- #
# Training examples: prefix -> next-item, excluding held-out val/test targets.
# --------------------------------------------------------------------------- #
def training_examples(ds: Dataset, min_prefix: int = 1, max_examples: int | None = None,
                      reward_fn=None):
    """Yield (uid, history_prefix, positive_interaction, reward). Targets are
    seq[1 .. len-3] so we never train on the val (seq[-2]) or test (seq[-1])
    items — no leakage.

    `reward_fn` maps the positive interaction to the scalar example weight (see
    rewards.py: long-term satisfaction x behaviour rebalancing).

    max_examples: if set, randomly subsample to this many (keeps a large real
    run tractable). eval is always on the full/holdout, so this only trades
    training signal for wall-clock, not evaluation integrity."""
    reward_fn = reward_fn or neutral_reward_fn()
    ex = []
    for uid, seq in ds.sequences.items():
        # last two are held out for val/test
        for t in range(min_prefix, len(seq) - 2):
            hist = seq[:t]
            pos = seq[t]
            ex.append((uid, hist, pos, reward_fn(pos)))
    if max_examples and len(ex) > max_examples:
        import random as _r
        ex = _r.Random(0).sample(ex, max_examples)
    return ex


class NegativeSampler:
    """Popularity-sampled negatives.

    Uniform negatives out of a 12k catalog are trivially separable, so the
    ranking loss saturates long before the 99-negative eval task does. Sampling
    proportional to popularity^0.75 (the word2vec exponent) puts real head items
    in the candidate set, which is much closer to the blog's "cross-entropy loss
    over the catalog or candidate set". We mix in uniform draws so the tail is
    still represented.
    """

    def __init__(self, ds: Dataset, pop_frac: float = 0.5, exponent: float = 0.75):
        from rewards import item_popularity
        train_seqs = {u: s[:-2] for u, s in ds.sequences.items() if len(s) > 2}
        pop = item_popularity(train_seqs)
        w = np.array([pop.get(i, 0) for i in range(ds.num_items)], dtype=np.float64)
        w = np.power(w, exponent)
        if w.sum() <= 0:
            w = np.ones_like(w)
        self.cum = np.cumsum(w / w.sum())
        self.n = ds.num_items
        self.pop_frac = pop_frac

    def draw(self, rng: random.Random, k: int, blocked: set) -> list[int]:
        out: list[int] = []
        guard = 0
        while len(out) < k and guard < 50 * k:
            guard += 1
            if rng.random() < self.pop_frac:
                c = int(np.searchsorted(self.cum, rng.random()))
                c = min(c, self.n - 1)
            else:
                c = rng.randrange(self.n)
            if c not in blocked:
                out.append(c)
        while len(out) < k:                       # pathological fallback
            out.append(rng.randrange(self.n))
        return out


class RankingData(TorchDataset):
    """One Phase-2 example = a single-turn conversation.

    user turn      : V(H, tau, item metadata) + the task
    assistant turn : the member's actual engagement (which item, how they rated it)

    The ranking head reads the user turn only; the LM objective reads both.
    """

    def __init__(self, ds: Dataset, examples, budget: int, n_neg: int, seed=0,
                 sampler: NegativeSampler | None = None):
        self.ds = ds
        self.examples = examples
        self.budget = budget
        self.n_neg = n_neg
        self.rng = random.Random(seed)
        self.sampler = sampler

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        uid, hist, pos, reward = self.examples[i]
        prompt = verbalize_history(hist, self.ds.catalog, budget=self.budget,
                                   now_ts=pos.ts)
        response = verbalize_target(self.ds.catalog[pos.item_id], pos.rating)
        blocked = {pos.item_id} | {it.item_id for it in hist}
        if self.sampler is not None:
            negs = self.sampler.draw(self.rng, self.n_neg, blocked)
        else:
            negs = []
            while len(negs) < self.n_neg:
                c = self.rng.randrange(self.ds.num_items)
                if c not in blocked:
                    negs.append(c)
        return prompt, response, pos.item_id, negs, reward


def amp_ctx(device, enabled: bool = True):
    """bf16 autocast on CUDA.

    At max_len=384 the logits tensor alone is [16, 384, 151936] — 3.7 GB in fp32,
    and the LM cross-entropy backward wants another copy. bf16 halves that and
    roughly doubles throughput, which is what pays for the longer prompts. Master
    weights stay fp32, and bf16 needs no GradScaler.
    """
    if enabled and torch.cuda.is_available() and str(device).startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def make_collate(tokenizer, max_len: int, resp_max: int = 48,
                 with_response: bool = True):
    """Tokenize the user turn and the assistant turn SEPARATELY, then concatenate
    ids, so we know exactly where the prompt ends.

    Returns (input_ids, attention_mask, pool_mask, candidates, reward).

    `pool_mask` covers the user turn only. Pooling the user vector from a
    single forward over [prompt + response] is identical to a prompt-only
    forward because attention is causal, so we get the LM-over-inputs-and-outputs
    objective and the ranking objective for the price of ONE backbone pass, with
    no possibility of the answer leaking into the user vector.
    """
    pad_id = tokenizer.pad_token_id or 0

    def collate(batch):
        prompts, responses, pos, negs, rewards = zip(*batch)
        p_ids = tokenizer(list(prompts), truncation=True,
                          max_length=max_len)["input_ids"]
        if with_response:
            r_ids = tokenizer(list(responses), truncation=True,
                              max_length=resp_max)["input_ids"]
        else:
            r_ids = [[] for _ in p_ids]

        full = [p + r for p, r in zip(p_ids, r_ids)]
        T = max(len(f) for f in full)
        ids = torch.full((len(full), T), pad_id, dtype=torch.long)
        attn = torch.zeros((len(full), T), dtype=torch.long)
        pool = torch.zeros((len(full), T), dtype=torch.long)
        for i, (f, p) in enumerate(zip(full, p_ids)):
            ids[i, :len(f)] = torch.tensor(f, dtype=torch.long)
            attn[i, :len(f)] = 1
            pool[i, :len(p)] = 1

        cand = torch.tensor([[p] + n for p, n in zip(pos, negs)], dtype=torch.long)
        reward = torch.tensor(rewards, dtype=torch.float)
        return ids, attn, pool, cand, reward
    return collate


# --------------------------------------------------------------------------- #
# Phase 1 — domain adaptation (causal LM)
# --------------------------------------------------------------------------- #
def phase1_adapt(model: GenRec, tokenizer, ds: Dataset, device,
                 epochs=1, lr=5e-5, max_len=384, batch_size=8, budget=10,
                 tracker=None, log_every=20, val_every=100, gstep0=0,
                 n_hist_users=4000, amp=True):
    corpus = [verbalize_item(it) for it in ds.catalog.values()]
    # random sample of users, not the first N by load order
    users = list(ds.sequences)
    if len(users) > n_hist_users:
        users = random.Random(0).sample(users, n_hist_users)
    corpus += [verbalize_history(ds.sequences[u][:-2], ds.catalog, budget=budget,
                                 now_ts=ds.sequences[u][-2].ts)
               for u in users if len(ds.sequences[u]) > 2]
    random.shuffle(corpus)
    n_val = max(batch_size, len(corpus) // 20)          # ~5% held out
    val_corpus, corpus = corpus[:n_val], corpus[n_val:]

    if tracker is not None and tracker.tokens_per_byte is None:
        from tracker import tokens_per_byte
        tracker.tokens_per_byte = tokens_per_byte(corpus, tokenizer)
        print(f"[phase1] tokens/byte={tracker.tokens_per_byte:.3f}")

    def _val_lm_loss():
        model.eval()
        with torch.no_grad():
            tot, k = 0.0, 0
            for i in range(0, len(val_corpus), batch_size):
                enc = tokenizer(val_corpus[i:i + batch_size], return_tensors="pt",
                                padding=True, truncation=True,
                                max_length=max_len).to(device)
                with amp_ctx(device, amp):
                    tot += lm_loss(
                        model.lm_logits(enc["input_ids"], enc["attention_mask"]),
                        enc["input_ids"], enc["attention_mask"]).item()
                k += 1
        model.train()
        return tot / max(k, 1)

    opt = torch.optim.AdamW(model.backbone.parameters(), lr=lr)
    model.train()
    gstep = gstep0
    for ep in range(epochs):
        total, n = 0.0, 0
        for i in range(0, len(corpus), batch_size):
            chunk = corpus[i:i + batch_size]
            enc = tokenizer(chunk, return_tensors="pt", padding=True,
                            truncation=True, max_length=max_len).to(device)
            with amp_ctx(device, amp):
                logits = model.lm_logits(enc["input_ids"], enc["attention_mask"])
                loss = lm_loss(logits, enc["input_ids"], enc["attention_mask"])
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); n += 1; gstep += 1

            if tracker is not None and gstep % log_every == 0:
                tracker.log(gstep, "phase1_train", **tracker.lm_stats(loss.item()))
            if tracker is not None and gstep % val_every == 0:
                vl = _val_lm_loss()
                tracker.log(gstep, "phase1_val",
                            **{("val_" + k): v for k, v in tracker.lm_stats(vl).items()})
        print(f"[phase1] epoch {ep+1} lm_loss={total/max(n,1):.4f}")
    return model, gstep


# --------------------------------------------------------------------------- #
# Phase 2 — ranking post-training (three losses)
# --------------------------------------------------------------------------- #
def phase2_rank(model: GenRec, tokenizer, ds: Dataset, device,
                epochs=1, backbone_lr=2e-5, head_lr=1e-3, max_len=384,
                batch_size=8, budget=10, n_neg=16, weights=(1.0, 0.1),
                use_lm=True, use_reward=True, max_examples=None, tracker=None,
                log_every=20, val_every=200, val_users=400, gstep0=0,
                hard_negatives=True, grad_clip=1.0, amp=True, grad_accum=1):
    """Phase-2 post-training.

    One backbone forward per step yields both the pooled user vector (from the
    user-turn positions) and the token logits (over the whole conversation), so
    the ranking objective and the LM objective share a single pass.

    Total = a * (reward-scaled ranking CE) + b * (LM CE over input and output).
    """
    a, b = weights
    reward_fn = build_reward_fn(ds) if use_reward else neutral_reward_fn()
    examples = training_examples(ds, max_examples=max_examples, reward_fn=reward_fn)
    rw = [e[3] for e in examples]
    print(f"[phase2] {len(examples)} training examples | reward "
          f"min={min(rw):.2f} mean={sum(rw)/len(rw):.2f} max={max(rw):.2f}")
    sampler = NegativeSampler(ds) if hard_negatives else None
    data = RankingData(ds, examples, budget=budget, n_neg=n_neg, sampler=sampler)
    collate = make_collate(tokenizer, max_len)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True,
                        collate_fn=collate)

    # --- fixed validation batch (val_hist -> val item) for a stable val loss ---
    val_batch = None
    val_sample = []
    if tracker is not None:
        from eval import evaluate
        val_ids = [u for u in ds.val if u in ds.val_hist and ds.val_hist[u]]
        val_sample = random.Random(1).sample(val_ids, min(val_users, len(val_ids)))
        vb = RankingData(
            ds,
            [(u, ds.val_hist[u], ds.sequences[u][-2], 1.0)
             for u in val_sample[:min(256, len(val_sample))]],
            budget=budget, n_neg=n_neg, seed=1, sampler=sampler)
        val_batch = collate([vb[i] for i in range(len(vb))])

    def _val_metrics():
        model.eval()
        vi, va, vp, vc, _ = (t.to(device) for t in val_batch)
        with torch.no_grad(), amp_ctx(device, amp):
            uv = model.user_vector(vi, va, pool_mask=vp)
            sc = model.score_items(uv, vc)
            vloss = ranking_loss(sc).item()
            sc_val = GenRecScorer(model, tokenizer, ds, device, budget=budget,
                                  max_len=max_len, which="val")
            excl = {u: {it.item_id for it in ds.val_hist[u]} for u in val_sample}
            m = evaluate(sc_val.scorer, {u: ds.val[u] for u in val_sample},
                         ds.num_items, ks=(10,), exclude=excl)
        model.train()
        return vloss, m

    # Separate learning rates: the backbone is pretrained and wants a small step,
    # the item table and projection start from noise and want a large one.
    opt = torch.optim.AdamW(model.param_groups(backbone_lr, head_lr))
    # gstep counts OPTIMIZER steps, so curves stay comparable across runs that
    # split the same effective batch differently (batch_size x grad_accum).
    eff = batch_size * grad_accum
    total_steps = max(1, epochs * ((len(data) + eff - 1) // eff))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[backbone_lr, head_lr], total_steps=total_steps,
        pct_start=0.05, anneal_strategy="cos")
    model.train()
    gstep = gstep0
    micro = 0
    opt.zero_grad(set_to_none=True)
    for ep in range(epochs):
        agg = {"rank": 0.0, "lm": 0.0, "rew": 0.0}; n = 0
        for input_ids, attn, pool, cand, reward in loader:
            input_ids, attn, pool = input_ids.to(device), attn.to(device), pool.to(device)
            cand, reward = cand.to(device), reward.to(device)

            # ---- ONE forward: pooled user vector + token logits ----
            with amp_ctx(device, amp):
                uv, logits = model.forward_shared(input_ids, attn, pool_mask=pool)
                scores = model.score_items(uv, cand)

                l_rank = ranking_loss(scores)                 # logged, unweighted
                l_rew = reward_weighted_loss(scores, reward)
                rank_term = l_rew if use_reward else l_rank   # blog: ONE ranking loss
                l_lm = (lm_loss(logits, input_ids, attn) if use_lm
                        else torch.zeros((), device=device))
                loss = combined_loss(rank_term, l_lm, a, b)

            (loss / grad_accum).backward()
            agg["rank"] += l_rank.item(); agg["lm"] += l_lm.detach().item()
            agg["rew"] += l_rew.detach().item(); n += 1
            micro += 1
            if micro % grad_accum:
                continue                       # keep accumulating, no step yet

            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            if sched.last_epoch < total_steps - 1:
                sched.step()
            gstep += 1

            if tracker is not None and gstep % log_every == 0:
                rec = {"loss_total": loss.item(), "loss_rank": l_rank.item(),
                       "loss_reward": l_rew.detach().item()}
                rec.update(tracker.lm_stats(l_lm.detach().item()))
                tracker.log(gstep, "phase2_train", **rec)
            if tracker is not None and val_batch is not None and gstep % val_every == 0:
                vloss, vm = _val_metrics()
                tracker.log(gstep, "phase2_val", val_loss_rank=vloss,
                            val_MRR=vm["MRR"], val_Recall_10=vm["Recall@10"],
                            val_NDCG_10=vm["NDCG@10"])
                print(f"[phase2] step {gstep} val_rank={vloss:.4f} "
                      f"val_MRR={vm['MRR']:.4f}", flush=True)
        print(f"[phase2] epoch {ep+1} rank={agg['rank']/n:.4f} "
              f"lm={agg['lm']/n:.4f} reward={agg['rew']/n:.4f}", flush=True)
    return model, gstep


# --------------------------------------------------------------------------- #
# GenRec as an eval scorer (prefill-only path)
# --------------------------------------------------------------------------- #
class GenRecScorer:
    """Serving path: prefill once, pool, score the whole candidate set. No decode.

    The assistant turn exists only at training time; at serving there is nothing
    to condition on past the user turn, so the sequence is the prompt alone.
    """

    def __init__(self, model, tokenizer, ds, device, budget=10, max_len=384,
                 which="test", amp=True):
        self.model, self.tok, self.ds, self.device = model, tokenizer, ds, device
        self.budget, self.max_len, self.amp = budget, max_len, amp
        self.hist = ds.train_hist if which == "test" else ds.val_hist
        # prediction time = timestamp of the held-out target (drives the tau block)
        self.pos = -1 if which == "test" else -2
        model.eval()

    def scorer(self, uid, candidates):
        seq = self.ds.sequences[uid]
        now_ts = seq[self.pos].ts if len(seq) >= abs(self.pos) else None
        prompt = verbalize_history(self.hist[uid], self.ds.catalog,
                                   budget=self.budget, now_ts=now_ts)
        enc = self.tok([prompt], return_tensors="pt", padding=True,
                       truncation=True, max_length=self.max_len).to(self.device)
        cand = torch.tensor([candidates], dtype=torch.long, device=self.device)
        with amp_ctx(self.device, self.amp):
            scores = self.model.rank(enc["input_ids"], enc["attention_mask"], cand)
        return scores[0].float().tolist()


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
    model = GenRec(lm, hid, num_items=ds.num_items, scorer="dot",
                   pooling="mean", use_item_text=True).to(device)

    from baselines import ItemKNNBaseline, PopularityBaseline
    from eval import evaluate, pretty
    exclude = {u: {it.item_id for it in ds.train_hist[u]} for u in ds.test}

    print("\n--- baselines ---")
    for name, m in [("Popularity", PopularityBaseline(ds, ds.train_hist)),
                    ("ItemKNN", ItemKNNBaseline(ds, ds.train_hist))]:
        print(pretty(name, evaluate(m.scorer, ds.test, ds.num_items,
                                    ks=(10,), exclude=exclude)))

    from tracker import Tracker
    tracker = Tracker("artifacts/history_local.jsonl")
    print("\n--- GenRec Phase 1 ---")
    _, gstep = phase1_adapt(model, tok, ds, device, epochs=args.p1_epochs,
                            budget=args.budget, tracker=tracker,
                            log_every=10, val_every=40)
    # ground item vectors in the ADAPTED backbone's text understanding (cold start)
    model.build_item_text_embeddings(tok, ds.catalog, verbalize_item, device)
    print("\n--- GenRec Phase 2 ---")
    _, gstep = phase2_rank(model, tok, ds, device, epochs=args.p2_epochs,
                           budget=args.budget, tracker=tracker, gstep0=gstep,
                           log_every=10, val_every=60, val_users=100)
    tracker.close()
    print(f"[local] logged {len(tracker.history())} records to artifacts/history_local.jsonl")

    print("\n--- GenRec eval ---")
    gr = GenRecScorer(model, tok, ds, device, budget=args.budget)
    print(pretty("GenRec", evaluate(gr.scorer, ds.test, ds.num_items,
                                    ks=(10,), exclude=exclude)))


if __name__ == "__main__":
    main()
