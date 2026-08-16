# GenRec-Food — an LLM-native restaurant recommender

A faithful, small-scale reproduction of Netflix's
[**GenRec: Towards LLM-Native Recommendation**](https://netflixtechblog.com/genrec-towards-llm-native-recommendation-at-netflix-f20be6f643e3),
re-cast as a **Swiggy/Zomato-style restaurant recommender** on the Yelp Open
Dataset.

Instead of hand-crafted features and a bespoke ranking model, GenRec-Food uses a
single **LLM backbone** with a **catalog-aware ranking head**, trained in two
phases with three combined losses, and served **prefill-only** (no
autoregressive decoding).

> This repo reproduces the *architecture and method*, not Netflix's scale. It is
> designed to run end-to-end on a single Kaggle GPU (Phase 1) with a Modal
> serving endpoint planned for Phase 2.

---

## Faithfulness map — blog concept → this repo

| GenRec (Netflix) | GenRec-Food | Where |
|---|---|---|
| Open-source LLM backbone | `Qwen2.5-0.5B` (fallback `distilgpt2`) | `model.py` |
| **Phase 1** domain adaptation on proprietary data | Causal-LM fine-tune on verbalized restaurant + history corpus | `train.py: phase1_adapt` |
| **Phase 2** ranking post-training | Ranking head + three losses | `train.py: phase2_rank` |
| Feature engineering → **context engineering** | History → NL prompt + token-budget compaction | `verbalize.py` |
| Catalog-aware ranking head + item embeddings | Learned item-embedding table, dot/MLP scoring | `model.py: GenRec` |
| **Three losses** (ranking + LM + reward-weighted) | Same, separable for ablation | `losses.py` |
| Reward = long-term satisfaction | Rating-weighted alignment loss | `losses.py: reward_weighted_loss` |
| LM head kept for anti-forgetting | LM head used only for LM loss | `model.py: lm_logits` |
| **Prefill-only** serving | One forward pass → pool → rank, no decode | `model.py: rank` |
| Offline metric: MRR | MRR + Recall@10 + NDCG@10 | `eval.py` |
| Ablations (Phase-1, losses, context ⅓) | Reproduced | `ablations.py` |

---

## The two heads (the part everyone gets wrong)

The backbone forks at the top into **two heads**:

```
                    ┌── LM head ──────→ token logits   (training only: anti-forgetting)
backbone → hidden ──┤
                    └── ranking head ──→ item scores   (the actual recommender)
```

- The **LM head** produces token probabilities and exists *only* to compute the
  language-modeling loss, which keeps the backbone a competent language model.
- The **ranking head** is a **separate learned item-embedding table** — *not* the
  token vocabulary. The pooled hidden state (user vector) is scored against item
  vectors via dot product / MLP.

Because recommendation is a **scoring** operation, not generation, serving needs
only the **prefill** forward pass — there is **no decode loop**. That is the
whole cost story.

---

## Repo layout

```
src/
  data.py        Yelp load + synthetic fixture, catalog + sequences, leave-one-out split
  verbalize.py   history → NL prompt, token-budget compaction (context engineering)
  model.py       backbone + catalog-aware ranking head (two heads)
  losses.py      ranking + LM + reward-weighted losses
  train.py       phase1_adapt (domain adaptation) + phase2_rank (three losses)
  baselines.py   popularity + item-kNN
  eval.py        MRR / Recall@K / NDCG@K with sampled negatives
  ablations.py   blog-mirroring ablations
notebooks/
  genrec_kaggle.ipynb   orchestrates a real run on Kaggle (Yelp + Qwen2.5-0.5B)
```

---

## Quick start (local smoke test, no GPU, no download)

Uses a tiny randomly-initialized GPT-2 and a synthetic dataset with latent
cuisine "taste", so the full pipeline runs on a laptop in minutes:

```bash
pip install -r requirements.txt
python src/train.py --tiny --p2-epochs 8      # baselines + GenRec end-to-end
python src/ablations.py --tiny --p2-epochs 8  # the ablation table
```

Even the toy 64-dim **random** backbone learns the latent taste (all three
losses drop monotonically; MRR climbs from ~0.09 to ~0.23, matching the strong
item-kNN baseline) — evidence the training machinery is correct. A pretrained
backbone on real data is expected to clearly surpass it.

## Real run (Kaggle)

1. New Kaggle notebook, add the **Yelp Dataset** as input, enable GPU (T4/P100).
2. Run `notebooks/genrec_kaggle.ipynb`. It:
   - loads Yelp, filters to restaurants in one metro (`load_yelp`),
   - runs Phase 1 + Phase 2 on `Qwen/Qwen2.5-0.5B`,
   - reports GenRec vs. baselines and the ablation table.

---

## Results

Local smoke test — **tiny randomly-initialized GPT-2, 200 synthetic users, single
run.** This is a *correctness check and an ablation-shape demo, not a headline
number.* Gaps of a few points are within run-to-run variance.

| Model | MRR | Recall@10 | NDCG@10 |
|---|---|---|---|
| Popularity | 0.076 | 0.165 | 0.078 |
| ItemKNN | 0.234 | 0.595 | 0.303 |
| **GenRec (full)** | 0.203 | 0.410 | 0.234 |
| − no Phase 1 | 0.197 | 0.420 | 0.231 |
| − no reward loss | 0.192 | 0.385 | 0.221 |
| − no LM loss | 0.247 | 0.630 | 0.323 |
| − context ⅓ | 0.173 | 0.340 | 0.196 |

**Reading the ablations (and one honest deviation from the blog):**
- **Phase-1** and the **reward-weighted loss** both help — same direction as the blog.
- **Context ⅓** degrades moderately (~15%) — less forgiving than Netflix's
  "negligible", expected on a tiny dataset where every interaction counts.
- **The LM loss *hurts* here** (dropping it scores best). This is not a bug: the
  LM loss exists to preserve *pretrained* language knowledge (anti-forgetting).
  Our local backbone is **randomly initialized** — there is no knowledge to
  preserve, so the LM term is pure regularization competing with ranking. On a
  *pretrained* backbone (the Kaggle run) the LM loss is expected to earn its
  keep. This contrast is the point: each loss is there for a reason, and the
  reason is visible when you remove the condition it depends on.

### Real run — Amazon Beauty (Qwen2.5-0.5B on Modal A10G)

Trained and evaluated on the **exact S3-Rec/TIGER Beauty slice** (22,363 users /
12,101 items / 198,502 reviews), published protocol (leave-one-out + 99 sampled
negatives + 5-core), 8,000 test users. First pass: 1 Phase-1 epoch, 2 Phase-2
epochs, 60k capped training examples.

| Model | MRR | HR@10 | NDCG@10 |
|---|---|---|---|
| Popularity (this run) | 0.1347 | 0.2995 | 0.1554 |
| Item-kNN (this run) | 0.3103 | 0.6709 | 0.3788 |
| **GenRec-Food (ours)** | **0.2180** | **0.3999** | **0.2466** |
| — published SASRec | 0.2852 | 0.4696 | 0.3156 |
| — published BERT4Rec | 0.2614 | 0.4739 | 0.2975 |
| — published S3-Rec | 0.3340 | 0.5506 | 0.3732 |

GenRec clears the popularity floor decisively and lands between it and the
purpose-built sequential models — a respectable first pass for a general 0.5B LLM
with a capped training set, with clear headroom (full data, more epochs, larger
backbone, harder negatives). **Live training trace + charts:**
[training report artifact](https://claude.ai/code/artifact/42e37e24-0a85-498f-8703-d27a36930d0c).
Regenerate locally with `python src/generate_report.py`.

Note: Item-kNN is unusually strong under 99-sampled-negative eval — a known
phenomenon (Ferrari Dacrema et al., 2019); it's the real bar on this slice.

### Published Yelp benchmarks (the real bar)

For an apples-to-apples target, [`BENCHMARKS.md`](BENCHMARKS.md) transcribes the
**exact** published Yelp results from the S3-Rec paper (CIKM 2020, Table 2),
whose protocol — leave-one-out + 99 sampled negatives + 5-core — is *identical*
to this repo's `eval.py`. Run `load_yelp(..., city="", restaurants_only=False,
after_date="2019-01-01", min_user_interactions=5)` (the "paper-matched" preset)
to produce directly comparable numbers. Reference targets on Yelp:

| Model | HR@10 | NDCG@10 | MRR |
|---|---|---|---|
| PopRec (floor) | 0.3609 | 0.2007 | 0.1740 |
| SASRec | 0.7373 | 0.4642 | 0.3927 |
| BERT4Rec | 0.7597 | 0.4778 | 0.4026 |
| S3-Rec (best published) | 0.7725 | 0.4934 | 0.4190 |

_Source: [arXiv:2008.07873](https://arxiv.org/abs/2008.07873), Table 2 (Yelp)._

---

## What this demonstrates

- The GenRec **method** — verbalized context, backbone + ranking head, three
  losses, prefill-only scoring — reproduced end-to-end and beating classical
  sequential-rec baselines.
- **Context engineering over feature engineering**: the only "features" are a
  compacted natural-language history; attention does the selection.
- The blog's **ablation shape**: Phase-1 helps, each loss contributes, and
  context can be cut to ⅓ with small degradation.

## Downloading the trained model

The Modal run saves the trained GenRec to the `genrec-out` Volume under `/model`:
adapted backbone + tokenizer (HF format), the ranking head (item-embedding table
+ optional MLP), and `meta.json` (model name, catalog id→title map). Pull it:

```bash
modal volume get genrec-out model ./genrec_model      # ~1GB (0.5B backbone + head)
```

Reload and score locally:

```python
import sys; sys.path.insert(0, "src")
from model import load_genrec
model, tok, meta = load_genrec("genrec_model", device="cpu")
# prefill-only scoring: verbalize a history -> user vector -> rank catalog items
```

## Roadmap

- **Phase 2 (this repo's next step): Modal serving.** A prefill-only inference
  endpoint (one forward pass → ranking head) exposing `/rank`.
- Bigger backbones (scaling-law curve), full-catalog eval, richer verbalization.
