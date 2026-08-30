# GenRec v2: Faithfulness Audit, Fixes, and Results

A strict re-read of the Netflix
[GenRec blog post](https://netflixtechblog.com/genrec-towards-llm-native-recommendation-at-netflix-f20be6f643e3)
against this repo, the fixes that came out of it, and a like-for-like rerun on
Amazon Beauty.

**Nothing about the training budget changed between the two runs.** Same backbone
(Qwen2.5-0.5B), same 1 Phase-1 epoch, same 2 Phase-2 epochs, same 60,000 capped
training examples, same 8,000 test users, same A10G, same effective batch of 16.
Every difference below is a fidelity or correctness fix, not extra compute.

---

## TL;DR

| Model | MRR | HR@10 | NDCG@10 |
|---|---|---|---|
| Popularity | 0.1347 | 0.2995 | 0.1554 |
| Item-kNN | 0.3103 | 0.6709 | 0.3788 |
| **GenRec v1 (before)** | 0.2180 | 0.3999 | 0.2466 |
| **GenRec v2 (after)** | **0.2752** | **0.4798** | **0.3101** |
| | **+26.2%** | **+20.0%** | **+25.7%** |

Against the published numbers on the same slice and protocol:

| Model | MRR | HR@10 | NDCG@10 | v2 verdict |
|---|---|---|---|---|
| **GenRec v2 (ours)** | **0.2752** | **0.4798** | **0.3101** | |
| BERT4Rec | 0.2614 | 0.4739 | 0.2975 | **we beat it on all three** |
| SASRec | 0.2852 | 0.4696 | 0.3156 | we win HR@10, just short on MRR/NDCG |
| S3-Rec | 0.3340 | 0.5506 | 0.3732 | still ahead of us |

v1 lost to all three. v2 beats BERT4Rec outright and is at rough parity with
SASRec, on a general-purpose 0.5B LLM with a capped training set.

---

## Part 1: What the audit found

### 1a. Architecture (mostly correct in v1)

The blog's formulation: a verbalizer `V` serializes user history `H`, context
`tau`, and item metadata into text `x`; the LLM produces "a pooled hidden state
`h`"; each catalog item has a learned embedding `e_i`; a scoring head `phi`
combines them "via dot product or small MLP"; and "all parameters, the backbone,
scoring head, and item embeddings, are trained jointly."

| Blog element | v1 status |
|---|---|
| Decoder-only backbone | Pass |
| Catalog-aware head scoring only in-catalog items | Pass |
| `phi(h, e_i)` as dot product or small MLP | Pass |
| Backbone + head + item embeddings trained jointly | Pass |
| Prefill-only scoring, no decode loop | Pass |
| Pooled hidden state `h` | Partial |
| Verbalizer `V(H, tau, metadata)` | Partial, `tau` absent |
| Cold-start item handling | **Fail** |

The skeleton was right. That is the part most reproductions get wrong.

### 1b. Phase 1 (correct)

Backbone-only causal-LM adaptation on verbalized catalog items plus verbalized
histories. Leakage check passed: the corpus uses `sequences[u][:-2]`, so val and
test targets never appear.

### 1c. Phase 2 (four real divergences)

1. **The reward weight was additive, not multiplicative.**
   The blog: "The example's ranking loss **is scaled by this weight**."
   v1 computed `1.0 * CE + 0.1 * LM + 0.5 * (w * CE)`, so the effective per-example
   weight was `1 + 0.5w`. A 5-star engagement got about **1.4x** the gradient of a
   1-star one, where the blog's formulation gives 5x. The reward objective was
   largely inert.

2. **The reward carried one of the blog's two signal families.**
   The blog names "two types of signals: long-term satisfaction proxies" and
   "behaviour rebalancing." v1 used the raw star rating only. No popularity
   debiasing, and a star rating is a point-in-time signal, not a long-term one.

3. **The LM objective covered inputs only, and there were no outputs.**
   The blog puts the LM objective "over the verbalized **inputs and outputs**,"
   with Phase-2 data as conversations whose assistant message is "the member's
   actual engagement," so that "the LLM learns how assistant messages depend on
   user messages." v1 had no assistant turn at all. Its LM term was pure
   anti-forgetting and carried none of the supervision the blog assigns it.

4. **Item vectors were pure ID embeddings.**
   A brand-new item was a random vector forever. Cold start is the single clearest
   place an LLM-native recommender should beat a classical one, and it was absent.

### 1d. The bug that undercut the v1 numbers

**`max_len=96` silently truncated away the back half of every prompt.**

The tracker's own ratio (`bits_per_byte / bits_per_token` = 0.295) gives about 3.4
characters per token. A budget-10 Beauty prompt runs roughly 850 characters, so
about **250 tokens**. HuggingFace truncates from the right, so at 96 tokens the
model never saw:

* the most recent interactions, the highest-signal part of a sequential prompt, and
* the call-to-action `"Recommend the next product this shopper will buy."`

Consequences: the reported 0.2180 came from a model reading roughly 40% of its
intended context, and the context-budget ablation was confounded (budget 10
truncated hard, budget 3 fit uncompromised, so it compared a truncated long prompt
against a complete short one).

### 1e. Training-quality issues

* **Double forward pass per step.** `user_vector` already returned logits from its
  own call, then `lm_logits` ran the backbone a second time. 2x forward and 2x
  backward through a 0.5B model, every step, for nothing.
* **One learning rate (1e-4) for both** a pretrained 0.5B backbone (wants ~2e-5)
  and a randomly initialized 12,101 x 896 item table (wants ~1e-3).
* **Train/eval negative mismatch.** 8 uniform negatives at train time, 99 at eval.
  Uniform negatives from a 12k catalog are near-trivial.
* **Ablations never ran on real data.** `ablations.py` hardcoded the synthetic set,
  so the README's claim to mirror the blog's ablations had no real-data backing.

---

## Part 2: What changed in v2, and why

| # | Change | Why | Where |
|---|---|---|---|
| 1 | `max_len` 96 to 384, plus a 64-char title cap | Nothing gets right-truncated; the token budget becomes the only thing controlling context size | `train.py`, `verbalize.py` |
| 2 | Reward **scales** the ranking loss instead of sitting beside it | The blog's literal formulation, one ranking loss weighted per example | `losses.py` |
| 3 | Reward = long-term satisfaction x behaviour rebalancing | The blog's "two types of signals" | `rewards.py` (new) |
| 4 | Assistant turn added; LM loss over input **and** output | The blog's conversational Phase-2 format | `verbalize.py`, `train.py` |
| 5 | Item vector `e_i` = ID embedding + projected frozen text embedding | Gives cold-start items a meaningful vector; grounds the item side in the backbone's content understanding | `model.py` |
| 6 | Single forward yields both pooled vector and logits | Removes the duplicated backbone pass | `model.py: forward_shared` |
| 7 | Parameter groups: backbone 2e-5, head 1e-3, no weight decay on the item table | Two very different parameter populations; decaying a sparse item table re-creates the popularity bias the reward is undoing | `model.py: param_groups` |
| 8 | Negatives: 16, popularity-sampled at `pop^0.75`, half uniform | Closer to "cross-entropy over the catalog or candidate set", and actually hard | `train.py: NegativeSampler` |
| 9 | Context `tau` added to the prompt | Implements the `tau` argument of `V(H, tau, metadata)` | `verbalize.py: verbalize_context` |
| 10 | Scoring path never touches the LM head | Prefill-only, literally: pool a hidden state, do not predict a token | `model.py: _encode` |
| 11 | Ablations run on real data, with new arms | The blog's ablations deserve real-data numbers | `ablations.py`, `modal_app.py: ablate` |
| 12 | Phase-1 history corpus randomly sampled | v1 took the first 2,000 users by load order | `train.py` |

### Detail on the reward (change 2 and 3)

The new scalar per example is `satisfaction x rebalancing`, clipped to [0.25, 3.0]:

* **Satisfaction** blends the star rating with an *engagement* proxy derived from
  the raw review: review length (log-scaled, saturating at ~600 chars) and helpful
  votes (Bayesian-smoothed, confidence-weighted by vote count). This is the closest
  Amazon analogue to Netflix's watch duration and retention signals.
* **Rebalancing** is damped inverse propensity, `(mean_pop / pop_i) ^ 0.5`.

Measured on the real 60,000-example Beauty training set:

| | effective weight ratio, best example vs worst |
|---|---|
| v1 (`1 + 0.5w`) | about **1.4x** |
| v2 (min 0.25, mean 1.14, max 3.00) | **12x** |

### Detail on the shared forward (change 6 and 10)

The Phase-2 sequence is `[user turn + assistant turn]`. The user vector is pooled
from the user-turn positions only. Because attention is causal, no prompt position
can attend to a later position, so this is **exactly** a prompt-only forward.
Verified empirically:

```
prompt-only vs pooled-from-full   max abs diff = 4.768e-07   (float noise)
naive full-sequence pool          differs by     2.105e+00   (so the test has teeth)
```

That gives the LM-over-inputs-and-outputs objective and the ranking objective for
the price of one backbone pass, with no possibility of the answer leaking into the
user vector.

---

## Part 3: Results

### 3a. Test set, 8,000 users, leave-one-out, 99 sampled negatives

| Metric | v1 | v2 | change |
|---|---|---|---|
| MRR | 0.2180 | **0.2752** | **+26.2%** |
| Recall@5 | 0.3001 | **0.3748** | +24.9% |
| NDCG@5 | 0.2143 | **0.2761** | +28.8% |
| Recall@10 | 0.3999 | **0.4798** | +20.0% |
| NDCG@10 | 0.2466 | **0.3101** | +25.7% |

### 3b. Data efficiency

Validation MRR against Phase-2 optimizer steps, aligned so both runs start Phase 2
at step 0 (they have the same 3,750 steps per epoch since the effective batch is
identical).

| Reach this validation MRR | v1 steps | v2 steps | speedup |
|---|---|---|---|
| 0.15 | 1,180 | 300 | **3.9x** |
| 0.20 | 2,380 | 900 | **2.6x** |
| 0.2180 (v1's *final test* score) | 3,180 | 1,500 | **2.1x** |
| 0.25 | 5,980 | 2,300 | **2.6x** |
| best reached in the whole run | 0.2549 | **0.3022** | |

v2 passed v1's best-ever validation point at about a third of the training steps.
This is the same direction as the blog's own data-efficiency claim of "10 to 40x
fewer Phase-2 labeled training examples," at a much smaller scale.

### 3c. Training health

| | v1 | v2 |
|---|---|---|
| Final Phase-2 ranking loss | 0.6487 | 0.1921 |
| Final LM loss | 1.8672 | 1.0736 |
| Final LM perplexity | 6.47 | 2.93 |

Note the ranking losses are not directly comparable (v1 scored against 9
candidates, v2 against 17, so chance is 2.197 vs 2.833). The LM figures are
comparable and show the anti-forgetting leash holding better, which is expected
now that the LM objective has real supervision (the assistant turn) instead of
only the prompt.

---

## Part 4: Found during the rerun

Three things surfaced only when the fixed code actually ran. All are worth
recording because two of them are conclusions that contradict my own initial
recommendation.

### 4a. Last-token pooling was the wrong call here

The audit flagged mean pooling as a weak choice for a causal decoder and
recommended last-token pooling, since only the final position has attended to the
whole prompt. Measured on the toy setup, this was **wrong for this design**:

| pooling | toy MRR |
|---|---|
| mean | **0.2251** |
| last token | 0.0888 |

The reason: every prompt ends with the *same* call-to-action, so the final hidden
state is dominated by identical tokens and barely differentiates users. The blog
only says "a pooled hidden state", so both are in spec. v2 therefore keeps mean
pooling (which also matches v1, isolating the other fixes), and `pooling=last` is
now an **ablation arm** so a pretrained backbone can settle it on real data.

### 4b. Two out-of-memory failures, both real defects

* **OOM 1:** at 384 tokens the logits tensor is ~3.5 GB in fp32, four times v1's
  footprint, and cross-entropy upcasts logits even under bf16 autocast. Fixed with
  gradient accumulation: micro-batch 4 with accumulation 4, holding the effective
  batch at 16 exactly as in v1. Verified equivalent locally (final loss 2.5840 vs
  2.5873 against a same-effective-batch run).
* **OOM 2:** a single 25 GiB allocation. The validation step ran 256 examples in
  one forward, and `user_vector` was invoking the full causal LM, computing the
  151,936-wide token logits it never uses. Fixed by giving the scoring path a
  decoder-only encode (change 10 above), which is both the memory fix and the more
  faithful reading of prefill-only serving. Verified bit-identical to the training
  path (diff 0.0), so train and eval cannot silently diverge.

Neither was papered over by shortening the sequence, which would have reintroduced
the truncation bug that started the whole audit.

### 4c. MPS is a bad smoke-test target

The tiny local smoke test ran at 2,800 ms/step on Apple MPS versus 106 ms/step on
plain CPU for the same model, a 25x slowdown. Use `--tiny` on CPU.

---

## Part 5: Still open, and honest caveats

1. **Three of the twelve changes actively HURT offline MRR.** See Part 7: the
   ablations have now run, and popularity-sampled negatives, mean pooling, and
   reward weighting each cost measurable MRR. The v2 config is therefore not the
   best available config, only a more faithful one.
2. **Cold start is implemented but not measured.** Every item in the Beauty 5-core
   slice appears at least 5 times, so this benchmark structurally cannot test the
   cold-start path. A held-out-items evaluation would be needed.
3. **Reward rebalancing may cost offline MRR.** Inverse-propensity weighting
   deliberately down-weights popular positives, but the test set is drawn from the
   same popularity-biased distribution. The blog does this for long-term business
   reasons that offline MRR does not capture. The `no reward weighting` ablation
   arm will quantify the cost.
4. **Item-kNN still wins on this slice** (MRR 0.3103). That is a known phenomenon
   under 99-sampled-negative evaluation (Ferrari Dacrema et al., 2019) and it
   remains the real bar here.
5. **Multi-turn conversations are still absent.** The blog mentions "single-turn or
   multi-turn" conversations; v2 implements single-turn only.
6. **`tau` is thin.** These datasets carry no device or daypart, so the context
   block uses history span, density, and staleness at prediction time. That is
   what the logs actually contain.

---

## Part 6: Reproducing

```bash
modal deploy modal_app.py
```

```bash
python -c "import modal; modal.Function.from_name('genrec-food','train').spawn(dataset='amazon_beauty', run_tag='v2')"
```

Artifacts land on the `genrec-out` volume under `v2/`, leaving v1 at the root
untouched for comparison:

```bash
modal volume get genrec-out v2/model ./genrec_model_v2
```

To run the real-data ablations (separate multi-hour job):

```bash
python -c "import modal; modal.Function.from_name('genrec-food','ablate').spawn(dataset='amazon_beauty', run_tag='v2')"
```

Local smoke test, on CPU rather than MPS:

```bash
python src/train.py --tiny --p2-epochs 3
```

---

## Part 7: Real-data ablations

Run on Amazon Beauty at a reduced budget (20,000 training examples, 1 Phase-2
epoch, 2,000 eval users) so eight arms fit one job. Phase 1 is trained once and
shared by every arm that uses it. **Compare arms against each other, not against
the 0.2752 headline**, which used 3x the data and 2x the epochs.

Raw results: [`docs/v2/ablations.json`](docs/v2/ablations.json).

| arm | MRR | vs full | what the removed thing was worth |
|---|---|---|---|
| **GenRec (full)** | **0.2140** | | |
| no item text | 0.1552 | **-27.5%** | item text is the biggest single win |
| no Phase 1 | 0.1946 | **-9.0%** | Phase 1 worth +10.0% |
| no LM loss | 0.2221 | +3.8% | small ranking cost |
| context 1/3 | 0.2212 | +3.4% | cheaper AND slightly better |
| last-token pooling | 0.2298 | +7.4% | mean pooling was the wrong default |
| uniform negatives | 0.2393 | +11.8% | popularity sampling was a mistake |
| no reward weighting | 0.2436 | +13.8% | faithful, but expensive |
| Popularity | 0.1404 | | |
| Item-kNN | 0.3118 | | |

### What this confirms from the blog

* **Phase 1 is worth +10.0%.** The blog claims Phase-1 adaptation beats a raw
  open-source backbone by "10-20%". Dead on the lower bound.
* **Context to one third costs nothing.** The blog reports "negligible
  degradation" at roughly one-third the tokens. We measured a small *gain*
  (+3.4%), and [`SERVING.md`](SERVING.md) shows it also halves serving cost.

### What this overturns

* **Popularity-sampled negatives were a mistake (-11.8%).** The intent was to fix
  a train/eval mismatch, but evaluation uses 99 *uniform* negatives, so training
  on uniform negatives matches the eval distribution. One mismatch was traded for
  a worse one.
* **Mean pooling was the wrong default (-7.4%).** The audit originally recommended
  last-token pooling, then a toy experiment on a *randomly initialised* 2-layer
  backbone reversed the call. With a real pretrained Qwen the final position does
  carry a good summary, exactly as the causal-mask argument predicted. The toy was
  not a valid proxy for this decision.
* **Reward weighting costs 13.8% offline MRR.** This one is *not* a bug. It is a
  faithful implementation of a deliberate blog choice: inverse-propensity
  weighting down-weights popular positives, but the test set is drawn from that
  same popularity-biased distribution, so offline MRR structurally cannot see the
  benefit. Keep it, and know the price.

### The implied better configuration

Item text on, uniform negatives, last-token pooling, budget 3, and reward
weighting off if optimising for offline MRR. Not yet run, but it should beat
0.2752 while being roughly half as expensive to serve.
