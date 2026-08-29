# GenRec prefill-only serving

Live endpoint: `https://aakashmallik7777--genrec-serve-server-web.modal.run`

```bash
modal deploy modal_serve.py
```

## The idea in one box

At serving time GenRec is **not a language model**. There is no decoding, no
sampling, no KV cache reuse. The LM head is never touched.

```
prompt text -> transformer stack -> pooled h [896]     the expensive part
h @ item_table.T                 -> scores [12,101]    ~11 MFLOPs, free
```

`src/serve.py` enforces this: the scoring path goes through `model._encode()`,
which runs the decoder stack only.

## Measured on the trained Amazon Beauty model (A10G, batch 8)

| budget | prompt tokens | verbalize | prefill | rank | wall | req/s |
|---|---|---|---|---|---|---|
| 1 | 70 | 0.29 ms | 26.7 ms | 0.36 ms | 27.6 ms | 290 |
| 3 | 115 | 0.27 | 24.4 | 0.33 | 25.2 | 318 |
| 5 | 158 | 0.27 | 32.1 | 0.34 | 32.9 | 243 |
| 10 | 272 | 0.30 | 51.2 | 0.36 | 52.1 | 154 |
| 20 | 384 | 0.24 | 64.5 | 0.41 | 65.4 | 122 |

Single warm request at budget 10: **23 ms**, of which prefill is 22.0 ms and the
ranking head is 0.28 ms. **The recommender is 1.2% of the serving cost.** The
encoder is the entire bill.

## Testing the blog's serving-cost claim

The blog states serving cost is "approximately proportional to context length"
and that cutting context to "roughly one-third" cost negligible MRR.

| budget | tokens (rel) | wall (rel) |
|---|---|---|
| 3 | 0.42 | **0.48** |
| 5 | 0.58 | 0.63 |
| 10 | 1.00 | 1.00 |
| 20 | 1.41 | 1.26 |

**Confirmed, with a caveat.** Cost tracks context, but sublinearly: there is a
fixed floor of roughly 24 ms that dominates below ~150 tokens, so proportionality
only holds in the upper range.

Combined with the ablation result (context 1/3 scored MRR 0.2212 vs 0.2140 at
full context, i.e. **+3.4%**):

> Cutting context to one third **halves serving cost and slightly improves
> quality.** The blog said "negligible degradation"; we measured a small gain.

That makes budget 3 look strictly better than the budget 10 the model was trained
at, which is the most actionable finding in this repo.

## Endpoints

| route | purpose |
|---|---|
| `GET /health` | model, catalog size, pooling, backend |
| `POST /recommend` | one user, returns top-K with per-stage timing |
| `POST /recommend/batch` | batched prefill |
| `POST /catalog/add` | **add a product live, no retraining** |
| `GET /bench` | the context/latency sweep above |
| `GET /metrics` | Prometheus |

```bash
curl -s -X POST $URL/recommend -H 'content-type: application/json' \
  -d '{"history":[{"item_id":11,"rating":5},{"item_id":877,"rating":5}],"k":5}'
```

## Continuous catalogue growth

`POST /catalog/add` is the v2 cold-start path finally doing something visible.
A new item gets `e_i = text_proj(encode(its metadata))` immediately, with no
retraining. Under v1's pure-ID table a new item was a random vector forever.

One real bug found here: a warm item's vector is `id_embedding + text_proj(text)`,
but a cold item only has the text half, so its norm was **70.6%** of a warm
item's. Since dot-product scores scale with `||e_i||`, cold items were penalised
regardless of content match. We have evidence for the item's *direction*
(content) but none for its *norm* (which acts as a popularity prior), so
`add_item(rescale=True)` sets the norm to the catalog mean.

## Two profiling traps this exercise hit

Both produced confidently wrong conclusions before being caught.

1. **Async CUDA makes naive timers lie.** Kernels are launched asynchronously, so
   `perf_counter` between stages measures *launch* time. The real GPU cost then
   lands in whichever later stage happens to synchronise first, which made the
   ranking step look like it cost 41 ms. Stage timings need explicit
   `torch.cuda.synchronize()` barriers (`GenRecService.profile`).
2. **Modal reuses warm containers.** After `modal deploy`, an in-flight container
   keeps serving the previous code. Measurements silently mixed new benchmark
   code with an old model path. Force it with `modal app stop <app> --yes`.

Also worth recording: an early version tokenized the prompt twice, once purely to
log token counts. Instrumentation that costs real time shows up in the profile as
if it were inherent cost.

## Next: the vLLM comparison

The number that motivates it: at budget 1, prefill is **26.7 ms for 560 tokens
total**. That is nowhere near compute-bound on an A10G. A 0.5B model has ~24
layers of small kernels, so we are dominated by **per-kernel launch overhead**,
which is exactly what CUDA graphs eliminate and what vLLM enables by default.

Prediction to test: vLLM should crush the ~24 ms floor at short contexts and help
progressively less as context grows and real compute starts to dominate. It will
do nothing for verbalization or ranking, which are already 0.3 ms each.

`src/serve.py` has a `VLLMEncoder` that serves the backbone through vLLM's
**pooling/embedding** path (not generate, since we want a hidden state, not
tokens), keeping the ranking head in-process. It is written but **untested**:
vLLM's pooling API has been renamed across releases, so it tries
`runner="pooling"` then `task="embed"` then bare construction.
