# Exercise: serve GenRec's backbone with vLLM

This is yours to implement. `src/serve.py: VLLMEncoder` is a stub that raises
`NotImplementedError`, and `tests/test_vllm_conformance.py` is the red test that
tells you when you are done.

## Why this is a real exercise and not busywork

GenRec is prefill-only. That means most of what vLLM is famous for does not
apply, and the interesting part is working out which 20% does.

| vLLM feature | solves | relevant here? |
|---|---|---|
| PagedAttention | KV cache fragmentation across many *generating* sequences | **No.** Our KV cache is built once and discarded. |
| Continuous batching | sequences finish generating at different times | **Barely.** Every request is one forward pass. |
| Speculative decoding | making token-by-token generation faster | **No.** No tokens generated. |
| CUDA graphs | per-kernel launch overhead | **Yes, this is the prize.** |
| Prefix caching | reusing KV for shared prompt prefixes | Only if you move the task instruction to the front. |
| Pooling / embed task | returning hidden states instead of logits | **Yes, this is the API you need.** |

## The measured prediction you are testing

From the current PyTorch service (`SERVING.md`), prefill fits:

```
prefill_ms = 12.3  +  0.136 x (tokens per request)      R² = 0.969
```

That 12.3 ms intercept is not compute. It is ~400 kernel launches per forward
at roughly 30 microseconds of CPU dispatch each. At budget 3, our best operating
point, it is **51% of prefill**.

CUDA graphs collapse those 400 launches into one replay. So:

> **Hypothesis:** vLLM should largely eliminate the 12.3 ms intercept and help
> progressively less as context grows and real compute takes over. It should do
> nothing for verbalization or ranking, which are already ~0.3 ms each.

Your job is to test that. Write down what you expect before you measure.

## Steps

**1. Get the checkpoint locally.**

```bash
python scripts/fetch_model.py
```

**2. Read the contract.** `src/serve.py: VLLMEncoder` docstring lists the four
things that will bite you, in the order they will bite you.

**3. Bring up the stack.**

```bash
docker compose up -d
docker compose logs -f vllm      # watch for the pooling model to load
```

Needs a Linux host with an NVIDIA GPU. It will not run on Apple Silicon.

**4. Implement `encode()`.** vLLM serves an OpenAI-compatible
`POST /v1/embeddings`. You send `{"model": ..., "input": [prompts]}` and get
back one embedding per prompt. Use `httpx`. Return
`(torch.Tensor [B, D] float32, token_counts)`.

**5. Make the test green.**

```bash
GENREC_MODEL_DIR=./genrec_model_v2 pytest tests/ -v
```

The test asserts your encoder produces the **same top-10 item ids in the same
order** as the PyTorch encoder. It deliberately does not compare raw vectors,
because vLLM normalizes and runs bf16, and neither of those can reorder a
dot-product ranking. Understanding why that is true is half the exercise.

**6. Measure, then compare against your written-down prediction.**

```bash
curl "localhost:8080/bench?n=64&batch=8&hist_len=20"
```

Compare the fitted intercept against 12.3 ms.

## Stretch goals, in increasing difficulty

1. **Server-side batching.** `/recommend` currently runs batch=1 per HTTP
   request; the throughput numbers in SERVING.md come from explicit batches.
   Add a queue that holds arriving requests ~5 ms, takes up to N, runs one
   forward, and scatters results. Then measure real concurrent throughput.
2. **Prefix caching.** Move the task instruction to the *front* of the prompt so
   requests share a cacheable prefix, enable `--enable-prefix-caching`, measure.
   With mean pooling the position does not affect quality, so it is free.
   This is a good example of serving constraints feeding back into model design.
3. **Skip vLLM entirely and just use `torch.compile(model, mode="reduce-overhead")`**
   with padding to fixed length buckets. This also enables CUDA graphs. If it
   recovers most of the 12.3 ms, you have learned something important about when
   a heavyweight serving engine is and is not worth adopting.

Honestly, do (3) first. It is one line, it isolates the variable, and it tells
you how much of vLLM's benefit you can get without adopting vLLM.
