"""Prefill-only GenRec serving on Modal.

    modal deploy modal_serve.py      # -> a live HTTPS endpoint

Serving GenRec is NOT LLM text serving. There is no decode loop, so none of the
usual generation machinery applies: one forward pass produces a pooled hidden
state, and one matmul against the item table produces the scores. The endpoints
below expose that, plus the two things that only become visible at serving time:

  * /bench   the blog's cost claim, measured. "Serving cost is approximately
             proportional to context length" and "context to roughly one-third
             with negligible degradation" are testable on our own model.
  * /catalog/add   continuous catalogue growth with NO retraining, which is the
             v2 cold-start path. The 5-core benchmark structurally cannot test
             this, so serving is the only place it can be demonstrated.
"""
import modal

app = modal.App("genrec-serve")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.2", "transformers>=4.44", "numpy",
                 "fastapi[standard]", "prometheus-client")
    .add_local_dir("src", remote_path="/root/src")
)

out_vol = modal.Volume.from_name("genrec-out", create_if_missing=True)
MODEL_DIR = "/out/v2/model"


@app.cls(image=image, gpu="A10G", volumes={"/out": out_vol},
         scaledown_window=300, timeout=600)
class Server:
    @modal.enter()
    def load(self):
        import sys, time
        sys.path.insert(0, "/root/src")
        from serve import GenRecService
        t0 = time.time()
        self.svc = GenRecService(MODEL_DIR, device="cuda", backend="torch",
                                 domain="product")
        self.load_s = time.time() - t0
        print(f"[serve] model ready in {self.load_s:.1f}s")

    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
        from prometheus_client import Counter, Gauge, Histogram, generate_latest
        from collections import Counter as PyCounter
        from fastapi.responses import PlainTextResponse
        import torch, time

        api = FastAPI(title="GenRec prefill-only serving")
        svc = self.svc

        # --- LABEL CARDINALITY ---------------------------------------------
        # Every distinct label VALUE creates its own time series. Labelling by
        # item_id would mint 12,101 series from one metric, which is how you
        # melt a Prometheus. Rule of thumb: labels must be low-cardinality and
        # bounded (endpoint, stage, status). Never user_id, item_id, or a
        # timestamp. Per-item behaviour is tracked in-process below and exposed
        # as a handful of AGGREGATE gauges instead.
        REQS = Counter("genrec_requests_total", "requests", ["endpoint"])
        LAT = Histogram("genrec_latency_ms", "stage latency (ms)", ["stage"],
                        buckets=(1, 2, 5, 10, 20, 50, 100, 200, 500, 1000))
        TOKENS = Histogram("genrec_prompt_tokens", "prompt tokens per request",
                           buckets=(32, 64, 96, 128, 192, 256, 384, 512, 768))
        BATCH = Histogram("genrec_batch_size", "requests per forward pass",
                          buckets=(1, 2, 4, 8, 16, 32, 64))
        CATALOG = Gauge("genrec_catalog_items", "items currently servable")
        COVERAGE = Gauge("genrec_catalog_coverage_ratio",
                         "fraction of the catalog ever recommended")
        CONCENTRATION = Gauge("genrec_served_head_share",
                              "share of recommendations taken by the top 1% "
                              "most-served items (1.0 = total popularity collapse)")
        GINI = Gauge("genrec_served_gini",
                     "Gini of the served-item distribution (0 = uniform)")

        # in-process, NOT a Prometheus label: bounded memory, no series blowup
        served_counts = PyCounter()

        class Interaction(BaseModel):
            item_id: int
            rating: float = 5.0
            ts: int = 0

        class RecReq(BaseModel):
            history: list[Interaction]
            k: int = 10
            exclude_seen: bool = True

        class NewItem(BaseModel):
            name: str
            categories: list[str] = []
            rescale: bool = True

        @api.get("/health")
        def health():
            return {"ok": True, "items": svc.num_items,
                    "pooling": svc.model.pooling, "budget": svc.budget,
                    "backend": svc.encoder.name, "load_seconds": self.load_s}

        @api.post("/recommend")
        def recommend(req: RecReq):
            REQS.labels("recommend").inc()
            hist = [h.model_dump() for h in req.history]
            recs, t = svc.recommend_batch([hist], k=req.k,
                                          exclude_seen=req.exclude_seen)
            for stage in ("verbalize_ms", "prefill_ms", "rank_ms", "total_ms"):
                LAT.labels(stage.replace("_ms", "")).observe(t[stage])
            TOKENS.observe(t["prompt_tokens_mean"])
            BATCH.observe(t["batch"])
            for r in recs[0]:
                served_counts[r.item_id] += 1
            return {"recommendations": [r.__dict__ for r in recs[0]], "timing": t}

        @api.post("/recommend/batch")
        def recommend_batch(reqs: list[RecReq]):
            REQS.labels("recommend_batch").inc()
            hists = [[h.model_dump() for h in r.history] for r in reqs]
            recs, t = svc.recommend_batch(hists, k=reqs[0].k if reqs else 10)
            return {"recommendations": [[r.__dict__ for r in row] for row in recs],
                    "timing": t}

        @api.post("/catalog/add")
        def add(item: NewItem):
            """Add a product live. No retraining, no restart."""
            REQS.labels("catalog_add").inc()
            try:
                iid = svc.add_item(item.name, item.categories, rescale=item.rescale)
            except RuntimeError as e:
                raise HTTPException(400, str(e))
            return {"item_id": iid, "catalog_size": svc.num_items,
                    "norm": float(svc.item_matrix[iid].norm()),
                    "warm_mean_norm": svc.warm_norm()}

        @api.get("/bench")
        def bench(n: int = 64, batch: int = 8, hist_len: int = 20):
            """Latency vs context budget: the blog's serving-cost claim, measured.

            Same synthetic histories at every budget, so the ONLY variable is how
            many tokens the prompt carries.
            """
            REQS.labels("bench").inc()
            import random
            rng = random.Random(0)
            hists = [[{"item_id": rng.randrange(svc.model.num_items),
                       "rating": rng.choice([3.0, 4.0, 5.0]), "ts": 0}
                      for _ in range(hist_len)] for _ in range(n)]
            rows = []
            base_budget = svc.budget
            try:
                for budget in (1, 2, 3, 5, 10, 20):
                    svc.budget = budget
                    svc.recommend_batch(hists[:batch], k=10)      # warmup
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    acc = {"verbalize_ms": 0.0, "prefill_ms": 0.0,
                           "rank_ms": 0.0, "total_ms": 0.0}
                    toks = 0.0
                    t0 = time.perf_counter()
                    for i in range(0, n, batch):
                        _, t = svc.recommend_batch(hists[i:i + batch], k=10)
                        for kk in acc:
                            acc[kk] += t[kk]
                        toks += t["prompt_tokens_mean"]
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    wall = time.perf_counter() - t0
                    nb = max(1, (n + batch - 1) // batch)
                    rows.append({"budget": budget,
                                 "prompt_tokens": round(toks / nb, 1),
                                 "verbalize_ms_per_batch": round(acc["verbalize_ms"] / nb, 2),
                                 "prefill_ms_per_batch": round(acc["prefill_ms"] / nb, 2),
                                 "rank_ms_per_batch": round(acc["rank_ms"] / nb, 2),
                                 "total_ms_per_batch": round(acc["total_ms"] / nb, 2),
                                 "wall_ms_per_batch": round(wall * 1e3 / nb, 2),
                                 "req_per_s": round(n / wall, 1)})
            finally:
                svc.budget = base_budget
            base = rows[-2] if len(rows) >= 2 else rows[0]     # budget=10 row
            for r in rows:
                r["tokens_vs_b10"] = round(r["prompt_tokens"] / base["prompt_tokens"], 3)
                r["prefill_vs_b10"] = round(r["prefill_ms_per_batch"]
                                            / base["prefill_ms_per_batch"], 3)
            return {"batch": batch, "n": n, "rows": rows,
                    "note": "compare tokens_vs_b10 against prefill_vs_b10: if they "
                            "track, serving cost is proportional to context length"}

        @api.get("/metrics")
        def metrics():
            """Prometheus scrapes this. Aggregates are computed at scrape time
            so the hot request path stays free of bookkeeping."""
            CATALOG.set(svc.num_items)
            total = sum(served_counts.values())
            if total:
                counts = sorted(served_counts.values(), reverse=True)
                COVERAGE.set(len(counts) / max(svc.num_items, 1))
                head_n = max(1, svc.num_items // 100)
                CONCENTRATION.set(sum(counts[:head_n]) / total)
                # Gini over the served distribution, padded with the items that
                # were never served (they are genuine zeros, not missing data).
                vals = sorted(counts + [0] * max(0, svc.num_items - len(counts)))
                n = len(vals)
                cum = sum((i + 1) * v for i, v in enumerate(vals))
                GINI.set((2 * cum) / (n * sum(vals)) - (n + 1) / n if sum(vals) else 0.0)
            return PlainTextResponse(generate_latest())

        return api
