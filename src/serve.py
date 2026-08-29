"""Prefill-only serving for GenRec.

The whole point: at serving time GenRec is NOT a language model. There is no
decoding, no sampling, no KV cache reuse. One forward pass produces a pooled
hidden state, and a single matmul against the item table produces the scores:

    prompt text -> transformer stack -> pooled h [D]      (the expensive part)
    h @ item_table.T                 -> scores  [num_items]  (~11 MFLOPs, free)

So the model being "served" is an ENCODER. That is why the vLLM backend below
uses vLLM's pooling/embedding path and not its generate path: we want a hidden
state back, not tokens. The ranking head stays here, in our process, because it
is a rounding error next to the transformer forward.

Backends
    torch : loads the fine-tuned GenRec directly. The honest baseline.
    vllm  : the backbone is served by vLLM as an embedding model; we keep the
            ranking head. Same maths, different encoder runtime.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import torch

from collections import Counter

from data import Interaction, Item
import verbalize
from verbalize import (DOMAINS, verbalize_context, verbalize_history,
                       verbalize_item, _clip, _descriptor)


# --------------------------------------------------------------------------- #
# Catalog reconstructed from the saved checkpoint's meta.json
# --------------------------------------------------------------------------- #
def load_catalog(meta: dict) -> dict[int, Item]:
    """meta['catalog'] is {item_id: {name, categories}} written by save_genrec."""
    cat = {}
    for k, v in (meta.get("catalog") or {}).items():
        iid = int(k)
        cat[iid] = Item(iid, v.get("name", f"Item {iid}"),
                        list(v.get("categories") or []), "", 0.0)
    return cat


@dataclass
class Rec:
    item_id: int
    name: str
    score: float


# --------------------------------------------------------------------------- #
# Encoder backends: prompt text -> pooled user vector
# --------------------------------------------------------------------------- #
class TorchEncoder:
    """The baseline. Runs the decoder stack directly, never the LM head."""

    name = "torch"

    def __init__(self, model, tokenizer, device, max_len=384, amp=True):
        self.model, self.tok, self.device = model, tokenizer, device
        self.max_len, self.amp = max_len, amp
        self.model.eval()

    @torch.no_grad()
    def encode(self, prompts: list[str]):
        """Returns (user_vectors [B,D], token_counts). Tokenizes ONCE: counting
        tokens with a second tokenizer pass would add real CPU time to every
        request and then show up in the profile as if it were inherent cost."""
        from train import amp_ctx
        enc = self.tok(prompts, return_tensors="pt", padding=True,
                       truncation=True, max_length=self.max_len).to(self.device)
        n_tok = enc["attention_mask"].sum(dim=1).tolist()
        with amp_ctx(self.device, self.amp):
            # _encode skips lm_head entirely: this IS the prefill-only path
            h = self.model._encode(enc["input_ids"], enc["attention_mask"])
            v = self.model._pool(h, enc["attention_mask"])
        return v.float(), n_tok


class VLLMEncoder:
    """vLLM serving the backbone as a POOLING (embedding) model.

    vLLM's marquee features (PagedAttention, continuous batching of decode,
    speculative decoding) target autoregressive generation and are largely moot
    here: our KV cache is built once and discarded. What we actually want from
    vLLM is batched prefill, CUDA graphs, and a production server loop.

    NOTE ON NORMALIZATION: vLLM's embedding path L2-normalizes by default. That
    is harmless for us: dividing every candidate score for ONE user by the same
    ||h|| is a monotone transform within that query, so the ranking is
    unchanged. It would NOT be harmless with scorer="mlp".

    vLLM's pooling API has been renamed across releases (older: task="embed";
    newer: runner="pooling" + convert="embed"). We try the variants in order.
    """

    name = "vllm"

    def __init__(self, model_path: str, max_len=384, pooling="MEAN",
                 gpu_memory_utilization=0.45, enforce_eager=False):
        from vllm import LLM
        from vllm.config import PoolerConfig

        common = dict(model=model_path, max_model_len=max_len,
                      gpu_memory_utilization=gpu_memory_utilization,
                      enforce_eager=enforce_eager,
                      override_pooler_config=PoolerConfig(pooling_type=pooling,
                                                          normalize=False))
        errors = []
        for kwargs in ({"runner": "pooling", "convert": "embed"},
                       {"task": "embed"},
                       {}):
            try:
                self.llm = LLM(**common, **kwargs)
                self.api = kwargs
                break
            except (TypeError, ValueError) as e:
                errors.append(f"{kwargs}: {type(e).__name__}: {e}")
        else:
            raise RuntimeError("could not construct a vLLM pooling engine. "
                               "Tried:\n  " + "\n  ".join(errors))
        print(f"[serve] vLLM pooling engine up via {self.api}")

    def encode(self, prompts: list[str]):
        outs = self.llm.embed(prompts) if hasattr(self.llm, "embed") \
            else self.llm.encode(prompts)
        vecs, n_tok = [], []
        for o in outs:
            e = o.outputs.embedding if hasattr(o.outputs, "embedding") else o.outputs.data
            vecs.append(torch.as_tensor(e, dtype=torch.float32))
            n_tok.append(len(getattr(o, "prompt_token_ids", []) or []))
        return torch.stack(vecs), n_tok


# --------------------------------------------------------------------------- #
# The service
# --------------------------------------------------------------------------- #
class GenRecService:
    def __init__(self, model_dir: str, device: str = "cuda",
                 backend: str = "torch", budget: int | None = None,
                 max_len: int = 384, domain: str = "product",
                 profile: bool = True, **backend_kw):
        from model import load_genrec

        self.model, self.tok, self.meta = load_genrec(model_dir, device=device)
        self.device = device
        self.profile = profile
        cfg = self.meta.get("config", {})
        self.budget = budget if budget is not None else cfg.get("budget", 10)
        self.max_len = max_len
        verbalize.set_domain(domain)

        self.catalog = load_catalog(self.meta)
        self.num_items = self.model.num_items
        self.model.to(device).eval()

        if backend == "vllm":
            self.encoder = VLLMEncoder(os.path.join(model_dir, "backbone"),
                                       max_len=max_len, **backend_kw)
        else:
            self.encoder = TorchEncoder(self.model, self.tok, device,
                                        max_len=max_len, **backend_kw)

        self._precompute_descriptions()

        # Precompute the full item matrix ONCE. This is the ranking head:
        # e_i = id_embedding[i] + text_proj(frozen text embedding[i]).
        self.item_matrix = self._build_item_matrix()
        print(f"[serve] backend={self.encoder.name} items={self.num_items} "
              f"budget={self.budget} pooling={self.model.pooling}")

    @torch.no_grad()
    def _build_item_matrix(self) -> torch.Tensor:
        ids = torch.arange(self.num_items, device=self.device)
        return self.model.item_vectors(ids).float()          # [num_items, D]

    # ---- prompt building ------------------------------------------------- #
    def _precompute_descriptions(self) -> None:
        """Per-item description strings, built ONCE at load.

        Measured on the deployed service: rebuilding these per request made CPU
        verbalization the single largest cost at long contexts (45 ms/batch vs a
        flat 26 ms for tokenize + GPU). The item half of the string never
        changes, so it has no business being recomputed on every request.
        """
        self._desc = {}
        for iid, it in self.catalog.items():
            d = _descriptor(it)
            self._desc[iid] = f"{_clip(it.name)}{', ' + d if d else ''}"
        self._cat_of = {iid: _descriptor(it) for iid, it in self.catalog.items()}

    def build_prompt(self, history: list[dict], now_ts: int | None = None) -> str:
        """Fast path producing output byte-identical to verbalize_history.
        `assert_matches_training()` proves that, so the optimisation cannot
        silently drift the serving prompt away from the training prompt."""
        inters = [Interaction(int(h["item_id"]), float(h.get("rating", 5.0)),
                              int(h.get("ts", 0))) for h in history]
        dom = DOMAINS[verbalize._DOMAIN]
        if not inters:
            return f"A new {dom['actor']} with no history. {dom['cta']}"

        b = self.budget
        recent, older = inters[-b:], inters[:-b]
        lines = []
        ctx = verbalize_context(inters, now_ts)
        if ctx:
            lines.append(ctx)
        if older:
            cats = Counter(self._cat_of.get(it.item_id, "") for it in older)
            cats.pop("", None)
            avg = sum(it.rating for it in older) / len(older)
            top = ", ".join(c for c, _ in cats.most_common(3))
            lines.append(f"Earlier, they had {len(older)} interactions "
                         f"(favouring {top}; avg {avg:.1f} stars).")
        d = self._desc
        recent_str = "; ".join(f"{d[it.item_id]} ({it.rating:.0f} stars)"
                               for it in recent)
        lines.append(f"Recently {dom['verb']}: {recent_str}.")
        lines.append(dom["cta"])
        return " ".join(lines)

    def assert_matches_training(self, histories: list[list[dict]],
                                now_ts: int | None = None) -> None:
        """Byte-compare the fast path against the training-time verbalizer."""
        for h in histories:
            inters = [Interaction(int(x["item_id"]), float(x.get("rating", 5.0)),
                                  int(x.get("ts", 0))) for x in h]
            ref = verbalize_history(inters, self.catalog, budget=self.budget,
                                    now_ts=now_ts)
            got = self.build_prompt(h, now_ts=now_ts)
            if ref != got:
                raise AssertionError(
                    "serving prompt diverged from training verbalizer\n"
                    f"  train: {ref!r}\n  serve: {got!r}")

    def _sync(self):
        """CUDA kernels are launched asynchronously, so a naive perf_counter
        between stages measures LAUNCH time, not execution time: the real GPU
        cost then silently lands in whichever later stage happens to sync first.
        Stage timings are only meaningful with an explicit barrier."""
        if self.profile and self.device.startswith("cuda"):
            torch.cuda.synchronize()

    @torch.no_grad()
    def recommend_batch(self, histories: list[list[dict]], k: int = 10,
                        exclude_seen: bool = True) -> tuple[list[list[Rec]], dict]:
        self._sync()
        t0 = time.perf_counter()
        prompts = [self.build_prompt(h) for h in histories]

        self._sync()
        t1 = time.perf_counter()
        uv, n_tok = self.encoder.encode(prompts)              # PREFILL
        uv = uv.to(self.device)
        self._sync()
        t2 = time.perf_counter()

        scores = uv @ self.item_matrix.T                      # [B, num_items]
        if exclude_seen:
            rows, cols = [], []
            for i, h in enumerate(histories):
                for x in h:
                    rows.append(i); cols.append(int(x["item_id"]))
            if rows:      # one H2D copy for the whole batch, not one per request
                scores[torch.tensor(rows, device=self.device),
                       torch.tensor(cols, device=self.device)] = -1e30
        top = torch.topk(scores, k=min(k, self.num_items), dim=-1)
        self._sync()
        t3 = time.perf_counter()

        out = []
        for row_i, row_s in zip(top.indices.tolist(), top.values.tolist()):
            out.append([Rec(i, self.catalog.get(i, Item(i, f"Item {i}", [], "", 0)).name,
                            float(s)) for i, s in zip(row_i, row_s)])
        timing = {"batch": len(histories),
                  "verbalize_ms": (t1 - t0) * 1e3,
                  "prefill_ms": (t2 - t1) * 1e3,
                  "rank_ms": (t3 - t2) * 1e3,
                  "total_ms": (t3 - t0) * 1e3,
                  "prompt_tokens_mean": (sum(n_tok) / len(n_tok)) if n_tok else 0,
                  "prompt_tokens_max": max(n_tok) if n_tok else 0}
        return out, timing

    # ---- continuous catalogue growth (the cold-start payoff) ------------- #
    @torch.no_grad()
    def warm_norm(self) -> float:
        """Mean L2 norm of the item vectors the model was actually trained on."""
        if not hasattr(self, "_warm_norm"):
            n = self.item_matrix[:self.model.num_items].norm(dim=-1)
            self._warm_norm = float(n.mean())
        return self._warm_norm

    @torch.no_grad()
    def add_item(self, name: str, categories: list[str],
                 item_id: int | None = None,
                 rescale: bool = True) -> int:
        """Add a product at runtime with NO retraining.

        This is the v2 cold-start path finally doing something visible. The new
        item gets e_i = (zero ID embedding) + text_proj(encode(its metadata)).
        Under v1's pure-ID table a new item was a random vector forever.

        `rescale` fixes a subtle but real ranking bug. A warm item's vector is
        id_embedding + text_proj(text); a cold item only has the text half, so
        its norm is systematically smaller (measured ~70% on our checkpoints).
        Since dot-product scores scale with ||e_i||, the cold item is penalised
        no matter how well its content matches. We have evidence for its
        DIRECTION (content) but none for its NORM, which here behaves like a
        popularity/confidence prior, so the neutral choice is the catalog mean
        rather than an accidental 70%. Set rescale=False to keep the raw vector.
        """
        if not self.model.use_item_text:
            raise RuntimeError("model was trained with use_item_text=False; "
                               "it has no cold-start path")
        iid = self.num_items if item_id is None else item_id
        item = Item(iid, name, categories, "", 0.0)

        enc = self.tok([verbalize_item(item)], return_tensors="pt",
                       padding=True, truncation=True, max_length=64).to(self.device)
        h = self.model._encode(enc["input_ids"], enc["attention_mask"])
        v = self.model._pool(h, enc["attention_mask"]).float()
        v = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-6)   # match training scale

        vec = self.model.text_proj(v.to(self.model.text_proj.weight.dtype)).float()
        if rescale:
            target = self.warm_norm()
            vec = vec * (target / vec.norm(dim=-1, keepdim=True).clamp(min=1e-6))
        if iid >= self.num_items:                 # grow the served matrix
            self.item_matrix = torch.cat([self.item_matrix, vec], dim=0)
            self.num_items += 1
        else:                                     # or refresh an existing row
            self.item_matrix[iid] = vec[0]
        self.catalog[iid] = item
        d = _descriptor(item)
        self._desc[iid] = f"{_clip(item.name)}{', ' + d if d else ''}"
        self._cat_of[iid] = d
        return iid
