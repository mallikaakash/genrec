"""GenRec model: LLM backbone + catalog-aware ranking head.

Blog mapping — the two heads:
  * LM head  (from the base causal-LM): outputs token logits. Kept ONLY to
    compute the language-modeling loss (anti-forgetting). Never used to produce
    recommendations.
  * Ranking head: a learned item-embedding table [num_items x d]. The pooled
    hidden state (the user vector) is scored against item vectors via dot product
    (or a small MLP). THIS is what ranks the catalog.

Serving is prefill-only: one forward pass -> pooled hidden state -> ranking head.
No autoregressive decode, because we score, we don't generate.

`build_backbone` supports:
  * a HF model name (e.g. "Qwen/Qwen2.5-0.5B") for Kaggle/real runs, and
  * tiny=True: a small randomly-initialized GPT-2 so the whole pipeline can be
    smoke-tested locally with no download.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_backbone(model_name: str = "Qwen/Qwen2.5-0.5B", tiny: bool = False):
    """Returns (causal_lm, tokenizer, hidden_size)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if tiny:
        from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast
        cfg = GPT2Config(n_layer=2, n_head=2, n_embd=64, n_positions=256,
                         vocab_size=50257)
        lm = GPT2LMHeadModel(cfg)
        tok = GPT2TokenizerFast.from_pretrained("gpt2")
        tok.pad_token = tok.eos_token
        tok.padding_side = "right"     # last_token_pool assumes left-aligned seqs
        return lm, tok, cfg.n_embd

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"         # last_token_pool assumes left-aligned seqs
    lm = AutoModelForCausalLM.from_pretrained(model_name)
    hidden = lm.config.hidden_size
    return lm, tok, hidden


def masked_mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token hidden states over the masked span -> user vector.
    hidden: [B, T, D], mask: [B, T] -> [B, D]."""
    mask = mask.unsqueeze(-1).float()
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


def last_token_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Take the hidden state at the LAST masked position.

    This is the right pooling for a causal decoder: only the final position has
    attended to the whole prompt, so it is the only one that can 'summarize the
    user's current preferences and context'. Mean pooling averages in early
    positions that have not yet seen most of the history.
    """
    idx = mask.long().sum(dim=1) - 1              # [B], last non-masked position
    idx = idx.clamp(min=0)
    b = torch.arange(hidden.size(0), device=hidden.device)
    return hidden[b, idx]                          # [B, D]


class GenRec(nn.Module):
    """LLM backbone + catalog-aware ranking head.

    Item representation (blog: "each catalog item i has a learned embedding e_i"):
        e_i = id_embedding[i] + text_proj(text_embedding[i])
    The ID term is the free-to-learn collaborative signal. The text term is the
    backbone's own encoding of the item's verbalized metadata, held fixed, which
    is what makes the model usable on items it never saw a single interaction for
    (cold start). With `use_item_text=False` this degrades to a pure ID table.
    """

    def __init__(self, backbone, hidden_size: int, num_items: int,
                 scorer: str = "dot", pooling: str = "mean",
                 use_item_text: bool = True):
        # pooling: the blog only says "a pooled hidden state h", so both are in
        # spec. `last` is the textbook choice for a causal decoder (only the final
        # position has attended to the whole prompt), but it leans entirely on the
        # backbone having a good summary representation — and every prompt here
        # ends with the SAME call-to-action tokens, so a weak backbone produces a
        # near-constant user vector. Measured on the toy: mean 0.225 MRR vs last
        # 0.089. Default to `mean` and let the real-data ablation settle it.
        super().__init__()
        self.backbone = backbone  # a *CausalLM* (exposes .lm_head via forward)
        self.hidden_size = hidden_size
        self.num_items = num_items
        self.pooling = pooling
        # --- the ranking head: a dedicated item-embedding table (NOT the vocab) ---
        self.item_emb = nn.Embedding(num_items, hidden_size)
        nn.init.normal_(self.item_emb.weight, std=0.02)
        self.scorer_kind = scorer
        if scorer == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size), nn.ReLU(),
                nn.Linear(hidden_size, 1),
            )
        # --- cold-start path: frozen text embeddings + a learned projection ---
        self.use_item_text = use_item_text
        if use_item_text:
            self.register_buffer("item_text",
                                 torch.zeros(num_items, hidden_size),
                                 persistent=True)
            self.text_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            nn.init.normal_(self.text_proj.weight, std=0.02)
            self.has_item_text = False

    # ---- pooling policy ----
    def _pool(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            return masked_mean_pool(hidden, mask)
        return last_token_pool(hidden, mask)

    # ---- item vectors: ID embedding (+ frozen text embedding) ----
    def item_vectors(self, item_ids: torch.Tensor) -> torch.Tensor:
        v = self.item_emb(item_ids)
        if self.use_item_text and self.has_item_text:
            v = v + self.text_proj(self.item_text[item_ids])
        return v

    # ---- hidden states WITHOUT the LM head ----
    def _encode(self, input_ids, attention_mask):
        """Run the decoder stack only, skipping lm_head.

        Ranking never needs token logits, and that projection is enormous: at
        vocab 151936 a [256, 384] batch of logits is ~60 GB in fp32. Every
        scoring path (validation, eval, serving, cold-start encoding) goes
        through here, so it never materializes them. This is also what
        "prefill-only" means literally — you pool a hidden state, you do not
        predict a token.
        """
        dec = None
        for getter in ("get_decoder",):
            fn = getattr(self.backbone, getter, None)
            if callable(fn):
                try:
                    dec = fn()
                except Exception:
                    dec = None
        if dec is None:
            dec = getattr(self.backbone, "model", None) or \
                  getattr(self.backbone, "transformer", None)
        if dec is None:                     # last resort: full forward
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                                output_hidden_states=True)
            return out.hidden_states[-1]
        out = dec(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state

    # ---- ONE backbone pass -> (pooled user vector, token logits) ----
    def forward_shared(self, input_ids, attention_mask, pool_mask=None):
        """The single forward every training step needs.

        `pool_mask` selects which positions the user vector is pooled from. In
        Phase 2 the sequence is [prompt + assistant turn] and pool_mask covers
        only the prompt, so the user vector is computed as if the answer were not
        there. Causal attention guarantees this is EXACTLY the prompt-only
        forward: no prompt position can attend to a later position.

        Returns (user_vec [B, D], logits [B, T, V]).
        """
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                            output_hidden_states=True)
        hidden = out.hidden_states[-1]                       # [B, T, D]
        m = attention_mask if pool_mask is None else pool_mask
        return self._pool(hidden, m), out.logits

    # ---- shared: prompt -> user vector (the prefill pass, no LM head) ----
    def user_vector(self, input_ids, attention_mask, pool_mask=None,
                    chunk: int = 32) -> torch.Tensor:
        """Pooled user vector. Chunked over the batch so a large validation or
        eval batch cannot blow up activation memory in one shot."""
        m = attention_mask if pool_mask is None else pool_mask
        if input_ids.size(0) <= chunk:
            return self._pool(self._encode(input_ids, attention_mask), m)
        outs = []
        for i in range(0, input_ids.size(0), chunk):
            h = self._encode(input_ids[i:i + chunk], attention_mask[i:i + chunk])
            outs.append(self._pool(h, m[i:i + chunk]))
        return torch.cat(outs, dim=0)

    # ---- ranking head: user vector x candidate item vectors -> scores ----
    def score_items(self, user_vec: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """user_vec: [B, D]; item_ids: [B, C] -> scores [B, C]."""
        item_vecs = self.item_vectors(item_ids)  # [B, C, D]
        if self.scorer_kind == "dot":
            return torch.einsum("bd,bcd->bc", user_vec, item_vecs)
        # mlp
        B, C, D = item_vecs.shape
        u = user_vec.unsqueeze(1).expand(-1, C, -1)
        return self.mlp(torch.cat([u, item_vecs], dim=-1)).squeeze(-1)

    # ---- LM head passthrough (for the language-modeling loss) ----
    def lm_logits(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return out.logits  # [B, T, vocab]

    # ---- cold start: ground item vectors in the backbone's text understanding ----
    @torch.no_grad()
    def build_item_text_embeddings(self, tokenizer, catalog, verbalize_item_fn,
                                   device, batch_size: int = 64,
                                   max_len: int = 64) -> None:
        """Encode every catalog item's verbalized metadata with the (adapted)
        backbone and store the pooled vector. Run AFTER Phase 1 so the encoding
        reflects the domain-adapted model."""
        if not self.use_item_text:
            return
        was_training = self.training
        self.eval()
        ids = sorted(catalog)
        texts = [verbalize_item_fn(catalog[i]) for i in ids]
        vecs = []
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(texts[i:i + batch_size], return_tensors="pt",
                            padding=True, truncation=True,
                            max_length=max_len).to(device)
            h = self._encode(enc["input_ids"], enc["attention_mask"])
            vecs.append(self._pool(h, enc["attention_mask"]).float().cpu())
        emb = torch.cat(vecs, dim=0)
        # normalize to unit scale so the text term does not swamp the ID term
        emb = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        table = torch.zeros(self.num_items, self.hidden_size)
        table[torch.tensor(ids)] = emb
        self.item_text.copy_(table.to(self.item_text.dtype).to(self.item_text.device))
        self.has_item_text = True
        if was_training:
            self.train()
        print(f"[cold-start] built text embeddings for {len(ids)} items")

    @torch.no_grad()
    def rank(self, input_ids, attention_mask, candidate_ids) -> torch.Tensor:
        """Serving path: prefill -> pooled vector -> score candidates. No decode."""
        uv = self.user_vector(input_ids, attention_mask)
        return self.score_items(uv, candidate_ids)

    # ---- parameter groups: a pretrained backbone and a cold table want very
    # ---- different learning rates.
    def param_groups(self, backbone_lr: float, head_lr: float,
                     backbone_wd: float = 0.01):
        """Two groups. The head also gets weight_decay=0: decaying a sparse item
        table pulls rarely-seen (tail) items toward zero, which is exactly the
        popularity bias the reward's rebalancing term is trying to undo."""
        head = [self.item_emb.weight]
        if self.scorer_kind == "mlp":
            head += list(self.mlp.parameters())
        if self.use_item_text:
            head += list(self.text_proj.parameters())
        return [{"params": list(self.backbone.parameters()), "lr": backbone_lr,
                 "weight_decay": backbone_wd},
                {"params": head, "lr": head_lr, "weight_decay": 0.0}]


def save_genrec(model: "GenRec", tokenizer, path: str, meta: dict | None = None):
    """Persist a trained GenRec: adapted backbone + tokenizer (HF format) and the
    ranking head (item-embedding table + optional MLP) + metadata. Downloadable
    and reloadable with load_genrec()."""
    import json
    import os
    os.makedirs(path, exist_ok=True)
    model.backbone.save_pretrained(os.path.join(path, "backbone"))
    tokenizer.save_pretrained(os.path.join(path, "backbone"))
    head = {"item_emb": model.item_emb.state_dict(),
            "scorer_kind": model.scorer_kind,
            "num_items": model.num_items, "hidden_size": model.hidden_size,
            "pooling": model.pooling, "use_item_text": model.use_item_text}
    if model.scorer_kind == "mlp":
        head["mlp"] = model.mlp.state_dict()
    if model.use_item_text:
        head["text_proj"] = model.text_proj.state_dict()
        head["item_text"] = model.item_text.detach().cpu()
        head["has_item_text"] = model.has_item_text
    torch.save(head, os.path.join(path, "ranking_head.pt"))
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump(meta or {}, f, indent=2)
    print(f"[save] wrote GenRec to {path}")


def load_genrec(path: str, device: str = "cpu"):
    """Reload a saved GenRec. Returns (model, tokenizer, meta)."""
    import json
    import os
    from transformers import AutoModelForCausalLM, AutoTokenizer
    backbone = AutoModelForCausalLM.from_pretrained(os.path.join(path, "backbone"))
    tok = AutoTokenizer.from_pretrained(os.path.join(path, "backbone"))
    head = torch.load(os.path.join(path, "ranking_head.pt"), map_location=device)
    model = GenRec(backbone, head["hidden_size"], head["num_items"],
                   scorer=head["scorer_kind"],
                   pooling=head.get("pooling", "last"),
                   use_item_text=head.get("use_item_text", False))
    model.item_emb.load_state_dict(head["item_emb"])
    if head["scorer_kind"] == "mlp":
        model.mlp.load_state_dict(head["mlp"])
    if model.use_item_text and "text_proj" in head:
        model.text_proj.load_state_dict(head["text_proj"])
        model.item_text.copy_(head["item_text"].to(model.item_text.device))
        model.has_item_text = head.get("has_item_text", True)
    meta = {}
    mp = os.path.join(path, "meta.json")
    if os.path.exists(mp):
        meta = json.load(open(mp))
    return model.to(device), tok, meta


if __name__ == "__main__":
    torch.manual_seed(0)
    lm, tok, hid = build_backbone(tiny=True)
    model = GenRec(lm, hid, num_items=120, scorer="dot")
    enc = tok(["Recently ordered from: Italian Spot (5 stars). Recommend next.",
               "A new diner. Recommend a restaurant."],
              return_tensors="pt", padding=True, truncation=True, max_length=64)
    uv = model.user_vector(enc["input_ids"], enc["attention_mask"])
    cand = torch.randint(0, 120, (2, 5))
    scores = model.score_items(uv, cand)
    print("user_vec:", tuple(uv.shape), "scores:", tuple(scores.shape))
    print("lm_logits:", tuple(model.lm_logits(enc["input_ids"], enc["attention_mask"]).shape))
    print("OK")
