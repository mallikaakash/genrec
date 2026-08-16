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
        return lm, tok, cfg.n_embd

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained(model_name)
    hidden = lm.config.hidden_size
    return lm, tok, hidden


def masked_mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token hidden states over the (non-pad) prompt -> user vector.
    hidden: [B, T, D], mask: [B, T] -> [B, D]."""
    mask = mask.unsqueeze(-1).float()
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


class GenRec(nn.Module):
    def __init__(self, backbone, hidden_size: int, num_items: int,
                 scorer: str = "dot"):
        super().__init__()
        self.backbone = backbone  # a *CausalLM* (exposes .lm_head via forward)
        self.hidden_size = hidden_size
        self.num_items = num_items
        # --- the ranking head: a dedicated item-embedding table (NOT the vocab) ---
        self.item_emb = nn.Embedding(num_items, hidden_size)
        nn.init.normal_(self.item_emb.weight, std=0.02)
        self.scorer_kind = scorer
        if scorer == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size), nn.ReLU(),
                nn.Linear(hidden_size, 1),
            )

    # ---- shared: prompt -> user vector (the prefill pass) ----
    def user_vector(self, input_ids, attention_mask) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                            output_hidden_states=True)
        last_hidden = out.hidden_states[-1]           # [B, T, D]
        return masked_mean_pool(last_hidden, attention_mask)  # [B, D]

    # ---- ranking head: user vector x candidate item vectors -> scores ----
    def score_items(self, user_vec: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """user_vec: [B, D]; item_ids: [B, C] -> scores [B, C]."""
        item_vecs = self.item_emb(item_ids)  # [B, C, D]
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

    @torch.no_grad()
    def rank(self, input_ids, attention_mask, candidate_ids) -> torch.Tensor:
        """Serving path: prefill -> pooled vector -> score candidates. No decode."""
        uv = self.user_vector(input_ids, attention_mask)
        return self.score_items(uv, candidate_ids)


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
