"""The three GenRec losses.

Blog mapping (Phase-2 combines three objectives):
  1. ranking loss             — score real engagements above negatives.
  2. language-modeling loss    — preserve language understanding (anti-forget).
  3. reward-weighted alignment — weight toward long-term satisfaction, not just
                                 "engaged once".

Combined: L = a*rank_term + b*lm, where rank_term is the reward-SCALED ranking
loss when the reward objective is on and the plain one when it is off. The blog
scales the ranking loss by the reward rather than adding a second copy of it, so
there are three objectives but two additive terms. Both are separately ablatable
(blog reports Phase-2 losses contributing 35-50% of ranking gains).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def ranking_loss(scores: torch.Tensor) -> torch.Tensor:
    """Contrastive / softmax ranking. `scores` is [B, 1+N] where column 0 is the
    positive and 1..N are sampled negatives. Cross-entropy toward index 0 pushes
    the positive above the negatives."""
    target = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
    return F.cross_entropy(scores, target)


def reward_weighted_loss(scores: torch.Tensor, reward: torch.Tensor) -> torch.Tensor:
    """THE ranking loss, with each example's contribution scaled by its reward.

    Blog: "The example's ranking loss is scaled by this weight: high-value
    engagements receive larger weights." Note this REPLACES the plain ranking
    loss rather than being added alongside it — adding an unweighted copy would
    compress the effective weight ratio toward 1 and neuter the signal.

    `reward` already comes normalized around 1.0 from rewards.build_reward_fn
    (satisfaction x behaviour-rebalancing), so no batch-dependent renormalization
    here: a batch that happens to be all-tail should keep its larger weights.
    """
    target = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
    per_ex = F.cross_entropy(scores, target, reduction="none")  # [B]
    return (per_ex * reward).mean()


def lm_loss(logits: torch.Tensor, input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            loss_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Next-token cross-entropy (shifted), masking pads.

    In Phase 2 the sequence is [verbalized input + assistant turn] and the blog
    puts the LM objective "over the verbalized inputs and outputs", so the
    default (loss_mask=None) scores every real token. Pass `loss_mask` to score
    only a span, e.g. the assistant turn alone.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous().clone()
    shift_mask = attention_mask[:, 1:].contiguous()
    if loss_mask is not None:
        shift_mask = shift_mask * loss_mask[:, 1:].contiguous()
    shift_labels[shift_mask == 0] = -100
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def combined_loss(rank_term, lm, a=1.0, b=0.1):
    """Phase-2 total.

    `rank_term` is the catalog-aware ranking loss, already reward-scaled when the
    reward objective is on (that is the blog's formulation: one ranking loss,
    weighted per example). `b` is the LM leash.
    """
    return a * rank_term + b * lm
