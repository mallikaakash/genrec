"""The three GenRec losses.

Blog mapping (Phase-2 combines three objectives):
  1. ranking loss             — score real engagements above negatives.
  2. language-modeling loss    — preserve language understanding (anti-forget).
  3. reward-weighted alignment — weight toward long-term satisfaction, not just
                                 "engaged once".

Combined: L = a*rank + b*lm + c*reward. Each term is separable so we can ablate
them (blog reports Phase-2 losses contributing 35-50% of ranking gains).
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
    """Same contrastive objective, but each example is weighted by its reward
    (e.g. normalized rating / long-term-satisfaction signal). High-reward
    engagements dominate the gradient; fleeting/low-value ones count for less."""
    target = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
    per_ex = F.cross_entropy(scores, target, reduction="none")  # [B]
    w = reward / reward.mean().clamp(min=1e-6)                   # normalize ~1
    return (per_ex * w).mean()


def lm_loss(logits: torch.Tensor, input_ids: torch.Tensor,
            attention_mask: torch.Tensor) -> torch.Tensor:
    """Standard next-token cross-entropy over the prompt (shifted), masking pads.
    Keeps the backbone a competent language model while it learns to rank."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous().clone()
    shift_mask = attention_mask[:, 1:].contiguous()
    shift_labels[shift_mask == 0] = -100
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def combined_loss(rank, lm, reward, a=1.0, b=0.1, c=0.5):
    """Weighted sum used in Phase 2. Weights are the knobs; b (LM) acts as the
    'leash', c (reward) shapes which positives matter."""
    return a * rank + b * lm + c * reward
