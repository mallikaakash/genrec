"""Reward signals for the reward-weighted ranking objective.

Blog mapping — Phase-2 assigns each training example "a scalar weight derived
from two types of signals":

  1. Long-term satisfaction proxies. Not "did they click", but "was this a
     genuinely valuable engagement". Netflix uses watch duration and downstream
     retention. On Amazon we have the star rating (short-term) plus a review
     effort / helpfulness signal (`Interaction.engagement`), which is the closest
     available proxy for "this purchase actually mattered to them".

  2. Behaviour rebalancing. Raw logs are dominated by already-popular items, so
     a model trained on them just relearns popularity. We down-weight examples
     whose positive is a head item and up-weight tail items, via a damped inverse
     -propensity term (1/pop)^alpha.

The two multiply into one scalar per example, which then SCALES that example's
ranking loss (see losses.reward_weighted_loss). This is the blog's formulation:
"The example's ranking loss is scaled by this weight."
"""
from __future__ import annotations

from collections import Counter


def item_popularity(sequences: dict) -> Counter:
    """Training-side interaction count per item (held-out items excluded, since
    sequences are sliced before this is called by build_reward_fn)."""
    return Counter(it.item_id for seq in sequences.values() for it in seq)


def build_reward_fn(ds, alpha: float = 0.5, sat_weight: float = 0.5,
                    clip: tuple[float, float] = (0.25, 3.0)):
    """Returns reward(interaction) -> float, the per-example scalar weight.

    alpha:      strength of behaviour rebalancing. 0 = off, 1 = full inverse
                propensity. 0.5 is the usual damped setting.
    sat_weight: blend between the star rating and the engagement proxy inside
                the satisfaction term.
    clip:       keep the weight in a sane band so one tail item can't dominate a
                batch's gradient.
    """
    # popularity over TRAIN interactions only (everything before the val item)
    train_seqs = {u: seq[:-2] for u, seq in ds.sequences.items() if len(seq) > 2}
    pop = item_popularity(train_seqs)
    mean_pop = sum(pop.values()) / max(len(pop), 1)
    lo, hi = clip

    def reward(inter) -> float:
        # --- signal 1: long-term satisfaction proxy ---
        # rating 1..5 -> 0..1, blended with the engagement proxy
        r = (inter.rating - 1.0) / 4.0
        r = max(0.0, min(1.0, r))
        sat = (1.0 - sat_weight) * r + sat_weight * inter.engagement
        # map to a multiplier around 1.0: a 1-star/low-effort example counts ~0.4x,
        # a 5-star/high-effort one ~1.6x
        sat_w = 0.4 + 1.2 * sat

        # --- signal 2: behaviour rebalancing (damped inverse propensity) ---
        p = max(pop.get(inter.item_id, 1), 1)
        rebal = (mean_pop / p) ** alpha if alpha else 1.0

        return max(lo, min(hi, sat_w * rebal))

    return reward


def neutral_reward_fn():
    """Ablation arm: every example weighted 1.0 (i.e. plain ranking loss)."""
    return lambda inter: 1.0
