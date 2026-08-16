"""Verbalization + context engineering (domain-aware).

Blog mapping: GenRec replaces *feature engineering* with *context engineering*.
Instead of hand-crafting dense features, we render the user's history as natural
language and let the backbone's attention decide what matters. The real work is
COMPACTION under a token budget: keep high-signal events, summarize repetitive
behaviour, drop low-value ones.

Domains let the same code serve Amazon products or restaurants by swapping the
natural-language framing (actor / verb / noun / call-to-action). `budget` is the
context-length lever we ablate (blog: "context length -> 1/3, negligible
degradation").
"""
from __future__ import annotations

from collections import Counter

from data import Interaction, Item

# --- domain framing --------------------------------------------------------- #
DOMAINS = {
    "product": dict(actor="shopper", verb="bought", noun="product",
                    cta="Recommend the next product this shopper will buy.",
                    outcome="They went on to buy"),
    "restaurant": dict(actor="diner", verb="ordered from", noun="restaurant",
                       cta="Recommend the next restaurant this diner will order from.",
                       outcome="They went on to order from"),
}
_DOMAIN = "product"
# generic category tags that carry no taste signal, skipped when describing items
_GENERIC = {"Restaurants", "Food", "Beauty", "Products", ""}


def set_domain(name: str) -> None:
    """Select the natural-language framing: 'product' (Amazon) or 'restaurant'."""
    global _DOMAIN
    assert name in DOMAINS, f"unknown domain {name}"
    _DOMAIN = name


def _descriptor(item: Item) -> str:
    """A short taste-carrying label: first non-generic category (cuisine / product
    type), e.g. 'Skin Care' or 'Thai'."""
    for c in item.categories:
        if c not in _GENERIC:
            return c
    return item.categories[0] if item.categories else ""


# Amazon titles run to 200+ chars ("... 3 Pack, 1.7 Fl Oz, Pack of 3, New Formula").
# Capping them keeps the prompt length bounded so the token budget is the ONLY
# thing controlling context size, and nothing gets silently right-truncated away.
_NAME_CAP = 64


def _clip(name: str, cap: int = _NAME_CAP) -> str:
    if len(name) <= cap:
        return name
    cut = name[:cap].rsplit(" ", 1)[0]
    return (cut or name[:cap]).rstrip(" ,-")


def _fmt_item(item: Item, rating: float) -> str:
    d = _descriptor(item)
    d = f", {d}" if d else ""
    return f"{_clip(item.name)}{d} ({rating:.0f} stars)"


def _span(seconds: float) -> str:
    """Human-readable elapsed time for the context block."""
    d = seconds / 86400.0
    if d < 1.5:
        return "under a day"
    if d < 45:
        return f"{d:.0f} days"
    if d < 400:
        return f"{d / 30.4:.0f} months"
    return f"{d / 365.0:.1f} years"


def verbalize_context(history: list[Interaction], now_ts: int | None = None) -> str:
    """The `tau` argument of the blog's verbalizer V(H, tau, metadata): the
    situational context the prediction is being made in, as opposed to the
    history itself. We have no device/daypart on these datasets, so we use what
    the logs do carry: how long the history spans, how dense it is, and how stale
    the last interaction is at prediction time."""
    if not history:
        return ""
    dom = DOMAINS[_DOMAIN]
    first, last = history[0].ts, history[-1].ts
    bits = [f"{len(history)} interactions"]
    if last > first:
        bits.append(f"over {_span(last - first)}")
    if now_ts and now_ts > last:
        bits.append(f"last one {_span(now_ts - last)} ago")
    return f"This {dom['actor']} has {', '.join(bits)}."


def verbalize_history(
    history: list[Interaction],
    catalog: dict[int, Item],
    budget: int = 10,
    now_ts: int | None = None,
) -> str:
    """Render a chronological history into a compacted natural-language prompt.

    This is V(H, tau, item metadata) -> x from the blog: situational context,
    then the history with its item metadata, then the task.

    Compaction (the 'context engineering'):
      1. Keep the most RECENT `budget` interactions verbatim (high signal).
      2. SUMMARIZE older ones into a taste profile (top categories + avg rating).
    """
    dom = DOMAINS[_DOMAIN]
    if not history:
        return f"A new {dom['actor']} with no history. {dom['cta']}"

    recent, older = history[-budget:], history[:-budget]
    lines: list[str] = []
    ctx = verbalize_context(history, now_ts)
    if ctx:
        lines.append(ctx)
    if older:
        cats = Counter(_descriptor(catalog[it.item_id]) for it in older)
        cats.pop("", None)
        avg = sum(it.rating for it in older) / len(older)
        top = ", ".join(c for c, _ in cats.most_common(3))
        lines.append(
            f"Earlier, they had {len(older)} interactions "
            f"(favouring {top}; avg {avg:.1f} stars).")

    recent_str = "; ".join(_fmt_item(catalog[it.item_id], it.rating) for it in recent)
    lines.append(f"Recently {dom['verb']}: {recent_str}.")
    lines.append(dom["cta"])
    return " ".join(lines)


def verbalize_target(item: Item, rating: float) -> str:
    """The ASSISTANT turn of the Phase-2 conversation.

    Blog: interaction logs become "single-turn or multi-turn 'conversations'"
    where the user message is the verbalized context/history/task and the
    assistant message is "the member's actual engagement (e.g. which titles were
    played, for how long)". The LM objective runs over inputs AND outputs, so the
    backbone learns how the engagement depends on the context rather than merely
    being restrained from drifting.
    """
    dom = DOMAINS[_DOMAIN]
    d = _descriptor(item)
    d = f", a {d} {dom['noun']}," if d else ""
    return f"{dom['outcome']} {_clip(item.name)}{d} and rated it {rating:.0f} stars."


def verbalize_item(item: Item) -> str:
    """Verbalize a catalog item (Phase-1 domain-adaptation corpus). The
    description/cost trade-off (blog: verbalization is a knob balancing signal
    vs. serving cost) lives here."""
    dom = DOMAINS[_DOMAIN]
    cats = ", ".join(c for c in item.categories if c not in _GENERIC) or item.city
    return (f"{_clip(item.name, 96)} — a {cats} {dom['noun']} "
            f"({item.city}) rated {item.stars} stars.")


if __name__ == "__main__":
    from data import synthetic_dataset

    set_domain("restaurant")
    ds = synthetic_dataset()
    u = next(iter(ds.test))
    hist = ds.train_hist[u]
    print("=== full budget ===")
    print(verbalize_history(hist, ds.catalog, budget=10))
    print("\n=== 1/3 budget (compacted) ===")
    print(verbalize_history(hist, ds.catalog, budget=3))
    print("\n=== item ===")
    print(verbalize_item(ds.catalog[0]))
