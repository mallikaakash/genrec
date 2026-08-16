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
                    cta="Recommend the next product this shopper will buy."),
    "restaurant": dict(actor="diner", verb="ordered from", noun="restaurant",
                       cta="Recommend the next restaurant this diner will order from."),
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


def _fmt_item(item: Item, rating: float) -> str:
    d = _descriptor(item)
    d = f", {d}" if d else ""
    return f"{item.name}{d} ({rating:.0f} stars)"


def verbalize_history(
    history: list[Interaction],
    catalog: dict[int, Item],
    budget: int = 10,
) -> str:
    """Render a chronological history into a compacted natural-language prompt.

    Compaction (the 'context engineering'):
      1. Keep the most RECENT `budget` interactions verbatim (high signal).
      2. SUMMARIZE older ones into a taste profile (top categories + avg rating).
    """
    dom = DOMAINS[_DOMAIN]
    if not history:
        return f"A new {dom['actor']} with no history. {dom['cta']}"

    recent, older = history[-budget:], history[:-budget]
    lines: list[str] = []
    if older:
        cats = Counter(_descriptor(catalog[it.item_id]) for it in older)
        cats.pop("", None)
        avg = sum(it.rating for it in older) / len(older)
        top = ", ".join(c for c, _ in cats.most_common(3))
        lines.append(
            f"Earlier, this {dom['actor']} had {len(older)} interactions "
            f"(favouring {top}; avg {avg:.1f} stars).")

    recent_str = "; ".join(_fmt_item(catalog[it.item_id], it.rating) for it in recent)
    lines.append(f"Recently {dom['verb']}: {recent_str}.")
    lines.append(dom["cta"])
    return " ".join(lines)


def verbalize_item(item: Item) -> str:
    """Verbalize a catalog item (Phase-1 domain-adaptation corpus). The
    description/cost trade-off (blog: verbalization is a knob balancing signal
    vs. serving cost) lives here."""
    dom = DOMAINS[_DOMAIN]
    cats = ", ".join(c for c in item.categories if c not in _GENERIC) or item.city
    return (f"{item.name} — a {cats} {dom['noun']} "
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
