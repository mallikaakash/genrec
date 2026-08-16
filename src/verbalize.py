"""Verbalization + context engineering.

Blog mapping: GenRec replaces *feature engineering* with *context engineering*.
Instead of hand-crafting dense features, we render the user's history as natural
language and let the backbone's attention decide what matters. The real work is
COMPACTION under a token budget: keep high-signal events, summarize repetitive
behaviour, drop low-value ones.

`verbalize_history` returns a prompt string. The `budget` knob is the
context-length lever we ablate (blog: "context length reduced to 1/3 with
negligible degradation").
"""
from __future__ import annotations

from collections import Counter

from data import Interaction, Item


def _fmt_item(item: Item, rating: float) -> str:
    cui = next((c for c in item.categories if c != "Restaurants"), "")
    cui = f", {cui}" if cui else ""
    return f"{item.name}{cui} ({rating:.0f} stars)"


def verbalize_history(
    history: list[Interaction],
    catalog: dict[int, Item],
    budget: int = 10,
) -> str:
    """Render a chronological history into a compacted natural-language prompt.

    Compaction strategy (the 'context engineering'):
      1. Always keep the most RECENT `budget` interactions verbatim (high signal).
      2. Everything older is SUMMARIZED into a taste profile (top cuisines +
         average rating) rather than dropped outright.
    """
    if not history:
        return "A new diner with no order history. Recommend a restaurant."

    recent = history[-budget:]
    older = history[:-budget]

    lines: list[str] = []
    if older:
        cui_counts = Counter(
            c for it in older
            for c in catalog[it.item_id].categories if c != "Restaurants"
        )
        avg = sum(it.rating for it in older) / len(older)
        top = ", ".join(c for c, _ in cui_counts.most_common(3))
        lines.append(
            f"Earlier, this diner placed {len(older)} orders "
            f"(favouring {top}; avg {avg:.1f} stars)."
        )

    recent_str = "; ".join(
        _fmt_item(catalog[it.item_id], it.rating) for it in recent
    )
    lines.append(f"Recently ordered from: {recent_str}.")
    lines.append("Recommend the next restaurant this diner will order from.")
    return " ".join(lines)


def verbalize_item(item: Item) -> str:
    """Verbalize a catalog item (used for Phase-1 domain adaptation corpus and
    for text-side item embeddings). The description/cost trade-off (blog:
    verbalization is a knob balancing signal vs. serving cost) lives here."""
    cats = ", ".join(c for c in item.categories if c != "Restaurants") or "Restaurant"
    return f"{item.name} — a {cats} restaurant in {item.city} rated {item.stars} stars."


if __name__ == "__main__":
    from data import synthetic_dataset

    ds = synthetic_dataset()
    u = next(iter(ds.test))
    hist = ds.train_hist[u]
    print("=== full budget ===")
    print(verbalize_history(hist, ds.catalog, budget=10))
    print("\n=== 1/3 budget (compacted) ===")
    print(verbalize_history(hist, ds.catalog, budget=3))
    print("\n=== item ===")
    print(verbalize_item(ds.catalog[0]))
