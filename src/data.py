"""Data prep for GenRec-food.

Loads Yelp Open Dataset (business.json + review.json), filters to restaurants
in one metro, builds:
  - a *catalog*: item_id -> metadata (name, categories, city, stars)
  - per-user chronological interaction *sequences* (with rating + timestamp)
  - a leave-one-out split (last item = test, 2nd-last = val, rest = train)

For local development without the ~5GB Yelp dump, `synthetic_dataset()` builds a
tiny in-memory dataset with the exact same shapes so the whole pipeline
(verbalize -> model -> train -> eval) can be smoke-tested on a laptop.

Blog mapping: this is the raw-interaction-log stage. We deliberately keep the
*text* metadata (name/categories) because verbalization (verbalize.py) turns it
into natural language rather than dense features.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Item:
    item_id: int
    name: str
    categories: list[str]
    city: str
    stars: float  # catalog-level average rating


@dataclass
class Interaction:
    item_id: int
    rating: float  # this user's rating (reward signal)
    ts: int        # unix timestamp, for chronological ordering


@dataclass
class Dataset:
    catalog: dict[int, Item]
    # user_id -> chronological list of interactions
    sequences: dict[int, list[Interaction]]
    # leave-one-out splits: user_id -> item_id
    val: dict[int, int] = field(default_factory=dict)
    test: dict[int, int] = field(default_factory=dict)
    # user_id -> history used at train/eval time (everything before the held-out item)
    train_hist: dict[int, list[Interaction]] = field(default_factory=dict)
    val_hist: dict[int, list[Interaction]] = field(default_factory=dict)

    @property
    def num_items(self) -> int:
        return len(self.catalog)


def _leave_one_out(ds: Dataset, min_len: int = 3) -> Dataset:
    """Standard sequential-rec split: for each user with >= min_len items,
    hold out the last (test) and second-to-last (val)."""
    for uid, seq in ds.sequences.items():
        if len(seq) < min_len:
            continue
        ds.test[uid] = seq[-1].item_id
        ds.val[uid] = seq[-2].item_id
        ds.val_hist[uid] = seq[:-2]      # history for predicting val item
        ds.train_hist[uid] = seq[:-1]    # history for predicting test item
    return ds


def synthetic_dataset(
    n_users: int = 200,
    n_items: int = 120,
    seed: int = 0,
) -> Dataset:
    """Tiny dataset with latent 'cuisine taste' so a real model can beat
    popularity. Each user has a preferred cuisine; they mostly order from it."""
    rng = random.Random(seed)
    cuisines = ["Italian", "Thai", "Indian", "Mexican", "Japanese", "American"]
    words = ["Kitchen", "House", "Grill", "Bistro", "Corner", "Spot", "Place", "Cafe"]

    catalog: dict[int, Item] = {}
    items_by_cuisine: dict[str, list[int]] = {c: [] for c in cuisines}
    for iid in range(n_items):
        cui = cuisines[iid % len(cuisines)]
        name = f"{cui} {rng.choice(words)} {iid}"
        catalog[iid] = Item(iid, name, [cui, "Restaurants"], "Testville",
                             round(rng.uniform(2.5, 5.0), 1))
        items_by_cuisine[cui].append(iid)

    sequences: dict[int, list[Interaction]] = {}
    for uid in range(n_users):
        fav = rng.choice(cuisines)
        seq_len = rng.randint(4, 12)
        seq: list[Interaction] = []
        ts = 1_500_000_000
        for _ in range(seq_len):
            # 80% order from favourite cuisine, else random
            if rng.random() < 0.8:
                iid = rng.choice(items_by_cuisine[fav])
                rating = rng.choice([4, 4, 5, 5, 3])
            else:
                iid = rng.randrange(n_items)
                rating = rng.choice([2, 3, 3, 4])
            ts += rng.randint(3600, 3 * 86400)
            seq.append(Interaction(iid, float(rating), ts))
        sequences[uid] = seq

    return _leave_one_out(Dataset(catalog=catalog, sequences=sequences))


def _food_categories() -> set[str]:
    # Yelp lumps many things under "Restaurants"; keep it broad but food-only.
    return {"Restaurants", "Food"}


def load_yelp(
    data_dir: str | Path,
    city: str | None = None,
    min_user_interactions: int = 5,
    max_users: int | None = None,
    restaurants_only: bool = True,
    after_date: str | None = None,
) -> Dataset:
    """Load the Yelp Open Dataset.

    Expects `business.json` and `review.json` (one JSON object per line) in
    `data_dir`. On Kaggle the 'yelp-dataset' input provides exactly these files.

    Real schema used (verified against Yelp's official documentation):
      business.json: business_id, name, city, stars (avg), categories (comma str)
      review.json:   user_id, business_id, stars (this review), date ("YYYY-MM-DD HH:MM:SS")

    Args:
        city: restrict to one metro to keep the catalog small. None = most-reviewed
              city. Set city="" to disable the city filter entirely.
        restaurants_only: keep only businesses whose categories contain
              Restaurants/Food. Set False to include ALL categories.
        after_date: e.g. "2019-01-01" keeps only reviews on/after this date.

    Two ready-made presets:
      * Portfolio (default): restaurants_only=True, one city  -> Swiggy/Zomato flavor.
      * Paper-matched (S3-Rec CIKM'20): restaurants_only=False, city="",
        after_date="2019-01-01", min_user_interactions=5  -> directly comparable
        to the published Yelp benchmark (leave-one-out + 99 negatives + 5-core).
    """
    data_dir = Path(data_dir)
    biz_path = data_dir / "yelp_academic_dataset_business.json"
    rev_path = data_dir / "yelp_academic_dataset_review.json"
    if not biz_path.exists():  # some mirrors drop the prefix
        biz_path = data_dir / "business.json"
        rev_path = data_dir / "review.json"

    food = _food_categories()

    # --- pass 1: pick the city if unspecified, collect (restaurant) businesses ---
    city_counts: dict[str, int] = {}
    biz_rows: list[dict] = []
    with open(biz_path) as f:
        for line in f:
            b = json.loads(line)
            cats = (b.get("categories") or "")
            if restaurants_only and not any(c in cats for c in food):
                continue
            biz_rows.append(b)
            city_counts[b.get("city", "")] = city_counts.get(b.get("city", ""), 0) + 1

    if city is None:
        city = max(city_counts, key=city_counts.get)
        print(f"[data] auto-selected city='{city}' ({city_counts[city]} businesses)")

    # --- build catalog (all cities if city == "") ---
    biz_to_item: dict[str, int] = {}
    catalog: dict[int, Item] = {}
    for b in biz_rows:
        if city and b.get("city") != city:
            continue
        cats = [c.strip() for c in (b.get("categories") or "").split(",") if c.strip()]
        iid = len(catalog)
        biz_to_item[b["business_id"]] = iid
        catalog[iid] = Item(iid, b.get("name", "Unknown"), cats,
                            b.get("city", city or ""), float(b.get("stars", 0.0)))
    print(f"[data] catalog: {len(catalog)} businesses"
          f"{' in ' + city if city else ' (all cities)'}")

    # --- pass 2: reviews -> interactions for those businesses ---
    cutoff = _parse_date(after_date + " 00:00:00") if after_date else 0
    raw: dict[str, list[Interaction]] = {}
    with open(rev_path) as f:
        for line in f:
            r = json.loads(line)
            bid = r.get("business_id")
            iid = biz_to_item.get(bid)
            if iid is None:
                continue
            ts = _parse_date(r.get("date", ""))
            if ts < cutoff:
                continue
            uid = r["user_id"]
            raw.setdefault(uid, []).append(
                Interaction(iid, float(r.get("stars", 0.0)), ts)
            )

    # Iterative k-core filtering (matches S3-Rec: repeatedly drop users AND items
    # with < k interactions until stable). k = min_user_interactions.
    raw = _kcore_filter(raw, k=min_user_interactions)

    # remap user ids to ints, sort chronologically
    sequences: dict[int, list[Interaction]] = {}
    for uid, inters in raw.items():
        inters.sort(key=lambda x: x.ts)
        sequences[len(sequences)] = inters
        if max_users and len(sequences) >= max_users:
            break
    n_items_used = len({it.item_id for s in sequences.values() for it in s})
    print(f"[data] after {min_user_interactions}-core: {len(sequences)} users, "
          f"{n_items_used} items with interactions")

    return _leave_one_out(Dataset(catalog=catalog, sequences=sequences))


def _kcore_filter(raw: dict[str, list[Interaction]], k: int
                  ) -> dict[str, list[Interaction]]:
    """Repeatedly remove users with < k interactions and items appearing < k
    times, until both conditions hold simultaneously (standard k-core)."""
    from collections import Counter
    while True:
        item_freq = Counter(it.item_id for inters in raw.values() for it in inters)
        changed = False
        new: dict[str, list[Interaction]] = {}
        for uid, inters in raw.items():
            kept = [it for it in inters if item_freq[it.item_id] >= k]
            if len(kept) != len(inters):
                changed = True
            if len(kept) >= k:
                new[uid] = kept
            else:
                changed = True
        raw = new
        if not changed:
            return raw


def _parse_date(s: str) -> int:
    # Yelp dates look like "2016-03-09 20:56:38"
    import datetime as _dt
    try:
        return int(_dt.datetime.fromisoformat(s).timestamp())
    except Exception:
        return 0


if __name__ == "__main__":
    ds = synthetic_dataset()
    print("num_items:", ds.num_items)
    print("num_users:", len(ds.sequences))
    print("eval users:", len(ds.test))
    u = next(iter(ds.test))
    print(f"user {u} history len:", len(ds.train_hist[u]),
          "-> test item:", ds.catalog[ds.test[u]].name)
