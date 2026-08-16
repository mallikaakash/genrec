"""Amazon Reviews (McAuley 2014) loader — the canonical sequential-rec benchmark.

Loads a single category's **5-core** file (already filtered so every user and item
has >=5 interactions), which is the exact slice used by SASRec, BERT4Rec, S3-Rec,
and TIGER. For Beauty this is 198,502 reviews — matching S3-Rec Table 1 exactly,
so our numbers are directly comparable to the published benchmark (BENCHMARKS.md).

Two files per category, auto-downloaded from the SNAP mirror (ungated):
  reviews_<Cat>_5.json.gz : JSON-per-line. reviewerID, asin, overall (rating),
                            unixReviewTime, summary, reviewText, ...
  meta_<Cat>.json.gz      : PYTHON-DICT-per-line (single quotes -> ast.literal_eval).
                            asin, title, categories (nested list), description, ...

Produces the same Dataset/Item/Interaction structure as data.py, so the rest of
the pipeline (verbalize -> model -> train -> eval) is unchanged.
"""
from __future__ import annotations

import ast
import gzip
import json
import urllib.request
from collections import defaultdict
from pathlib import Path

from data import Dataset, Interaction, Item, _kcore_filter, _leave_one_out

SNAP = "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles"


def _download(category: str, cache_dir: Path) -> tuple[Path, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    rev_fn, meta_fn = f"reviews_{category}_5.json.gz", f"meta_{category}.json.gz"
    paths = []
    for fn in (rev_fn, meta_fn):
        p = cache_dir / fn
        if not p.exists():
            print(f"[amazon] downloading {fn} ...")
            urllib.request.urlretrieve(f"{SNAP}/{fn}", p)
        paths.append(p)
    return paths[0], paths[1]


def load_amazon(
    category: str = "Beauty",
    cache_dir: str | Path = "amazon_cache",
    min_user_interactions: int = 5,
    max_users: int | None = None,
) -> Dataset:
    rev_path, meta_path = _download(category, Path(cache_dir))

    # --- reviews -> rows (user, item, rating, ts) + running item rating avg ---
    rows: list[tuple[str, str, float, int]] = []
    rsum: dict[str, float] = defaultdict(float)
    rcnt: dict[str, int] = defaultdict(int)
    with gzip.open(rev_path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            r = json.loads(line)
            asin = r["asin"]
            rating = float(r.get("overall", 0.0))
            rows.append((r["reviewerID"], asin, rating, int(r.get("unixReviewTime", 0))))
            rsum[asin] += rating
            rcnt[asin] += 1
    needed = set(rsum)  # only asins that actually appear in reviews

    # --- metadata: only literal_eval lines for needed asins (12k not 259k) ---
    import re
    asin_re = re.compile(r"^\{'asin':\s*'([^']+)'")
    meta: dict[str, tuple[str, list[str], str]] = {}
    with gzip.open(meta_path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = asin_re.match(line)
            if not m or m.group(1) not in needed:
                continue
            try:
                d = ast.literal_eval(line)      # single-quoted dicts, not JSON
            except (ValueError, SyntaxError):
                continue
            cats_nested = d.get("categories") or [[]]
            flat = [c for sub in cats_nested for c in sub]
            meta[d.get("asin", "")] = (
                d.get("title") or "Unknown item", flat, d.get("description", "") or "")

    # --- catalog: items that appear in reviews ---
    asin_to_item: dict[str, int] = {}
    catalog: dict[int, Item] = {}
    for _, asin, _, _ in rows:
        if asin in asin_to_item:
            continue
        title, cats, _desc = meta.get(asin, (f"Item {asin}", [category], ""))
        top = cats[0] if cats else category
        avg = round(rsum[asin] / max(rcnt[asin], 1), 2)
        iid = len(catalog)
        asin_to_item[asin] = iid
        # store the sub-category path (drop the top tag like "Beauty") for verbalization
        subcats = [c for c in cats if c != top] or [top]
        catalog[iid] = Item(iid, title, subcats, top, avg)
    print(f"[amazon] {category}: {len(catalog)} items, {len(rows)} reviews")

    # --- sequences ---
    raw: dict[str, list[Interaction]] = defaultdict(list)
    for uid, asin, rating, ts in rows:
        raw[uid].append(Interaction(asin_to_item[asin], rating, ts))

    raw = _kcore_filter(dict(raw), k=min_user_interactions)
    sequences: dict[int, list[Interaction]] = {}
    for uid, inters in raw.items():
        inters.sort(key=lambda x: x.ts)
        sequences[len(sequences)] = inters
        if max_users and len(sequences) >= max_users:
            break
    n_items = len({it.item_id for s in sequences.values() for it in s})
    print(f"[amazon] after {min_user_interactions}-core: {len(sequences)} users, "
          f"{n_items} items")
    return _leave_one_out(Dataset(catalog=catalog, sequences=sequences))


if __name__ == "__main__":
    import verbalize
    verbalize.set_domain("product")
    ds = load_amazon("Beauty", cache_dir="/tmp/amazon_cache")
    print("num_items:", ds.num_items, "users:", len(ds.sequences), "eval:", len(ds.test))
    u = next(iter(ds.test))
    print("\nsample prompt:\n", verbalize.verbalize_history(ds.train_hist[u], ds.catalog, budget=5))
    print("\nsample item:\n", verbalize.verbalize_item(ds.catalog[ds.test[u]]))
