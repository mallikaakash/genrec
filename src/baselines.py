"""Baselines — the bar GenRec must beat.

Blog mapping: off-the-shelf LLMs "over-recommend popular content". Popularity is
exactly that strawman. Item-kNN is a genuinely strong, classic sequential-rec
baseline; beating it is the real test.

Both expose the same `scorer(user_id, candidates) -> scores` interface as GenRec,
so eval.evaluate treats them identically.
"""
from __future__ import annotations

from collections import Counter, defaultdict

from data import Dataset


class PopularityBaseline:
    """Score every candidate by global order count. Ignores the user entirely."""

    def __init__(self, ds: Dataset, hist: dict[int, list]):
        counts: Counter[int] = Counter()
        for seq in hist.values():
            for it in seq:
                counts[it.item_id] += 1
        self.counts = counts

    def scorer(self, uid: int, candidates: list[int]) -> list[float]:
        return [float(self.counts.get(c, 0)) for c in candidates]


class ItemKNNBaseline:
    """Co-occurrence item-kNN. score(candidate) = sum of similarity between the
    candidate and the items in the user's history. Similarity = cosine over
    co-occurrence in user sequences. A strong classical sequential baseline."""

    def __init__(self, ds: Dataset, hist: dict[int, list]):
        self.hist = hist
        # build item -> set of users, and co-occurrence counts
        cooc: dict[int, Counter[int]] = defaultdict(Counter)
        item_freq: Counter[int] = Counter()
        for seq in hist.values():
            items = [it.item_id for it in seq]
            uniq = set(items)
            for i in uniq:
                item_freq[i] += 1
            for i in uniq:
                for j in uniq:
                    if i != j:
                        cooc[i][j] += 1
        self.cooc = cooc
        self.item_freq = item_freq

    def _sim(self, i: int, j: int) -> float:
        c = self.cooc.get(i, {}).get(j, 0)
        if c == 0:
            return 0.0
        denom = (self.item_freq[i] * self.item_freq[j]) ** 0.5
        return c / denom if denom else 0.0

    def scorer(self, uid: int, candidates: list[int]) -> list[float]:
        hist_items = [it.item_id for it in self.hist.get(uid, [])]
        out = []
        for c in candidates:
            out.append(sum(self._sim(c, h) for h in hist_items))
        return out


if __name__ == "__main__":
    from data import synthetic_dataset
    from eval import evaluate, pretty

    ds = synthetic_dataset()
    exclude = {u: {it.item_id for it in ds.train_hist[u]} for u in ds.test}

    pop = PopularityBaseline(ds, ds.train_hist)
    knn = ItemKNNBaseline(ds, ds.train_hist)

    for name, model in [("Popularity", pop), ("ItemKNN", knn)]:
        m = evaluate(model.scorer, ds.test, ds.num_items, ks=(5, 10),
                     n_neg=99, exclude=exclude)
        print(pretty(name, m))
