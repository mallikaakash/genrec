# Verified Yelp benchmarks (published, not reproduced by us)

These are the **real, published** sequential-recommendation results on the Yelp
dataset, transcribed exactly from **Table 2 of the S3-Rec paper** (Zhou et al.,
*S3-Rec: Self-Supervised Learning for Sequential Recommendation with Mutual
Information Maximization*, CIKM 2020, [arXiv:2008.07873](https://arxiv.org/abs/2008.07873)).
They are the bar GenRec-Food is measured against.

## Why these are directly comparable to our setup

The S3-Rec Yelp protocol matches what `eval.py` already does:

| Aspect | S3-Rec (paper) | GenRec-Food |
|---|---|---|
| Split | leave-one-out (last=test, 2nd-last=val) | leave-one-out (`data.py`) |
| Candidates at eval | 1 positive + **99 sampled negatives** | `n_neg=99` (`eval.py`) |
| Filtering | iterative **5-core** (users & items ≥5) | `_kcore_filter(k=5)` (`data.py`) |
| Metrics | HR@k, NDCG@k, MRR | Recall@k(=HR@k), NDCG@k, MRR |

To reproduce the paper's exact Yelp slice, load with the **paper-matched preset**:

```python
load_yelp(YELP_DIR, city="", restaurants_only=False,
          after_date="2019-01-01", min_user_interactions=5)
```

The paper's Yelp preprocessing: reviews after **2019-01-01**, **all** business
categories (not restaurants-only), categories used as item attributes, 5-core.
After preprocessing their Yelp slice has **30,431 users / 20,033 items / 316,354
actions** (avg 10.4 actions/user, 99.95% sparse).

> **Note on the two modes.** Our *portfolio* default (`restaurants_only=True`,
> one city) is a Swiggy/Zomato-flavored **subset** and is NOT directly comparable
> to these numbers. Use the *paper-matched* preset above for an apples-to-apples
> comparison; use the portfolio mode for the narrative demo.

## Published Yelp results — S3-Rec Table 2 (exact values)

| Model | HR@1 | HR@5 | NDCG@5 | HR@10 | NDCG@10 | MRR |
|---|---|---|---|---|---|---|
| PopRec | 0.0801 | 0.2415 | 0.1622 | 0.3609 | 0.2007 | 0.1740 |
| FM | 0.0624 | 0.2036 | 0.1333 | 0.3153 | 0.1692 | 0.1470 |
| AutoInt | 0.0731 | 0.2249 | 0.1501 | 0.3367 | 0.1860 | 0.1616 |
| GRU4Rec | 0.2053 | 0.5437 | 0.3784 | 0.7265 | 0.4375 | 0.3630 |
| Caser | 0.2188 | 0.5111 | 0.3696 | 0.6661 | 0.4198 | 0.3595 |
| SASRec | 0.2375 | 0.5745 | 0.4113 | 0.7373 | 0.4642 | 0.3927 |
| BERT4Rec | 0.2405 | 0.5976 | 0.4252 | 0.7597 | 0.4778 | 0.4026 |
| HGN | 0.2428 | 0.5768 | 0.4162 | 0.7411 | 0.4695 | 0.3988 |
| GRU4RecF | 0.2293 | 0.5858 | 0.4137 | 0.7574 | 0.4694 | 0.3929 |
| SASRecF | 0.2301 | 0.5937 | 0.4178 | 0.7706 | 0.4751 | 0.3962 |
| FDSA | 0.2198 | 0.5728 | 0.4014 | 0.7555 | 0.4607 | 0.3834 |
| **S3-Rec** | **0.2591** | **0.6085** | **0.4401** | **0.7725** | **0.4934** | **0.4190** |

HR@10(=Recall@10) and MRR are the columns to compare GenRec-Food against.
Reference targets on Yelp:

- **Popularity floor:** MRR 0.174, HR@10 0.361.
- **Strong self-attentive models (SASRec/BERT4Rec):** MRR ~0.39–0.40, HR@10 ~0.74–0.76, NDCG@10 ~0.46–0.48.
- **Best published (S3-Rec):** MRR 0.419, HR@10 0.773, NDCG@10 0.493.

A GenRec-Food run that lands near SASRec/BERT4Rec on the paper-matched slice
would be a genuinely competitive result.

## Dataset facts (verified)

Yelp Open Dataset (official, [business.yelp.com/data/resources/open-dataset](https://business.yelp.com/data/resources/open-dataset/)):
**6,990,280 reviews · 150,346 businesses · 200,100 pictures · 11 metropolitan areas.**
Files: `business.json`, `review.json`, `user.json`, `checkin.json`, `tip.json`
(one JSON object per line).

Fields we use (verified against Yelp's schema documentation):
- `business.json`: `business_id`, `name`, `city`, `state`, `stars` (avg rating),
  `review_count`, `categories` (comma-separated string), `attributes`, `hours`.
- `review.json`: `review_id`, `user_id`, `business_id`, `stars` (this review),
  `useful`, `funny`, `cool`, `text`, `date` (`"YYYY-MM-DD HH:MM:SS"`).

## Sources

- S3-Rec, CIKM 2020 — https://arxiv.org/abs/2008.07873 (Table 2, Yelp column; §5.1 protocol)
- Official code/data — https://github.com/RUCAIBox/CIKM2020-S3Rec
- Yelp Open Dataset — https://business.yelp.com/data/resources/open-dataset/
