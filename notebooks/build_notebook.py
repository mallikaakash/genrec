"""Generates genrec_kaggle.ipynb (kept as a script so the notebook is diffable)."""
import json
from pathlib import Path

def md(src): return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}
def code(src): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src.splitlines(keepends=True)}

cells = [
    md("""# GenRec-Food on Kaggle (Yelp + Qwen2.5-0.5B)

Faithful small-scale reproduction of Netflix's GenRec: LLM backbone + catalog-aware
ranking head, two-phase training, three losses, prefill-only scoring.

**Setup:** add the *Yelp Dataset* as input data, enable GPU (T4/P100), then run top to bottom."""),

    md("### 1. Get the source\nClone/copy the `src/` package. If you uploaded this repo as a Kaggle dataset, adjust the path."),
    code("""import sys, os
# Option A: repo uploaded as a Kaggle dataset input
SRC = "/kaggle/input/genrec-food/src"
# Option B: pip/git clone your repo, or paste src/*.py into /kaggle/working/src
if not os.path.exists(SRC):
    SRC = "/kaggle/working/src"
sys.path.insert(0, SRC)
print("src:", SRC, "exists:", os.path.exists(SRC))"""),

    code("""!pip -q install "transformers>=4.44" "torch>=2.2" tqdm"""),

    md("""### 2. Load Yelp → catalog + sequences

Two presets:
* **Portfolio** (Swiggy/Zomato flavor): restaurants in one metro.
* **Paper-matched** (S3-Rec CIKM'20): all categories, reviews after 2019-01-01,
  5-core — directly comparable to the published benchmark in `BENCHMARKS.md`."""),
    code("""from data import load_yelp
YELP_DIR = "/kaggle/input/yelp-dataset"   # yelp_academic_dataset_business.json + review.json

PAPER_MATCHED = False   # flip to True for the apples-to-apples S3-Rec Yelp slice
if PAPER_MATCHED:
    ds = load_yelp(YELP_DIR, city="", restaurants_only=False,
                   after_date="2019-01-01", min_user_interactions=5)
else:
    ds = load_yelp(YELP_DIR, city=None, min_user_interactions=5, max_users=20000)
print("items:", ds.num_items, "users:", len(ds.sequences), "eval users:", len(ds.test))"""),

    md("### 3. Baselines (the bar to beat)"),
    code("""from baselines import PopularityBaseline, ItemKNNBaseline
from eval import evaluate, pretty
exclude = {u: {it.item_id for it in ds.train_hist[u]} for u in ds.test}
for name, m in [("Popularity", PopularityBaseline(ds, ds.train_hist)),
                ("ItemKNN", ItemKNNBaseline(ds, ds.train_hist))]:
    print(pretty(name, evaluate(m.scorer, ds.test, ds.num_items, ks=(10,), exclude=exclude)))"""),

    md("### 4. Build GenRec on a real backbone"),
    code("""import torch
from model import GenRec, build_backbone
device = "cuda" if torch.cuda.is_available() else "cpu"
lm, tok, hid = build_backbone("Qwen/Qwen2.5-0.5B", tiny=False)
model = GenRec(lm, hid, num_items=ds.num_items, scorer="dot").to(device)
print("device:", device, "hidden:", hid)"""),

    md("### 5. Phase 1 — domain adaptation (causal LM)"),
    code("""from train import phase1_adapt
phase1_adapt(model, tok, ds, device, epochs=1, lr=5e-5, budget=10, batch_size=16)"""),

    md("### 6. Phase 2 — ranking post-training (three losses)"),
    code("""from train import phase2_rank
phase2_rank(model, tok, ds, device, epochs=2, lr=1e-4, budget=10,
            batch_size=16, n_neg=8, weights=(1.0, 0.1, 0.5))"""),

    md("### 7. Evaluate GenRec (prefill-only scoring path)"),
    code("""from train import GenRecScorer
gr = GenRecScorer(model, tok, ds, device, budget=10)
print(pretty("GenRec", evaluate(gr.scorer, ds.test, ds.num_items, ks=(10,), exclude=exclude)))"""),

    md("""### 8. Ablations (mirroring the blog)
Phase-1 on/off, each loss on/off, context ⅓ — each is a fresh short run.
For speed on Kaggle, reduce `max_users` above or `p2-epochs` here."""),
    code("""from ablations import run
common = dict(ds=ds, device=device, model_name="Qwen/Qwen2.5-0.5B", tiny=False,
              budget=10, p1_epochs=1, p2_epochs=2)
print("no Phase-1 :", run("no_p1", do_phase1=False, **common))
print("no LM loss :", run("no_lm", use_lm=False, **common))
print("no reward  :", run("no_rew", use_reward=False, **common))
print("context 1/3:", run("ctx13", **{**common, "budget": 3}))"""),

    md("""### 9. Compare against published Yelp benchmarks

Only meaningful with `PAPER_MATCHED = True` above (same slice + protocol).
Published Yelp results (S3-Rec, CIKM 2020, Table 2 — see `BENCHMARKS.md`), all
under leave-one-out + 99 sampled negatives + 5-core:

| Model | HR@10 | NDCG@10 | MRR |
|---|---|---|---|
| PopRec | 0.3609 | 0.2007 | 0.1740 |
| GRU4Rec | 0.7265 | 0.4375 | 0.3630 |
| SASRec | 0.7373 | 0.4642 | 0.3927 |
| BERT4Rec | 0.7597 | 0.4778 | 0.4026 |
| S3-Rec | 0.7725 | 0.4934 | 0.4190 |

Put your GenRec (Recall@10 = HR@10, plus NDCG@10 and MRR) next to this row.
Landing near SASRec/BERT4Rec is a competitive result."""),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}

out = Path(__file__).parent / "genrec_kaggle.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)
