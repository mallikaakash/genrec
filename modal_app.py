"""Run GenRec-Food training on Modal (serverless GPU).

Usage:
    modal run modal_app.py                      # synthetic data, real Qwen backbone
    modal run modal_app.py --dataset yelp       # real Yelp (requires the volume below)

Data:
  * synthetic: built in-container, no external data. Proves the GPU pipeline +
    real backbone end-to-end.
  * yelp: expects the Yelp Open Dataset JSON files in a Modal Volume named
    "yelp-data" at /data (business.json + review.json). Populate it once with:
        modal volume create yelp-data
        modal volume put yelp-data yelp_academic_dataset_business.json /
        modal volume put yelp-data yelp_academic_dataset_review.json /
"""
import modal

app = modal.App("genrec-food")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.2", "transformers>=4.44", "numpy", "tqdm")
    # ship our flat-module source into the container and onto sys.path
    .add_local_dir("src", remote_path="/root/src")
)

# Optional: attach the Yelp dataset volume if it exists (created on demand).
yelp_vol = modal.Volume.from_name("yelp-data", create_if_missing=True)
# Cache HF model weights across runs so we don't re-download Qwen every time.
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    timeout=90 * 60,
    volumes={"/data": yelp_vol, "/root/.cache/huggingface": hf_cache},
)
def train(dataset: str = "amazon_beauty", model_name: str = "Qwen/Qwen2.5-0.5B",
          p1_epochs: int = 1, p2_epochs: int = 2, budget: int = 10,
          max_users: int = 0, max_train_examples: int = 60000,
          eval_users: int = 8000):
    import sys
    sys.path.insert(0, "/root/src")
    import torch

    import verbalize
    from data import synthetic_dataset, load_yelp
    from model import GenRec, build_backbone
    from baselines import PopularityBaseline, ItemKNNBaseline
    from eval import evaluate, pretty
    from train import phase1_adapt, phase2_rank, GenRecScorer

    print("CUDA available:", torch.cuda.is_available(),
          "| device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mu = max_users or None

    if dataset.startswith("amazon_"):
        from amazon import load_amazon
        category = dataset.split("amazon_", 1)[1].title()  # amazon_beauty -> Beauty
        verbalize.set_domain("product")
        ds = load_amazon(category, cache_dir="/root/.cache/amazon",
                         min_user_interactions=5, max_users=mu)
        tag = f"Amazon {category} (2014 5-core — comparable to S3-Rec/TIGER)"
    elif dataset == "yelp":
        verbalize.set_domain("restaurant")
        ds = load_yelp("/data", city="", restaurants_only=False,
                       after_date="2019-01-01", min_user_interactions=5, max_users=mu)
        tag = "Yelp (paper-matched)"
    else:
        verbalize.set_domain("restaurant")
        ds = synthetic_dataset()
        tag = "SYNTHETIC (toy data — not a benchmark)"

    print(f"\n=== dataset: {tag} | items={ds.num_items} users={len(ds.sequences)} "
          f"eval={len(ds.test)} ===")

    # Eval protocol matches the published benchmark: leave-one-out, 99 sampled
    # negatives (evaluate default), HR@{5,10}/NDCG@{5,10}/MRR. For a tractable
    # first run we score a random sample of test users (unbiased MRR estimate).
    import random as _r
    test_users = list(ds.test)
    if eval_users and len(test_users) > eval_users:
        test_users = _r.Random(0).sample(test_users, eval_users)
    eval_items = {u: ds.test[u] for u in test_users}
    print(f"[eval] scoring {len(eval_items)} test users, 99 sampled negatives")

    exclude = {u: {it.item_id for it in ds.train_hist[u]} for u in eval_items}
    results = {}
    for name, m in [("Popularity", PopularityBaseline(ds, ds.train_hist)),
                    ("ItemKNN", ItemKNNBaseline(ds, ds.train_hist))]:
        results[name] = evaluate(m.scorer, eval_items, ds.num_items, ks=(5, 10),
                                 exclude=exclude)
        print(pretty(name, results[name]))

    lm, tok, hid = build_backbone(model_name, tiny=False)
    model = GenRec(lm, hid, num_items=ds.num_items, scorer="dot").to(device)
    print(f"\nbackbone={model_name} hidden={hid} params="
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # --- full training tracking: 3 losses, perplexity/bpb, train vs val ---
    from tracker import Tracker
    tracker = Tracker("/root/.cache/genrec_history.jsonl", use_tb=False,
                      use_wandb=bool(__import__("os").environ.get("WANDB_API_KEY")),
                      run_name=tag, config={"dataset": dataset, "model": model_name,
                                            "p1_epochs": p1_epochs, "p2_epochs": p2_epochs})

    _, gstep = phase1_adapt(model, tok, ds, device, epochs=p1_epochs, budget=budget,
                            batch_size=16, tracker=tracker)
    _, gstep = phase2_rank(model, tok, ds, device, epochs=p2_epochs, budget=budget,
                           batch_size=16, n_neg=8, max_examples=max_train_examples,
                           tracker=tracker, gstep0=gstep)
    tracker.close()

    gr = GenRecScorer(model, tok, ds, device, budget=budget)
    results["GenRec"] = evaluate(gr.scorer, eval_items, ds.num_items, ks=(5, 10),
                                 exclude=exclude)

    print("\n" + "=" * 60 + f"\nRESULTS — {tag}")
    for name, mt in results.items():
        print(pretty(name, mt))
    return {"tag": tag, "results": results, "history": tracker.history(),
            "items": ds.num_items, "users": len(ds.sequences),
            "eval_users": len(eval_items)}


@app.local_entrypoint()
def main(dataset: str = "amazon_beauty", p1_epochs: int = 1, p2_epochs: int = 2):
    import json
    from pathlib import Path
    out = train.remote(dataset=dataset, p1_epochs=p1_epochs, p2_epochs=p2_epochs)
    art = Path("artifacts"); art.mkdir(exist_ok=True)
    (art / "history.jsonl").write_text(
        "\n".join(json.dumps(r) for r in out["history"]))
    (art / "results.json").write_text(json.dumps(
        {k: out[k] for k in ("tag", "results", "items", "users", "eval_users")}, indent=2))
    print("\n[local] returned:", out["tag"], "| GenRec MRR:",
          round(out["results"]["GenRec"]["MRR"], 4))
    print("[local] wrote artifacts/history.jsonl + artifacts/results.json")
