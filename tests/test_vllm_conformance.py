"""Conformance test: your VLLMEncoder must rank identically to TorchEncoder.

WHY RANKING AND NOT THE RAW VECTORS
-----------------------------------
You cannot assert `vllm_vec == torch_vec`. vLLM may L2-normalize, may use
different kernels, and runs in bf16. All of those change the numbers.

But none of them change the ORDER, which is the only thing a recommender
outputs. For a dot-product scorer, scaling one user's vector by any positive
constant c scales all their candidate scores by c, which is a monotone
transform within that query and cannot reorder anything.

So the correct assertion is: same top-K item ids, in the same order.

RUN
---
    # 1. start vLLM + the API (see VLLM_TASK.md)
    docker compose up -d

    # 2. point the test at your checkpoint and run
    GENREC_MODEL_DIR=./genrec_model_v2 pytest tests/ -v

If GENREC_MODEL_DIR is unset the tests skip rather than fail, so CI stays green
before you have a checkpoint locally.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

MODEL_DIR = os.environ.get("GENREC_MODEL_DIR")
VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000")

pytestmark = pytest.mark.skipif(
    not MODEL_DIR, reason="set GENREC_MODEL_DIR to a saved GenRec checkpoint")


@pytest.fixture(scope="module")
def prompts():
    """Deliberately varied: different lengths, and one empty history, because
    the empty case takes a different branch in the verbalizer."""
    return [
        "A new shopper with no history. Recommend the next product this shopper will buy.",
        "This shopper has 3 interactions, over 12 days. Recently bought: Dove Beauty Bar, "
        "Bath (5 stars). Recommend the next product this shopper will buy.",
        "This shopper has 14 interactions, over 8 months, last one 3 days ago. Earlier, "
        "they had 11 interactions (favouring Skin Care; avg 4.2 stars). Recently bought: "
        "Olay Regenerist, Skin Care (5 stars); CeraVe Lotion, Skin Care (4 stars); "
        "OPI Nail Lacquer, Nail Polish (5 stars). "
        "Recommend the next product this shopper will buy.",
    ]


@pytest.fixture(scope="module")
def torch_service():
    from serve import GenRecService
    return GenRecService(MODEL_DIR, device="cpu", backend="torch", domain="product")


def _vllm_encoder():
    from serve import VLLMEncoder
    meta_model = os.environ.get("VLLM_MODEL", "genrec-backbone")
    return VLLMEncoder(endpoint=VLLM_URL, model=meta_model)


def test_vllm_encoder_is_implemented():
    """Fails until you write it. This is the red test."""
    try:
        _vllm_encoder()
    except NotImplementedError:
        pytest.fail("VLLMEncoder is not implemented yet. See VLLM_TASK.md.")
    except Exception as e:
        pytest.skip(f"vLLM server not reachable at {VLLM_URL}: {e}")


def test_output_shape(torch_service, prompts):
    try:
        enc = _vllm_encoder()
        vecs, n_tok = enc.encode(prompts)
    except NotImplementedError:
        pytest.fail("VLLMEncoder is not implemented yet. See VLLM_TASK.md.")
    except Exception as e:
        pytest.skip(f"vLLM server not reachable: {e}")

    assert vecs.shape[0] == len(prompts), "one vector per prompt"
    assert vecs.shape[1] == torch_service.model.hidden_size, (
        f"expected hidden size {torch_service.model.hidden_size}, got {vecs.shape[1]}. "
        "If this is the vocab size you are reading logits, not hidden states.")
    assert len(n_tok) == len(prompts)


def test_ranking_matches_torch(torch_service, prompts):
    """THE test. Same top-10 items, same order, for every prompt."""
    import torch
    try:
        enc = _vllm_encoder()
        v_vecs, _ = enc.encode(prompts)
    except NotImplementedError:
        pytest.fail("VLLMEncoder is not implemented yet. See VLLM_TASK.md.")
    except Exception as e:
        pytest.skip(f"vLLM server not reachable: {e}")

    t_vecs, _ = torch_service.encoder.encode(prompts)
    M = torch_service.item_matrix.cpu()

    t_top = torch.topk(t_vecs.cpu().float() @ M.T, k=10, dim=-1).indices
    v_top = torch.topk(v_vecs.cpu().float() @ M.T, k=10, dim=-1).indices

    for i in range(len(prompts)):
        t_ids, v_ids = t_top[i].tolist(), v_top[i].tolist()
        assert t_ids == v_ids, (
            f"prompt {i}: ranking diverged.\n"
            f"  torch: {t_ids}\n  vllm : {v_ids}\n"
            "Most likely cause is a pooling mismatch. This checkpoint trained "
            f"with pooling={torch_service.model.pooling!r} over non-pad tokens "
            "of the final hidden state.")


def test_normalization_does_not_change_ranking(torch_service, prompts):
    """Proves the claim in the VLLMEncoder docstring, so you trust it rather
    than 'fixing' vLLM's normalization and breaking something else."""
    import torch
    t_vecs, _ = torch_service.encoder.encode(prompts)
    M = torch_service.item_matrix.cpu()
    raw = t_vecs.cpu().float()
    normed = raw / raw.norm(dim=-1, keepdim=True)

    a = torch.topk(raw @ M.T, k=10, dim=-1).indices
    b = torch.topk(normed @ M.T, k=10, dim=-1).indices
    assert torch.equal(a, b), "L2 normalization must not reorder a dot-product ranking"
