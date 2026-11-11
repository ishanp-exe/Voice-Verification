"""REAL-MODEL tests: these load the actual pinned ECAPA-TDNN weights (no mocks).

They need the model files in speechbrain_cache/ or network access to download
them. Run only these with: pytest -m real_model
"""

from __future__ import annotations

import re

import pytest
import torch

from src.model import DEFAULT_MODEL_REVISION, SpeakerEmbeddingModel


def test_model_revision_is_exact_commit_hash():
    assert re.fullmatch(r"[0-9a-f]{40}", DEFAULT_MODEL_REVISION)


@pytest.fixture(scope="module")
def real_model() -> SpeakerEmbeddingModel:
    return SpeakerEmbeddingModel.load()


@pytest.mark.real_model
def test_real_model_embedding_shape_norm_and_determinism(real_model):
    torch.manual_seed(0)
    wave = torch.randn(1, 16000 * 2) * 0.1
    a = real_model.extract_embedding(wave)
    b = real_model.extract_embedding(wave)
    assert a.shape == (192,)
    assert torch.isclose(a.norm(), torch.tensor(1.0), atol=1e-4)
    assert torch.allclose(a, b, atol=1e-5)
