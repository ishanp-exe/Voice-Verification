"""Speaker verification module.

Computes cosine similarity between speaker embedding vectors and evaluates
match decisions against a configurable threshold.

Important Security & Conceptual Note:
Cosine similarity measures the geometric angle between two high-dimensional vectors.
It is NOT a probability percentage of speaker identity. A threshold is an operational
parameter that must be empirically calibrated based on security and operational tolerance.
No specific threshold value should be assumed as universal or "typical" without
empirical calibration on target domain audio.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class VerificationResult:
    """Container holding verification outcome and interpretability metadata."""
    similarity: float
    threshold: float
    is_match: bool
    explanation: str


def compute_cosine_similarity(
    embedding_a: Union[torch.Tensor, np.ndarray],
    embedding_b: Union[torch.Tensor, np.ndarray],
) -> float:
    """Computes the cosine similarity between two 1D embedding vectors.

    Formula:
        cos(u, v) = (u . v) / (||u|| * ||v||)

    Args:
        embedding_a: First embedding vector (1D tensor or ndarray).
        embedding_b: Second embedding vector (1D tensor or ndarray).

    Returns:
        Float value bounded between -1.0 and 1.0.
    """
    if isinstance(embedding_a, np.ndarray):
        embedding_a = torch.from_numpy(embedding_a)
    if isinstance(embedding_b, np.ndarray):
        embedding_b = torch.from_numpy(embedding_b)

    # Flatten to 1D
    u = embedding_a.view(-1).float()
    v = embedding_b.view(-1).float()

    if u.shape[0] != v.shape[0]:
        raise ValueError(
            f"Embedding dimension mismatch: vector A has {u.shape[0]} dimensions, "
            f"while vector B has {v.shape[0]} dimensions."
        )

    # Compute dot product over product of L2 norms
    norm_u = torch.norm(u, p=2)
    norm_v = torch.norm(v, p=2)

    if norm_u == 0.0 or norm_v == 0.0:
        return 0.0

    similarity = torch.dot(u, v) / (norm_u * norm_v)
    # Clamp to [-1.0, 1.0] to safeguard against numerical floating-point inaccuracies
    clamped_sim = float(torch.clamp(similarity, -1.0, 1.0).item())
    return clamped_sim


def verify_speakers(
    enrolled_embedding: Union[torch.Tensor, np.ndarray],
    candidate_embedding: Union[torch.Tensor, np.ndarray],
    threshold: float,
) -> VerificationResult:
    """Evaluates whether two speaker embeddings match given an operational threshold.

    Args:
        enrolled_embedding: Baseline enrolled embedding representation.
        candidate_embedding: New verification candidate embedding representation.
        threshold: Configurable decision boundary.

    Returns:
        VerificationResult with similarity score, decision, and explanation.
    """
    similarity = compute_cosine_similarity(enrolled_embedding, candidate_embedding)
    is_match = similarity >= threshold

    explanation = (
        f"Cosine similarity score: {similarity:.4f} (Decision threshold: {threshold:.4f}). "
        f"Result: {'MATCH' if is_match else 'NO MATCH'}. "
        "Notice: Cosine similarity is a geometric metric indicating angular proximity in "
        "vector space (-1.0 to +1.0). It is NOT an identity probability or confidence percentage. "
        "The decision threshold represents an operational setting where scores at or above the "
        "threshold accept the speaker, and scores below reject them."
    )

    return VerificationResult(
        similarity=similarity,
        threshold=threshold,
        is_match=is_match,
        explanation=explanation,
    )
