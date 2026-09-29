"""Evaluation metrics calculation module.

Calculates biometric verification performance metrics (FAR, FRR, Accuracy, EER)
strictly from empirical trial results.

Important Educational & Scientific Note:
All metrics computed in this module are derived entirely from actual, evaluated
trial scores. No placeholder, simulated, or fabricated performance numbers are produced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class EvaluatedTrial:
    """Represents a trial pair along with its empirically computed similarity score."""
    path_a: str
    path_b: str
    speaker_a: str
    speaker_b: str
    is_genuine: bool
    similarity: float


@dataclass
class ThresholdMetric:
    """Biometric performance metrics at a specific operational threshold."""
    threshold: float
    total_trials: int
    genuine_trials: int
    imposter_trials: int
    false_accepts: int
    false_rejects: int
    true_accepts: int
    true_rejects: int
    far: float  # False Accept Rate: FA / total_imposter
    frr: float  # False Reject Rate: FR / total_genuine
    accuracy: float  # (TA + TR) / total_trials


def compute_metrics_at_threshold(
    trials: List[EvaluatedTrial], threshold: float
) -> ThresholdMetric:
    """Computes FAR, FRR, and Accuracy for a given set of evaluated trials at a threshold.

    Definitions:
        False Accept (FA): Imposter pair (different speakers) with similarity >= threshold.
        False Reject (FR): Genuine pair (same speaker) with similarity < threshold.
        True Accept (TA): Genuine pair with similarity >= threshold.
        True Reject (TR): Imposter pair with similarity < threshold.

    Args:
        trials: List of evaluated trials containing ground truth and similarity score.
        threshold: The decision threshold to evaluate.

    Returns:
        ThresholdMetric dataclass containing empirical counts and rates.
    """
    total = len(trials)
    if total == 0:
        return ThresholdMetric(
            threshold=threshold,
            total_trials=0,
            genuine_trials=0,
            imposter_trials=0,
            false_accepts=0,
            false_rejects=0,
            true_accepts=0,
            true_rejects=0,
            far=0.0,
            frr=0.0,
            accuracy=0.0,
        )

    genuine_count = 0
    imposter_count = 0
    false_accepts = 0
    false_rejects = 0
    true_accepts = 0
    true_rejects = 0

    for t in trials:
        is_match_decision = t.similarity >= threshold
        if t.is_genuine:
            genuine_count += 1
            if is_match_decision:
                true_accepts += 1
            else:
                false_rejects += 1
        else:
            imposter_count += 1
            if is_match_decision:
                false_accepts += 1
            else:
                true_rejects += 1

    far = (false_accepts / imposter_count) if imposter_count > 0 else 0.0
    frr = (false_rejects / genuine_count) if genuine_count > 0 else 0.0
    accuracy = (true_accepts + true_rejects) / total if total > 0 else 0.0

    return ThresholdMetric(
        threshold=round(threshold, 4),
        total_trials=total,
        genuine_trials=genuine_count,
        imposter_trials=imposter_count,
        false_accepts=false_accepts,
        false_rejects=false_rejects,
        true_accepts=true_accepts,
        true_rejects=true_rejects,
        far=far,
        frr=frr,
        accuracy=accuracy,
    )


def compute_metric_sweep(
    trials: List[EvaluatedTrial],
    thresholds: Optional[List[float]] = None,
    num_steps: int = 50,
) -> List[ThresholdMetric]:
    """Sweeps over multiple candidate thresholds to evaluate trade-offs between FAR and FRR.

    Args:
        trials: List of evaluated trials.
        thresholds: Explicit list of thresholds, or None to generate uniform steps.
        num_steps: Number of threshold steps to evaluate if thresholds is None.

    Returns:
        List of ThresholdMetric entries, sorted by increasing threshold.
    """
    if not trials:
        return []

    if thresholds is None:
        scores = [t.similarity for t in trials]
        min_score = max(-1.0, min(scores) - 0.05)
        max_score = min(1.0, max(scores) + 0.05)
        thresholds = list(np.linspace(min_score, max_score, num_steps))

    metrics = [compute_metrics_at_threshold(trials, t) for t in thresholds]
    return metrics


def find_empirical_eer(metric_sweep: List[ThresholdMetric]) -> Optional[ThresholdMetric]:
    """Finds the operating point in the metric sweep where |FAR - FRR| is minimized.

    This represents the empirical Equal Error Rate (EER) threshold on the tested trials.

    Returns:
        The ThresholdMetric closest to FAR == FRR, or None if the sweep is empty.
    """
    if not metric_sweep:
        return None

    # Filter only points that have both genuine and imposter trials
    valid_points = [m for m in metric_sweep if m.genuine_trials > 0 and m.imposter_trials > 0]
    if not valid_points:
        return None

    best_metric = min(valid_points, key=lambda m: abs(m.far - m.frr))
    return best_metric
