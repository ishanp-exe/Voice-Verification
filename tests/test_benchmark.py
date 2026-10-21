"""Tests for the headless benchmark's calibration and reporting logic."""

from __future__ import annotations

import random

import pytest

from scripts.run_benchmark import _plot_det, _plot_scores, _write_trials_csv, summarize
from src.metrics import EvaluatedTrial


def _trials(n: int, seed: int, genuine_mean: float = 0.6, imposter_mean: float = 0.1):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        out.append(EvaluatedTrial(f"C:/private/a{i}.flac", f"C:/private/b{i}.flac", "1", "1", True,
                                  rng.gauss(genuine_mean, 0.1)))
        out.append(EvaluatedTrial(f"C:/private/c{i}.flac", f"C:/private/d{i}.flac", "1", "2", False,
                                  rng.gauss(imposter_mean, 0.1)))
    return out


def test_threshold_comes_from_calibration_only():
    cal = _trials(200, seed=1)
    # Shift the test partition: a test-tuned threshold would move, the calibrated one must not.
    test = _trials(200, seed=2, genuine_mean=0.8, imposter_mean=0.3)
    r = summarize(cal, test)
    assert 0.2 < r["calibrated_threshold"] < 0.5
    assert r["calibrated_threshold"] == summarize(cal, _trials(200, seed=3))["calibrated_threshold"]
    t = r["test_at_calibrated_threshold"]
    assert t["genuine_trials"] == 200 and t["imposter_trials"] == 200
    assert t["threshold"] == r["calibrated_threshold"]


def test_summarize_requires_both_classes():
    only_genuine = [t for t in _trials(20, seed=1) if t.is_genuine]
    with pytest.raises(ValueError):
        summarize(only_genuine, _trials(20, seed=2))


def test_artifacts_written_without_local_paths(tmp_path):
    trials = _trials(50, seed=4)
    _write_trials_csv([("test", t) for t in trials], tmp_path / "trials.csv")
    text = (tmp_path / "trials.csv").read_text(encoding="utf-8")
    assert "private" not in text and "a0.flac" in text

    _plot_scores(trials, 0.35, tmp_path / "scores.png")
    _plot_det(trials, 0.35, tmp_path / "det.png")
    assert (tmp_path / "scores.png").stat().st_size > 0
    assert (tmp_path / "det.png").stat().st_size > 0
