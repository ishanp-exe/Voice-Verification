"""Headless LibriSpeech benchmark with a disjoint calibration / test speaker split.

Runs the same protocol as the Streamlit Evaluation Mode (speaker split seed 42,
calibration trial seed 42, test trial seed 100, threshold chosen at the empirical
EER point of the calibration partition) and writes reproducible artifacts:

    <output-dir>/results.json            summary metrics, seeds, trial counts
    <output-dir>/trials.csv              every scored trial (file names only, no local paths)
    <output-dir>/score_distribution.png  genuine vs. imposter score histograms
    <output-dir>/det_curve.png           DET curve of the held-out test partition

Usage:
    python scripts/run_benchmark.py --dataset-dir "<path/to/LibriSpeech/test-clean>"
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics import (  # noqa: E402
    EvaluatedTrial,
    compute_metric_sweep,
    compute_metrics_at_threshold,
    find_empirical_eer,
)

CAL_SEED = 42
TEST_SEED = 100
SPLIT_SEED = 42


def summarize(cal_eval: List[EvaluatedTrial], test_eval: List[EvaluatedTrial]) -> dict:
    """Calibrates the threshold on the calibration trials and scores the test trials at it.

    The test-partition EER is reported for reference only; it is never used to pick
    the operating threshold.
    """
    cal_eer = find_empirical_eer(compute_metric_sweep(cal_eval, num_steps=400))
    if cal_eer is None:
        raise ValueError("Calibration partition needs both genuine and imposter trials.")
    threshold = cal_eer.threshold
    test_at_cal = compute_metrics_at_threshold(test_eval, threshold)
    test_eer = find_empirical_eer(compute_metric_sweep(test_eval, num_steps=400))

    return {
        "calibrated_threshold": threshold,
        "calibration_eer": (cal_eer.far + cal_eer.frr) / 2,
        "test_at_calibrated_threshold": asdict(test_at_cal),
        "test_eer_reference_only": (test_eer.far + test_eer.frr) / 2 if test_eer else None,
    }


def _plot_scores(test_eval: List[EvaluatedTrial], threshold: float, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    genuine = [t.similarity for t in test_eval if t.is_genuine]
    imposter = [t.similarity for t in test_eval if not t.is_genuine]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    bins = 40
    ax.hist(imposter, bins=bins, alpha=0.65, color="#d9534f", label=f"Imposter (n={len(imposter)})")
    ax.hist(genuine, bins=bins, alpha=0.65, color="#2b7bb9", label=f"Genuine (n={len(genuine)})")
    ax.axvline(threshold, color="#222222", linestyle="--", linewidth=1.5, label=f"τ_cal = {threshold:.3f}")
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Trials")
    ax.set_title("Held-out test partition: score distributions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_det(test_eval: List[EvaluatedTrial], threshold: float, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import norm

    sweep = compute_metric_sweep(test_eval, num_steps=1000)
    clip = 1e-4
    far = np.clip([m.far for m in sweep], clip, 1 - clip)
    frr = np.clip([m.frr for m in sweep], clip, 1 - clip)
    op = compute_metrics_at_threshold(test_eval, threshold)

    ticks = [0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    ax.plot(norm.ppf(far), norm.ppf(frr), color="#2b7bb9", linewidth=2, label="Test partition")
    ax.plot(
        norm.ppf(np.clip(op.far, clip, 1 - clip)),
        norm.ppf(np.clip(op.frr, clip, 1 - clip)),
        "o", color="#d9534f", markersize=8,
        label=f"τ_cal = {threshold:.3f} (FAR {op.far:.2%}, FRR {op.frr:.2%})",
    )
    lim = norm.ppf([0.0005, 0.5])
    ax.plot(lim, lim, color="#999999", linestyle=":", linewidth=1, label="FAR = FRR")
    ax.set_xticks(norm.ppf(ticks))
    ax.set_xticklabels([f"{t:.1%}" for t in ticks])
    ax.set_yticks(norm.ppf(ticks))
    ax.set_yticklabels([f"{t:.1%}" for t in ticks])
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("False Accept Rate")
    ax.set_ylabel("False Reject Rate")
    ax.set_title("DET curve (held-out test partition)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _write_trials_csv(rows: List[tuple], out_path: Path) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["partition", "file_a", "file_b", "speaker_a", "speaker_b", "is_genuine", "similarity"])
        for partition, t in rows:
            writer.writerow([
                partition,
                Path(t.path_a).name,
                Path(t.path_b).name,
                t.speaker_a,
                t.speaker_b,
                int(t.is_genuine),
                f"{t.similarity:.6f}",
            ])


def run(dataset_dir: Path, index_path: Optional[Path], output_dir: Path, num_genuine: int, num_imposter: int) -> dict:
    from src.audio import load_and_validate_audio
    from src.dataset_eval import LibriSpeechAdapter
    from src.model import SpeakerEmbeddingModel
    from src.verification import compute_cosine_similarity

    adapter = LibriSpeechAdapter(dataset_dir, valid_index_path=index_path)
    summary = adapter.validate()
    if not summary.is_valid:
        raise SystemExit(f"Dataset invalid: {summary.message}")
    print(f"{summary.message} {summary.num_speakers} speakers, {summary.num_recordings} recordings.")

    cal_spk, test_spk, reliable, note = adapter.split_speakers_for_calibration(0.5, seed=SPLIT_SEED)
    print(note)
    cal_trials = adapter.generate_trials(num_genuine, num_imposter, speaker_subset=cal_spk, seed=CAL_SEED)
    test_trials = adapter.generate_trials(num_genuine, num_imposter, speaker_subset=test_spk, seed=TEST_SEED)

    model = SpeakerEmbeddingModel.load(savedir=str(ROOT / "speechbrain_cache" / "spkrec-ecapa-voxceleb"))
    cache: Dict[str, object] = {}
    skipped: List[dict] = []

    def embed(p: Path):
        key = str(p.resolve())
        if key not in cache:
            cache[key] = model.extract_embedding(load_and_validate_audio(p, min_duration_sec=0.5).waveform)
        return cache[key]

    def score(trials, label):
        out: List[EvaluatedTrial] = []
        for i, t in enumerate(trials, 1):
            try:
                sim = compute_cosine_similarity(embed(t.path_a), embed(t.path_b))
                out.append(EvaluatedTrial(str(t.path_a), str(t.path_b), t.speaker_a, t.speaker_b, t.is_genuine, sim))
            except Exception as ex:  # report, never silently drop
                skipped.append({"partition": label, "file_a": t.path_a.name, "file_b": t.path_b.name, "reason": str(ex)})
            if i % 100 == 0 or i == len(trials):
                print(f"  [{label}] {i}/{len(trials)} trials scored ({len(cache)} embeddings cached)")
        return out

    start = time.time()
    cal_eval = score(cal_trials, "calibration")
    test_eval = score(test_trials, "test")
    elapsed = time.time() - start

    results = summarize(cal_eval, test_eval)
    results.update({
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": "LibriSpeech test-clean",
        "used_validated_index": summary.used_validated_index,
        "model": model.model_name,
        "split_reliable": reliable,
        "split_note": note,
        "calibration_speakers": len(cal_spk),
        "test_speakers": len(test_spk),
        "seeds": {"split": SPLIT_SEED, "calibration_trials": CAL_SEED, "test_trials": TEST_SEED},
        "requested_trials_per_partition": {"genuine": num_genuine, "imposter": num_imposter},
        "calibration_trials_scored": len(cal_eval),
        "test_trials_scored": len(test_eval),
        "skipped_trials": skipped,
        "unique_recordings_embedded": len(cache),
        "elapsed_sec": round(elapsed, 1),
    })

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    _write_trials_csv([("calibration", t) for t in cal_eval] + [("test", t) for t in test_eval], output_dir / "trials.csv")
    _plot_scores(test_eval, results["calibrated_threshold"], output_dir / "score_distribution.png")
    _plot_det(test_eval, results["calibrated_threshold"], output_dir / "det_curve.png")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", required=True, type=Path, help="LibriSpeech test-clean directory")
    parser.add_argument("--index", type=Path, default=ROOT / "data" / "evaluation" / "valid_audio_index.csv",
                        help="Validated index CSV (used when present)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs" / "results")
    parser.add_argument("--genuine", type=int, default=500, help="Genuine trials per partition")
    parser.add_argument("--imposter", type=int, default=500, help="Imposter trials per partition")
    args = parser.parse_args()

    index = args.index if args.index and args.index.is_file() else None
    r = run(args.dataset_dir, index, args.output_dir, args.genuine, args.imposter)
    t = r["test_at_calibrated_threshold"]
    print(
        f"\nτ_cal = {r['calibrated_threshold']:.4f} (calibration EER {r['calibration_eer']:.2%})\n"
        f"Test @ τ_cal: FAR {t['far']:.2%}  FRR {t['frr']:.2%}  accuracy {t['accuracy']:.2%}  "
        f"({t['genuine_trials']} genuine / {t['imposter_trials']} imposter)\n"
        f"Test EER (reference only): {r['test_eer_reference_only']:.2%}\n"
        f"Skipped trials: {len(r['skipped_trials'])}\nArtifacts written to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
