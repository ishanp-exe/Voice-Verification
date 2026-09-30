"""LibriSpeech Dataset Cleaning and Validation Script.

Inspects an external LibriSpeech test-clean directory without modifying, renaming,
moving, or deleting any files.

For each audio file:
- Attempts audio decoding via soundfile.
- Extracts speaker ID, duration, sample rate, and channel count.
- Computes SHA-256 hash to reliably detect duplicates.
- Validates against minimum duration (0.8s), non-empty content, directory/filename speaker labeling consistency.
- Records validation status ('VALID' vs 'REJECTED') with explicit rejection reasons.

Outputs (saved outside the dataset folder):
- audio_file_validation.csv: Full audit log of all inspected audio files.
- valid_audio_index.csv: Filtered index of files that passed all checks.
- dataset_cleaning_summary.txt: Formatted human-readable summary of cleaning metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import soundfile as sf


def compute_sha256(file_path: Path, block_size: int = 65536) -> str:
    """Computes SHA-256 checksum of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class AudioRecord:
    def __init__(
        self,
        file_path: Path,
        speaker_id: str,
        status: str,
        duration_sec: float,
        sample_rate: int,
        num_channels: int,
        rejection_reason: str,
        sha256_hash: str,
    ):
        self.file_path = str(file_path.resolve())
        self.speaker_id = speaker_id
        self.status = status
        self.duration_sec = round(duration_sec, 4)
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.rejection_reason = rejection_reason
        self.sha256 = sha256_hash


def validate_dataset(
    dataset_dir: Path,
    output_dir: Path,
    min_duration_sec: float = 0.8,
) -> Tuple[List[AudioRecord], List[AudioRecord], Dict[str, int]]:
    """Runs repeatable validation across all audio files in the dataset folder."""
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist or is not a directory: {dataset_dir}")

    # Discover candidate audio files (FLAC, WAV, MP3, OGG)
    audio_extensions = {".flac", ".wav", ".mp3", ".ogg"}
    audio_files: List[Path] = []
    for p in sorted(dataset_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in audio_extensions:
            audio_files.append(p)

    all_records: List[AudioRecord] = []
    valid_records: List[AudioRecord] = []
    seen_hashes: Dict[str, str] = {}  # sha256 -> original_file_path
    rejection_counts: Dict[str, int] = {
        "Empty file": 0,
        "Unreadable / Corrupt": 0,
        "Too short (< 0.8s)": 0,
        "Mislabeled speaker": 0,
        "Duplicate content": 0,
    }

    for file_path in audio_files:
        # Determine speaker ID from directory structure: {speaker_id}/{chapter_id}/...
        rel_parts = file_path.relative_to(dataset_dir).parts
        dir_speaker_id = rel_parts[0] if len(rel_parts) >= 2 else "unknown"

        # Check filename speaker prefix (LibriSpeech format: {speaker_id}-{chapter_id}-{utterance}.flac)
        stem_parts = file_path.stem.split("-")
        filename_speaker_id = stem_parts[0] if len(stem_parts) >= 2 else ""

        is_mislabeled = False
        if filename_speaker_id and dir_speaker_id != "unknown" and filename_speaker_id != dir_speaker_id:
            is_mislabeled = True

        # Check empty file
        if file_path.stat().st_size == 0:
            rejection_counts["Empty file"] += 1
            rec = AudioRecord(
                file_path=file_path,
                speaker_id=dir_speaker_id,
                status="REJECTED",
                duration_sec=0.0,
                sample_rate=0,
                num_channels=0,
                rejection_reason="Empty file (0 bytes)",
                sha256_hash="",
            )
            all_records.append(rec)
            continue

        # Compute hash
        try:
            file_hash = compute_sha256(file_path)
        except Exception as e:
            rejection_counts["Unreadable / Corrupt"] += 1
            rec = AudioRecord(
                file_path=file_path,
                speaker_id=dir_speaker_id,
                status="REJECTED",
                duration_sec=0.0,
                sample_rate=0,
                num_channels=0,
                rejection_reason=f"Failed to read file hash: {e}",
                sha256_hash="",
            )
            all_records.append(rec)
            continue

        # Check duplicate
        if file_hash in seen_hashes:
            rejection_counts["Duplicate content"] += 1
            orig_path = seen_hashes[file_hash]
            rec = AudioRecord(
                file_path=file_path,
                speaker_id=dir_speaker_id,
                status="REJECTED",
                duration_sec=0.0,
                sample_rate=0,
                num_channels=0,
                rejection_reason=f"Duplicate content of {orig_path}",
                sha256_hash=file_hash,
            )
            all_records.append(rec)
            continue

        # Attempt decoding with soundfile
        try:
            info = sf.info(str(file_path))
            duration = info.duration
            sr = info.samplerate
            channels = info.channels
        except Exception as e:
            rejection_counts["Unreadable / Corrupt"] += 1
            rec = AudioRecord(
                file_path=file_path,
                speaker_id=dir_speaker_id,
                status="REJECTED",
                duration_sec=0.0,
                sample_rate=0,
                num_channels=0,
                rejection_reason=f"Audio decoding failed: {e}",
                sha256_hash=file_hash,
            )
            all_records.append(rec)
            continue

        # Check mislabeled
        if is_mislabeled:
            rejection_counts["Mislabeled speaker"] += 1
            rec = AudioRecord(
                file_path=file_path,
                speaker_id=dir_speaker_id,
                status="REJECTED",
                duration_sec=duration,
                sample_rate=sr,
                num_channels=channels,
                rejection_reason=f"Mislabeled speaker: directory '{dir_speaker_id}' != filename '{filename_speaker_id}'",
                sha256_hash=file_hash,
            )
            all_records.append(rec)
            continue

        # Check too short
        if duration < min_duration_sec:
            rejection_counts["Too short (< 0.8s)"] += 1
            rec = AudioRecord(
                file_path=file_path,
                speaker_id=dir_speaker_id,
                status="REJECTED",
                duration_sec=duration,
                sample_rate=sr,
                num_channels=channels,
                rejection_reason=f"Duration too short ({duration:.2f}s < {min_duration_sec:.2f}s)",
                sha256_hash=file_hash,
            )
            all_records.append(rec)
            continue

        # All checks passed
        seen_hashes[file_hash] = str(file_path)
        rec = AudioRecord(
            file_path=file_path,
            speaker_id=dir_speaker_id,
            status="VALID",
            duration_sec=duration,
            sample_rate=sr,
            num_channels=channels,
            rejection_reason="",
            sha256_hash=file_hash,
        )
        all_records.append(rec)
        valid_records.append(rec)

    # Save outputs
    validation_csv_path = output_dir / "audio_file_validation.csv"
    with open(validation_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "file_path",
            "speaker_id",
            "status",
            "duration_sec",
            "sample_rate",
            "num_channels",
            "rejection_reason",
            "sha256",
        ])
        for r in all_records:
            writer.writerow([
                r.file_path,
                r.speaker_id,
                r.status,
                f"{r.duration_sec:.4f}",
                r.sample_rate,
                r.num_channels,
                r.rejection_reason,
                r.sha256,
            ])

    valid_csv_path = output_dir / "valid_audio_index.csv"
    with open(valid_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "file_path",
            "speaker_id",
            "duration_sec",
            "sample_rate",
            "num_channels",
            "sha256",
        ])
        for r in valid_records:
            writer.writerow([
                r.file_path,
                r.speaker_id,
                f"{r.duration_sec:.4f}",
                r.sample_rate,
                r.num_channels,
                r.sha256,
            ])

    # Summary statistics
    total_files = len(all_records)
    valid_count = len(valid_records)
    rejected_count = total_files - valid_count
    all_speakers = {r.speaker_id for r in all_records}
    valid_speakers = {r.speaker_id for r in valid_records}
    total_valid_duration = sum(r.duration_sec for r in valid_records)
    durations = [r.duration_sec for r in valid_records] if valid_records else [0.0]

    summary_txt_path = output_dir / "dataset_cleaning_summary.txt"
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write("==============================================================================\n")
        f.write("             LIBRISPEECH DATASET CLEANING & VALIDATION SUMMARY                \n")
        f.write("==============================================================================\n\n")
        f.write(f"Validation Timestamp (UTC): {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"Dataset Directory Inspected: {dataset_dir.resolve()}\n")
        f.write(f"Output Directory:            {output_dir.resolve()}\n\n")
        f.write("------------------------------------------------------------------------------\n")
        f.write("FILE & SPEAKER TOTALS\n")
        f.write("------------------------------------------------------------------------------\n")
        f.write(f"Total Audio Files Checked:   {total_files:,}\n")
        f.write(f"Valid Audio Files:           {valid_count:,} ({(valid_count/total_files*100):.2f}%)\n" if total_files > 0 else "0\n")
        f.write(f"Rejected Audio Files:        {rejected_count:,} ({(rejected_count/total_files*100):.2f}%)\n" if total_files > 0 else "0\n")
        f.write(f"Total Unique Speakers Found: {len(all_speakers)}\n")
        f.write(f"Speakers with Valid Audio:   {len(valid_speakers)}\n\n")
        f.write("------------------------------------------------------------------------------\n")
        f.write("REJECTION BREAKDOWN BY REASON\n")
        f.write("------------------------------------------------------------------------------\n")
        for reason, count in rejection_counts.items():
            f.write(f"- {reason:<28}: {count:,}\n")
        f.write("\n------------------------------------------------------------------------------\n")
        f.write("ACOUSTIC PROPERTIES & DURATION (VALID SUBSET)\n")
        f.write("------------------------------------------------------------------------------\n")
        f.write(f"Total Valid Speech Duration: {total_valid_duration:.2f} seconds ({total_valid_duration/3600:.2f} hours)\n")
        f.write(f"Min Duration:                {min(durations):.2f}s\n")
        f.write(f"Max Duration:                {max(durations):.2f}s\n")
        f.write(f"Mean Duration:               {sum(durations)/len(durations):.2f}s\n")
        sample_rates = {r.sample_rate for r in valid_records}
        channels = {r.num_channels for r in valid_records}
        f.write(f"Sample Rates Detected:       {sorted(list(sample_rates))}\n")
        f.write(f"Channel Counts Detected:     {sorted(list(channels))}\n\n")
        f.write("------------------------------------------------------------------------------\n")
        f.write("RECORDINGS PER SPEAKER (VALID SUBSET)\n")
        f.write("------------------------------------------------------------------------------\n")
        speaker_counts: Dict[str, int] = {}
        for r in valid_records:
            speaker_counts[r.speaker_id] = speaker_counts.get(r.speaker_id, 0) + 1
        for spk_id, cnt in sorted(speaker_counts.items()):
            f.write(f"Speaker {spk_id:<8}: {cnt:>4} recordings\n")
        f.write("==============================================================================\n")

    return all_records, valid_records, rejection_counts


def main():
    parser = argparse.ArgumentParser(description="Clean and validate LibriSpeech test-clean corpus.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=r"C:\Users\ishan\Downloads\LibriSpeech\test-clean",
        help="Path to LibriSpeech test-clean directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=r"data\evaluation",
        help="Directory to save CSV outputs and summary.",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.8,
        help="Minimum allowed audio duration in seconds (default: 0.8).",
    )
    args = parser.parse_args()

    print(f"Starting dataset validation for: {args.dataset_dir}")
    print(f"Output directory: {args.output_dir}")
    all_recs, valid_recs, rej_counts = validate_dataset(
        dataset_dir=Path(args.dataset_dir),
        output_dir=Path(args.output_dir),
        min_duration_sec=args.min_duration,
    )

    print("\n--- VALIDATION COMPLETE ---")
    print(f"Total audio files inspected: {len(all_recs)}")
    print(f"Valid files: {len(valid_recs)}")
    print(f"Rejected files: {len(all_recs) - len(valid_recs)}")
    for reason, count in rej_counts.items():
        if count > 0:
            print(f"  - {reason}: {count}")


if __name__ == "__main__":
    main()
