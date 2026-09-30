"""Dataset evaluation harness and trial generator.

Provides dataset adapters for external evaluation datasets (such as LibriSpeech test-clean).
Generates genuine (same-speaker) and imposter (different-speaker) trial pairs for
biometric verification benchmarking.

Domain Dataset Note:
A domain-specific evaluation dataset can be integrated in future phases. To ensure modularity,
all dataset interactions follow the BaseSpeakerDatasetAdapter interface. A specialized
adapter can be added once its format, labels, licensing, and handling requirements
are provided. No assumptions about target domain datasets are made in this module.
"""

from __future__ import annotations

import abc
import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TrialPair:
    """Represents a pair of audio recordings for verification benchmarking."""
    path_a: Path
    path_b: Path
    speaker_a: str
    speaker_b: str
    is_genuine: bool  # True if same speaker (target), False if different speakers (non-target)


@dataclass
class DatasetSummary:
    """Summary of discovered dataset structure."""
    is_valid: bool
    message: str
    num_speakers: int
    num_recordings: int
    sample_speakers: List[str]


class BaseSpeakerDatasetAdapter(abc.ABC):
    """Abstract interface for external evaluation datasets."""

    @abc.abstractmethod
    def validate(self) -> DatasetSummary:
        """Inspects the dataset directory and returns a validation summary."""
        pass

    @abc.abstractmethod
    def get_speaker_audio_map(self) -> Dict[str, List[Path]]:
        """Returns a mapping of speaker_id -> list of audio file paths."""
        pass


class LibriSpeechAdapter(BaseSpeakerDatasetAdapter):
    """Adapter for external LibriSpeech datasets (e.g. test-clean).

    LibriSpeech audio files are typically arranged as:
        <root>/<speaker_id>/<chapter_id>/<speaker_id>-<chapter_id>-<utterance_id>.flac
    or with an intermediate split directory:
        <root>/test-clean/<speaker_id>/...

    Important Context:
    LibriSpeech contains clean audiobook narrations and does NOT represent
    banking telephone audio (which has 8 kHz band-limiting, codec artifacts, and line noise).
    """

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir) if root_dir else None
        self._speaker_map: Optional[Dict[str, List[Path]]] = None

    def validate(self) -> DatasetSummary:
        """Validates that the directory exists and contains valid LibriSpeech audio."""
        if not self.root_dir or str(self.root_dir).strip() == "":
            return DatasetSummary(
                is_valid=False,
                message="No LibriSpeech dataset path has been provided.",
                num_speakers=0,
                num_recordings=0,
                sample_speakers=[],
            )

        if not self.root_dir.exists():
            return DatasetSummary(
                is_valid=False,
                message=f"Provided directory does not exist: {self.root_dir}",
                num_speakers=0,
                num_recordings=0,
                sample_speakers=[],
            )

        if not self.root_dir.is_dir():
            return DatasetSummary(
                is_valid=False,
                message=f"Provided path is a file, not a directory: {self.root_dir}",
                num_speakers=0,
                num_recordings=0,
                sample_speakers=[],
            )

        speaker_map = self.get_speaker_audio_map()
        num_speakers = len(speaker_map)
        num_recordings = sum(len(paths) for paths in speaker_map.values())

        if num_speakers == 0 or num_recordings == 0:
            return DatasetSummary(
                is_valid=False,
                message=(
                    f"No valid .flac or .wav audio files found in '{self.root_dir}'. "
                    "Ensure you point to an extracted LibriSpeech directory containing speaker folders "
                    "(e.g., LibriSpeech/test-clean/)."
                ),
                num_speakers=0,
                num_recordings=0,
                sample_speakers=[],
            )

        return DatasetSummary(
            is_valid=True,
            message="Valid LibriSpeech directory detected.",
            num_speakers=num_speakers,
            num_recordings=num_recordings,
            sample_speakers=sorted(list(speaker_map.keys()))[:5],
        )

    def get_speaker_audio_map(self) -> Dict[str, List[Path]]:
        """Scans the directory for speaker audio files, caching the result."""
        if self._speaker_map is not None:
            return self._speaker_map

        if not self.root_dir or not self.root_dir.is_dir():
            return {}

        speaker_map: Dict[str, List[Path]] = defaultdict(list)

        # Look for .flac and .wav files
        audio_extensions = {".flac", ".wav"}
        for path in self.root_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in audio_extensions:
                # In LibriSpeech, filename is <speaker_id>-<chapter_id>-<utterance_id>.ext
                filename_parts = path.stem.split("-")
                if len(filename_parts) >= 3 and filename_parts[0].isdigit():
                    speaker_id = filename_parts[0]
                else:
                    # Fallback to parent directory hierarchy if filename is not hyphenated
                    # Typically path.parent.parent.name is speaker_id
                    speaker_id = path.parent.parent.name
                    if not speaker_id.isdigit():
                        speaker_id = path.parent.name

                speaker_map[speaker_id].append(path)

        self._speaker_map = dict(speaker_map)
        return self._speaker_map

    def generate_trials(
        self,
        num_genuine: int = 50,
        num_imposter: int = 50,
        speaker_subset: Optional[Set[str]] = None,
        seed: int = 42,
    ) -> List[TrialPair]:
        """Generates target (genuine) and non-target (imposter) trial pairs.

        Args:
            num_genuine: Maximum number of genuine (same-speaker) trial pairs.
            num_imposter: Maximum number of imposter (different-speaker) trial pairs.
            speaker_subset: Optional subset of speaker IDs to draw from.
            seed: Random seed for reproducible trial selection.

        Returns:
            List of TrialPair objects.
        """
        rng = random.Random(seed)
        speaker_map = self.get_speaker_audio_map()

        if speaker_subset is not None:
            speaker_map = {spk: paths for spk, paths in speaker_map.items() if spk in speaker_subset}

        # Filter out speakers with fewer than 2 recordings (cannot form genuine pair)
        eligible_genuine_speakers = [
            spk for spk, paths in speaker_map.items() if len(paths) >= 2
        ]
        all_speakers = list(speaker_map.keys())

        trials: List[TrialPair] = []

        # 1. Genuine pairs (same speaker, different recordings)
        genuine_count = 0
        max_genuine_attempts = num_genuine * 10
        attempts = 0
        seen_genuine_pairs: Set[Tuple[str, str]] = set()

        while genuine_count < num_genuine and attempts < max_genuine_attempts and eligible_genuine_speakers:
            attempts += 1
            spk = rng.choice(eligible_genuine_speakers)
            paths = speaker_map[spk]
            f_a, f_b = rng.sample(paths, 2)
            pair_key = tuple(sorted([str(f_a), str(f_b)]))
            if pair_key in seen_genuine_pairs:
                continue
            seen_genuine_pairs.add(pair_key)
            trials.append(
                TrialPair(
                    path_a=f_a,
                    path_b=f_b,
                    speaker_a=spk,
                    speaker_b=spk,
                    is_genuine=True,
                )
            )
            genuine_count += 1

        # 2. Imposter pairs (different speakers)
        imposter_count = 0
        max_imposter_attempts = num_imposter * 10
        attempts = 0
        seen_imposter_pairs: Set[Tuple[str, str]] = set()

        if len(all_speakers) >= 2:
            while imposter_count < num_imposter and attempts < max_imposter_attempts:
                attempts += 1
                spk_a, spk_b = rng.sample(all_speakers, 2)
                f_a = rng.choice(speaker_map[spk_a])
                f_b = rng.choice(speaker_map[spk_b])
                pair_key = tuple(sorted([str(f_a), str(f_b)]))
                if pair_key in seen_imposter_pairs:
                    continue
                seen_imposter_pairs.add(pair_key)
                trials.append(
                    TrialPair(
                        path_a=f_a,
                        path_b=f_b,
                        speaker_a=spk_a,
                        speaker_b=spk_b,
                        is_genuine=False,
                    )
                )
                imposter_count += 1

        rng.shuffle(trials)
        return trials

    def split_speakers_for_calibration(
        self,
        calibration_fraction: float = 0.5,
        min_speakers_required: int = 6,
        seed: int = 42,
    ) -> Tuple[Set[str], Set[str], bool, str]:
        """Partitions speakers into separate calibration (dev) and evaluation (test) subsets.

        Ensures that threshold selection (calibration) and reporting of final metrics
        (evaluation) occur on disjoint speaker partitions when feasible, balancing
        multi-utterance speakers between subsets so both can form genuine pairs.

        Returns:
            Tuple of (calibration_speakers, evaluation_speakers, is_split_reliable, note)
        """
        rng = random.Random(seed)
        speaker_map = self.get_speaker_audio_map()
        all_speakers = sorted(list(speaker_map.keys()))
        num_speakers = len(all_speakers)

        eligible_genuine = [spk for spk, paths in speaker_map.items() if len(paths) >= 2]

        if num_speakers < min_speakers_required or len(eligible_genuine) < 4:
            note = (
                f"Dataset contains {num_speakers} speaker(s) ({len(eligible_genuine)} with >= 2 recordings). "
                f"At least {min_speakers_required} total speakers and 4 multi-utterance speakers are recommended "
                "for a disjoint calibration/evaluation split. A statistically reliable split cannot be performed; "
                "the evaluation will run as a Development Preview across available data."
            )
            return set(all_speakers), set(all_speakers), False, note

        # Balance multi-utterance speakers between calibration and evaluation sets
        shuffled_multi = eligible_genuine.copy()
        rng.shuffle(shuffled_multi)
        multi_split = max(2, int(len(shuffled_multi) * calibration_fraction))
        cal_multi = set(shuffled_multi[:multi_split])
        eval_multi = set(shuffled_multi[multi_split:])

        single_speakers = [spk for spk in all_speakers if spk not in set(eligible_genuine)]
        shuffled_single = single_speakers.copy()
        rng.shuffle(shuffled_single)
        single_split = int(len(shuffled_single) * calibration_fraction)
        cal_single = set(shuffled_single[:single_split])
        eval_single = set(shuffled_single[single_split:])

        cal_set = cal_multi | cal_single
        eval_set = eval_multi | eval_single

        note = (
            f"Successfully partitioned {num_speakers} speakers into {len(cal_set)} calibration speakers "
            f"({len(cal_multi)} multi-sample) and {len(eval_set)} evaluation speakers "
            f"({len(eval_multi)} multi-sample)."
        )
        return cal_set, eval_set, True, note
