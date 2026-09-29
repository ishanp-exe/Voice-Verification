"""Local storage and privacy management for volunteer enrollment profiles.

Manages saving and deleting volunteer embeddings and minimal non-PII metadata.
Strict Privacy Guarantees:
- Data is stored purely locally within the project's data/volunteers/ directory (gitignored).
- Raw audio files are discarded by default after embedding extraction to prevent
  unnecessary biometric audio retention.
- Deletion permanently removes all files associated with a volunteer ID from disk.
"""

from __future__ import annotations

import json
import logging
import random
import shutil
import string
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

DEFAULT_STORAGE_DIR = Path("data/volunteers")


@dataclass
class VolunteerMetadata:
    """Minimal non-personally identifiable metadata associated with an enrollment."""
    demo_id: str
    enrolled_at_utc: str
    audio_duration_sec: float
    original_sample_rate: int
    has_raw_audio: bool
    model_name: str


class VolunteerStorage:
    """Manages local storage, retrieval, and complete deletion of volunteer profiles."""

    def __init__(self, base_dir: Path = DEFAULT_STORAGE_DIR):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def generate_demo_id(prefix: str = "VOL") -> str:
        """Generates a random, non-identifying demo ID (e.g. VOL-7392)."""
        suffix = "".join(random.choices(string.digits, k=4))
        return f"{prefix}-{suffix}"

    def _get_volunteer_dir(self, demo_id: str) -> Path:
        # Sanitize demo_id to prevent directory traversal and accidental base_dir deletion
        if not demo_id or not isinstance(demo_id, str):
            raise ValueError(f"Invalid demo ID: '{demo_id}'. Demo ID must be a non-empty string.")
        sanitized_id = "".join(c for c in demo_id.strip() if c.isalnum() or c in ("-", "_"))
        if not sanitized_id:
            raise ValueError(f"Invalid demo ID: '{demo_id}'. ID must contain alphanumeric characters.")
        return self.base_dir / sanitized_id

    def list_volunteers(self) -> List[VolunteerMetadata]:
        """Lists all currently enrolled volunteers."""
        volunteers: List[VolunteerMetadata] = []
        if not self.base_dir.exists():
            return volunteers

        for folder in self.base_dir.iterdir():
            if folder.is_dir():
                meta_path = folder / "metadata.json"
                if meta_path.exists():
                    try:
                        with open(meta_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        volunteers.append(VolunteerMetadata(**data))
                    except Exception as e:
                        logger.warning("Failed to load metadata for %s: %e", folder.name, e)
        return volunteers

    def save_volunteer(
        self,
        demo_id: str,
        embedding: torch.Tensor,
        duration_sec: float,
        original_sample_rate: int,
        model_name: str,
        raw_audio_bytes: Optional[bytes] = None,
        save_raw_audio: bool = False,
    ) -> VolunteerMetadata:
        """Saves volunteer embedding vector and metadata.

        Args:
            demo_id: Anonymized volunteer ID.
            embedding: 1D normalized speaker embedding tensor.
            duration_sec: Duration of the enrollment audio.
            original_sample_rate: Sample rate of original recording.
            model_name: Model used to extract embedding.
            raw_audio_bytes: Optional audio bytes if explicitly retained.
            save_raw_audio: If True, saves raw audio sample; defaults to False for privacy.

        Returns:
            Saved VolunteerMetadata instance.
        """
        vol_dir = self._get_volunteer_dir(demo_id)
        vol_dir.mkdir(parents=True, exist_ok=True)

        # Save embedding tensor
        embedding_path = vol_dir / "embedding.pt"
        torch.save(embedding.detach().cpu(), embedding_path)

        has_raw = False
        if save_raw_audio and raw_audio_bytes:
            audio_path = vol_dir / "enrollment_sample.wav"
            with open(audio_path, "wb") as f:
                f.write(raw_audio_bytes)
            has_raw = True

        metadata = VolunteerMetadata(
            demo_id=demo_id,
            enrolled_at_utc=datetime.now(timezone.utc).isoformat(),
            audio_duration_sec=duration_sec,
            original_sample_rate=original_sample_rate,
            has_raw_audio=has_raw,
            model_name=model_name,
        )

        meta_path = vol_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(asdict(metadata), f, indent=2)

        return metadata

    def load_embedding(self, demo_id: str) -> Optional[torch.Tensor]:
        """Loads the stored embedding tensor for a given volunteer ID."""
        vol_dir = self._get_volunteer_dir(demo_id)
        embedding_path = vol_dir / "embedding.pt"
        if not embedding_path.exists():
            return None
        return torch.load(embedding_path)

    def load_metadata(self, demo_id: str) -> Optional[VolunteerMetadata]:
        """Loads stored metadata for a given volunteer ID."""
        vol_dir = self._get_volunteer_dir(demo_id)
        meta_path = vol_dir / "metadata.json"
        if not meta_path.exists():
            return None
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return VolunteerMetadata(**data)

    def delete_volunteer(self, demo_id: str) -> bool:
        """Permanently deletes all data and directories for a volunteer.

        Returns:
            True if profile existed and was deleted; False otherwise.
        """
        vol_dir = self._get_volunteer_dir(demo_id)
        if vol_dir.exists() and vol_dir.is_dir():
            shutil.rmtree(vol_dir)
            logger.info("Permanently removed profile directory for volunteer %s", demo_id)
            return True
        return False
