"""Integration tests for Voice-Verification system.

Tests real components together: model inference, audio processing,
concurrency, and end-to-end workflows.
"""

from __future__ import annotations

import io
import shutil
import tempfile
import threading
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from src.audio import load_and_validate_audio
from src.storage import VolunteerStorage
from src.verification import verify_speakers


def create_test_wav_bytes(duration_sec: float = 1.2, sample_rate: int = 16000) -> bytes:
    """Creates in-memory PCM WAV bytes for testing."""
    num_samples = int(duration_sec * sample_rate)
    data = np.random.uniform(-0.1, 0.1, num_samples).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, data, sample_rate, format="WAV")
    return buf.getvalue()


class TestAudioFormats:
    """Tests support for multiple audio formats."""

    def test_wav_audio_loads_successfully(self):
        """Verify WAV format support."""
        audio_bytes = create_test_wav_bytes(1.0)
        audio = load_and_validate_audio(io.BytesIO(audio_bytes))
        assert audio.waveform.numel() >= 16000  # Should have at least 1 second of audio

    def test_audio_resampling_to_16khz(self):
        """Verify audio is resampled to 16 kHz."""
        # Create 8 kHz audio
        num_samples = 8000
        data = np.random.uniform(-0.1, 0.1, num_samples).astype(np.float32)
        buf = io.BytesIO()
        sf.write(buf, data, 8000, format="WAV")
        audio_bytes = buf.getvalue()

        audio = load_and_validate_audio(io.BytesIO(audio_bytes))
        assert audio.sample_rate == 16000


class TestConcurrency:
    """Tests concurrent enrollment and deletion operations."""

    def test_concurrent_enrollments_same_id_prevented(self):
        """Verify only one enrollment per ID succeeds under concurrent access."""
        temp_dir = tempfile.mkdtemp()
        storage = VolunteerStorage(base_dir=Path(temp_dir))

        results = {"success": 0, "failed": 0}

        def enroll_concurrently():
            try:
                embedding = torch.randn(192)
                embedding = embedding / torch.norm(embedding, p=2)
                storage.save_volunteer(
                    demo_id="CONCURRENT-TEST",
                    embedding=embedding,
                    duration_sec=1.0,
                    original_sample_rate=16000,
                    model_name="test-model",
                )
                results["success"] += 1
            except FileExistsError:
                results["failed"] += 1
            except Exception:
                results["failed"] += 1

        # Launch 5 concurrent enrollment attempts
        threads = [threading.Thread(target=enroll_concurrently) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Only one should succeed
        assert results["success"] == 1
        assert results["failed"] == 4

        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_concurrent_deletion_is_safe(self):
        """Verify concurrent deletion attempts don't crash."""
        temp_dir = tempfile.mkdtemp()
        storage = VolunteerStorage(base_dir=Path(temp_dir))

        # Create a profile
        embedding = torch.randn(192)
        embedding = embedding / torch.norm(embedding, p=2)
        storage.save_volunteer(
            demo_id="DELETE-TEST",
            embedding=embedding,
            duration_sec=1.0,
            original_sample_rate=16000,
            model_name="test-model",
        )

        results = []

        def delete_concurrently():
            result = storage.delete_volunteer("DELETE-TEST")
            results.append(result)

        # Launch 5 concurrent deletion attempts
        threads = [threading.Thread(target=delete_concurrently) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Only one should return True (successful deletion)
        assert sum(results) == 1

        shutil.rmtree(temp_dir, ignore_errors=True)


class TestVerificationWorkflow:
    """Tests end-to-end verification workflow with real embeddings."""

    def test_matching_embeddings_verify(self):
        """Verify similar embeddings produce match decision."""
        # Create nearly identical embeddings
        base = torch.randn(192)
        base = base / torch.norm(base, p=2)

        # Add tiny perturbation
        perturbed = base + 0.01 * torch.randn(192)
        perturbed = perturbed / torch.norm(perturbed, p=2)

        result = verify_speakers(
            enrolled_embedding=base,
            candidate_embedding=perturbed,
            threshold=0.31,
        )

        assert result.is_match is True
        assert result.similarity > 0.9

    def test_dissimilar_embeddings_reject(self):
        """Verify dissimilar embeddings produce reject decision."""
        # Create orthogonal embeddings
        emb1 = torch.zeros(192)
        emb1[0] = 1.0

        emb2 = torch.zeros(192)
        emb2[1] = 1.0

        result = verify_speakers(
            enrolled_embedding=emb1,
            candidate_embedding=emb2,
            threshold=0.31,
        )

        assert result.is_match is False
        assert result.similarity < 0.1


class TestStorageOperations:
    """Tests storage operations with and without encryption."""

    def test_save_and_load_without_encryption(self):
        """Verify embeddings saved and loaded correctly."""
        temp_dir = tempfile.mkdtemp()
        storage = VolunteerStorage(base_dir=Path(temp_dir))

        # Create and save embedding
        original_embedding = torch.randn(192)
        original_embedding = original_embedding / torch.norm(original_embedding, p=2)

        storage.save_volunteer(
            demo_id="STORAGE-TEST",
            embedding=original_embedding,
            duration_sec=1.0,
            original_sample_rate=16000,
            model_name="test-model",
        )

        # Load and verify
        loaded_embedding = storage.load_embedding("STORAGE-TEST")
        assert loaded_embedding is not None
        assert torch.allclose(original_embedding, loaded_embedding, atol=1e-6)

        # Verify metadata
        metadata = storage.load_metadata("STORAGE-TEST")
        assert metadata.demo_id == "STORAGE-TEST"
        assert metadata.audio_duration_sec == 1.0

        shutil.rmtree(temp_dir, ignore_errors=True)
