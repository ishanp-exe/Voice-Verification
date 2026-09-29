"""Unit tests for Customer Voice Authentication components.

Uses mocked model outputs for testing code paths without requiring real speech inference.
Tests:
- Audio loading and validation (unreadable, empty, short, multi-channel, resampling)
- Verification math and threshold logic
- Local volunteer storage (saving, retrieval, metadata, and deletion)
- Dataset adapter and trial generation
- Metric calculation (FAR, FRR, accuracy, empirical EER)
"""

from __future__ import annotations

import io
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf
import torch

from src.audio import AudioValidationError, load_and_validate_audio
from src.dataset_eval import LibriSpeechAdapter, TrialPair
from src.metrics import (
    EvaluatedTrial,
    compute_metric_sweep,
    compute_metrics_at_threshold,
    find_empirical_eer,
)
from src.model import ModelLoadError, SpeakerEmbeddingModel
from src.storage import VolunteerMetadata, VolunteerStorage
from src.verification import compute_cosine_similarity, verify_speakers


# ---------------------------------------------------------------------------
# Helpers for generating test audio streams
# ---------------------------------------------------------------------------
def create_test_wav_bytes(duration_sec: float, sample_rate: int = 16000, num_channels: int = 1) -> bytes:
    """Creates an in-memory WAV file for unit tests."""
    num_samples = int(duration_sec * sample_rate)
    # Generate low-amplitude noise
    if num_channels == 1:
        data = np.random.uniform(-0.1, 0.1, num_samples).astype(np.float32)
    else:
        data = np.random.uniform(-0.1, 0.1, (num_samples, num_channels)).astype(np.float32)

    buf = io.BytesIO()
    sf.write(buf, data, sample_rate, format="WAV")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 1. Audio Validation Tests
# ---------------------------------------------------------------------------
class TestAudioValidation:
    def test_empty_audio_raises_error(self):
        with pytest.raises(AudioValidationError, match="empty"):
            load_and_validate_audio(b"")

    def test_corrupt_audio_raises_error(self):
        with pytest.raises(AudioValidationError, match="Unable to decode"):
            load_and_validate_audio(b"not_a_valid_wav_header_bytes")

    def test_too_short_audio_raises_error(self):
        short_wav = create_test_wav_bytes(duration_sec=0.4, sample_rate=16000)
        with pytest.raises(AudioValidationError, match="Recording is too short"):
            load_and_validate_audio(short_wav, min_duration_sec=1.0)

    def test_valid_audio_loads_and_resamples(self):
        # 1.5s audio sampled at 44100 Hz, stereo (2 channels)
        stereo_wav = create_test_wav_bytes(duration_sec=1.5, sample_rate=44100, num_channels=2)
        audio_data = load_and_validate_audio(stereo_wav, target_sample_rate=16000, min_duration_sec=1.0)

        assert audio_data.waveform.dim() == 2
        assert audio_data.waveform.shape[0] == 1  # Converted to mono
        assert audio_data.sample_rate == 16000
        assert audio_data.original_sr == 44100
        assert audio_data.num_channels == 2
        assert abs(audio_data.duration_sec - 1.5) < 0.05


# ---------------------------------------------------------------------------
# 2. Verification and Cosine Similarity Tests
# ---------------------------------------------------------------------------
class TestVerification:
    def test_cosine_similarity_identical_vectors(self):
        v = torch.tensor([1.0, 2.0, 3.0, 4.0])
        sim = compute_cosine_similarity(v, v)
        assert pytest.approx(sim, abs=1e-5) == 1.0

    def test_cosine_similarity_orthogonal_vectors(self):
        u = torch.tensor([1.0, 0.0])
        v = torch.tensor([0.0, 1.0])
        sim = compute_cosine_similarity(u, v)
        assert pytest.approx(sim, abs=1e-5) == 0.0

    def test_cosine_similarity_opposite_vectors(self):
        u = torch.tensor([1.0, 2.0])
        v = torch.tensor([-1.0, -2.0])
        sim = compute_cosine_similarity(u, v)
        assert pytest.approx(sim, abs=1e-5) == -1.0

    def test_dimension_mismatch_raises_error(self):
        u = torch.tensor([1.0, 2.0])
        v = torch.tensor([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="dimension mismatch"):
            compute_cosine_similarity(u, v)

    def test_verify_speakers_match_and_no_match(self):
        u = torch.tensor([1.0, 0.0])
        v_match = torch.tensor([0.9, 0.1])
        res_match = verify_speakers(u, v_match, threshold=0.5)
        assert res_match.is_match is True

        v_no_match = torch.tensor([0.1, 0.9])
        res_no_match = verify_speakers(u, v_no_match, threshold=0.5)
        assert res_no_match.is_match is False
        assert "Cosine similarity score" in res_no_match.explanation


# ---------------------------------------------------------------------------
# 3. Volunteer Storage & Privacy Tests
# ---------------------------------------------------------------------------
class TestVolunteerStorage:
    @pytest.fixture
    def temp_storage(self):
        temp_dir = Path(tempfile.mkdtemp(prefix="test_vol_storage_"))
        storage = VolunteerStorage(base_dir=temp_dir)
        yield storage
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    def test_save_and_load_volunteer_profile(self, temp_storage):
        demo_id = temp_storage.generate_demo_id()
        mock_embedding = torch.randn(192)
        mock_embedding = mock_embedding / torch.norm(mock_embedding)

        metadata = temp_storage.save_volunteer(
            demo_id=demo_id,
            embedding=mock_embedding,
            duration_sec=2.5,
            original_sample_rate=16000,
            model_name="mock_ecapa",
            save_raw_audio=False,
        )

        assert metadata.demo_id == demo_id
        assert metadata.has_raw_audio is False

        # Load embedding back
        loaded_emb = temp_storage.load_embedding(demo_id)
        assert loaded_emb is not None
        assert torch.allclose(mock_embedding, loaded_emb, atol=1e-5)

        # Ensure raw audio was NOT saved
        vol_dir = temp_storage._get_volunteer_dir(demo_id)
        assert not (vol_dir / "enrollment_sample.wav").exists()

        # Check list volunteers
        vols = temp_storage.list_volunteers()
        assert len(vols) == 1
        assert vols[0].demo_id == demo_id

        # Delete volunteer
        deleted = temp_storage.delete_volunteer(demo_id)
        assert deleted is True
        assert not vol_dir.exists()
        assert temp_storage.load_embedding(demo_id) is None
        assert len(temp_storage.list_volunteers()) == 0

    def test_invalid_demo_id_raises_value_error(self, temp_storage):
        with pytest.raises(ValueError, match="Invalid demo ID"):
            temp_storage.delete_volunteer("")
        with pytest.raises(ValueError, match="Invalid demo ID"):
            temp_storage.delete_volunteer("   ")
        with pytest.raises(ValueError, match="Invalid demo ID"):
            temp_storage._get_volunteer_dir("////")


# ---------------------------------------------------------------------------
# 4. Mocked Model Inference Tests
# ---------------------------------------------------------------------------
class TestModelInference:
    def test_extract_embedding_with_mock_classifier(self):
        # Create a mock SpeechBrain classifier
        mock_classifier = MagicMock()
        # Mock encode_batch returning shape [1, 1, 192]
        mock_classifier.encode_batch.return_value = torch.ones(1, 1, 192)

        model = SpeakerEmbeddingModel(classifier=mock_classifier, model_name="mock_model")
        dummy_waveform = torch.zeros(1, 16000)
        emb = model.extract_embedding(dummy_waveform)

        assert emb.dim() == 1
        assert emb.shape[0] == 192
        # Check L2 normalization (unit length)
        norm = torch.norm(emb, p=2).item()
        assert pytest.approx(norm, abs=1e-5) == 1.0

    def test_model_load_error_handling(self, monkeypatch):
        # Simulate import error or network failure
        def fake_load(*args, **kwargs):
            raise ConnectionError("Hugging Face Hub unreachable")

        monkeypatch.setattr(SpeakerEmbeddingModel, "load", fake_load)
        with pytest.raises(ConnectionError, match="Hugging Face Hub unreachable"):
            SpeakerEmbeddingModel.load()


# ---------------------------------------------------------------------------
# 5. Dataset Evaluation & Trial Generation Tests
# ---------------------------------------------------------------------------
class TestDatasetEvaluation:
    @pytest.fixture
    def dummy_librispeech_dir(self):
        temp_dir = Path(tempfile.mkdtemp(prefix="dummy_librispeech_"))
        # Create structure: 101/chapter1/*.wav and 102/chapter1/*.wav
        spk1 = temp_dir / "101" / "1001"
        spk2 = temp_dir / "102" / "2001"
        spk1.mkdir(parents=True)
        spk2.mkdir(parents=True)

        # Create dummy wav files
        for i in range(3):
            sf.write(spk1 / f"101-1001-{i:04d}.wav", np.zeros(16000, dtype=np.float32), 16000)
            sf.write(spk2 / f"102-2001-{i:04d}.wav", np.zeros(16000, dtype=np.float32), 16000)

        yield temp_dir
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    def test_adapter_discovery_and_trials(self, dummy_librispeech_dir):
        adapter = LibriSpeechAdapter(dummy_librispeech_dir)
        summary = adapter.validate()

        assert summary.is_valid is True
        assert summary.num_speakers == 2
        assert summary.num_recordings == 6

        trials = adapter.generate_trials(num_genuine=2, num_imposter=2, seed=42)
        assert len(trials) == 4

        genuine_trials = [t for t in trials if t.is_genuine]
        imposter_trials = [t for t in trials if not t.is_genuine]

        assert len(genuine_trials) == 2
        assert len(imposter_trials) == 2

        for t in genuine_trials:
            assert t.speaker_a == t.speaker_b
            assert t.path_a != t.path_b

        for t in imposter_trials:
            assert t.speaker_a != t.speaker_b


# ---------------------------------------------------------------------------
# 6. Empirical Metrics Calculation Tests
# ---------------------------------------------------------------------------
class TestMetricsCalculation:
    def test_empirical_metrics_exact_math(self):
        # 2 genuine trials (similarity 0.8, 0.4)
        # 2 imposter trials (similarity 0.2, 0.5)
        trials = [
            EvaluatedTrial("a", "b", "spk1", "spk1", True, 0.8),
            EvaluatedTrial("c", "d", "spk1", "spk1", True, 0.4),
            EvaluatedTrial("e", "f", "spk1", "spk2", False, 0.2),
            EvaluatedTrial("g", "h", "spk1", "spk3", False, 0.5),
        ]

        # At threshold 0.3:
        # Genuine: 0.8 >= 0.3 (TA), 0.4 >= 0.3 (TA) -> FR=0, FRR=0.0
        # Imposter: 0.2 < 0.3 (TR), 0.5 >= 0.3 (FA) -> FA=1, FAR=0.5
        metric_03 = compute_metrics_at_threshold(trials, threshold=0.3)
        assert metric_03.genuine_trials == 2
        assert metric_03.imposter_trials == 2
        assert metric_03.false_accepts == 1
        assert metric_03.false_rejects == 0
        assert metric_03.true_accepts == 2
        assert metric_03.true_rejects == 1
        assert pytest.approx(metric_03.far) == 0.5
        assert pytest.approx(metric_03.frr) == 0.0
        assert pytest.approx(metric_03.accuracy) == 0.75  # 3 / 4

        # At threshold 0.6:
        # Genuine: 0.8 >= 0.6 (TA), 0.4 < 0.6 (FR) -> FR=1, FRR=0.5
        # Imposter: 0.2 < 0.6 (TR), 0.5 < 0.6 (TR) -> FA=0, FAR=0.0
        metric_06 = compute_metrics_at_threshold(trials, threshold=0.6)
        assert metric_06.false_accepts == 0
        assert metric_06.false_rejects == 1
        assert pytest.approx(metric_06.far) == 0.0
        assert pytest.approx(metric_06.frr) == 0.5
        assert pytest.approx(metric_06.accuracy) == 0.75

    def test_metric_sweep_and_eer(self):
        trials = [
            EvaluatedTrial("a", "b", "spk1", "spk1", True, 0.7),
            EvaluatedTrial("c", "d", "spk1", "spk1", True, 0.8),
            EvaluatedTrial("e", "f", "spk1", "spk2", False, 0.2),
            EvaluatedTrial("g", "h", "spk1", "spk3", False, 0.3),
        ]
        sweep = compute_metric_sweep(trials, num_steps=20)
        assert len(sweep) == 20
        eer = find_empirical_eer(sweep)
        assert eer is not None
        assert 0.2 <= eer.threshold <= 0.8

    def test_calibration_evaluation_split_balanced(self):
        # Create a mock adapter with 6 multi-utterance speakers
        adapter = LibriSpeechAdapter(root_dir=None)
        mock_map = {
            f"spk_{i}": [Path(f"file_{i}_1.wav"), Path(f"file_{i}_2.wav")]
            for i in range(8)
        }
        adapter._speaker_map = mock_map

        cal, ev, reliable, note = adapter.split_speakers_for_calibration(calibration_fraction=0.5)
        assert reliable is True
        assert len(cal) > 0
        assert len(ev) > 0
        assert cal.isdisjoint(ev)
        assert len(cal) + len(ev) == 8

        # Test small dataset fallback
        adapter._speaker_map = {"spk_1": [Path("a.wav"), Path("b.wav")]}
        cal_small, ev_small, reliable_small, note_small = adapter.split_speakers_for_calibration()
        assert reliable_small is False
        assert "statistically reliable split cannot be performed" in note_small

    def test_calibration_protocol_applies_calibrated_threshold_to_test_data(self):
        # Calibration trials used to find operating point
        cal_trials = [
            EvaluatedTrial("a", "b", "c1", "c1", True, 0.75),
            EvaluatedTrial("c", "d", "c1", "c1", True, 0.85),
            EvaluatedTrial("e", "f", "c1", "c2", False, 0.15),
            EvaluatedTrial("g", "h", "c1", "c2", False, 0.25),
        ]
        cal_sweep = compute_metric_sweep(cal_trials, num_steps=50)
        cal_eer = find_empirical_eer(cal_sweep)
        assert cal_eer is not None
        tau_cal = cal_eer.threshold

        # Held-out test trials
        test_trials = [
            EvaluatedTrial("i", "j", "t1", "t1", True, 0.80),
            EvaluatedTrial("k", "l", "t1", "t2", False, 0.20),
        ]
        # Evaluate test metrics strictly at tau_cal
        final_test_metric = compute_metrics_at_threshold(test_trials, threshold=tau_cal)
        assert final_test_metric.threshold == round(tau_cal, 4)
        assert final_test_metric.total_trials == 2
        assert final_test_metric.false_accepts == 0
        assert final_test_metric.false_rejects == 0
        assert final_test_metric.accuracy == 1.0
