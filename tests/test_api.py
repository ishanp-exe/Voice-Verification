"""API tests for Customer Voice Authentication FastAPI backend.

Tests REST endpoints using FastAPI TestClient with mocked model extraction.
"""

from __future__ import annotations

import io
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
import torch
from fastapi.testclient import TestClient

import src.server as server_module
from src.server import app
from src.storage import VolunteerStorage


def create_test_wav_bytes(duration_sec: float = 1.2, sample_rate: int = 16000) -> bytes:
    """Creates in-memory PCM WAV bytes."""
    num_samples = int(duration_sec * sample_rate)
    data = np.random.uniform(-0.1, 0.1, num_samples).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, data, sample_rate, format="WAV")
    return buf.getvalue()


@pytest.fixture
def temp_storage():
    """Provides an isolated temp storage directory for tests."""
    temp_dir = tempfile.mkdtemp()
    storage = VolunteerStorage(base_dir=Path(temp_dir))
    original_storage = server_module.storage
    server_module.storage = storage
    yield storage
    server_module.storage = original_storage
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_model():
    """Provides a mocked SpeakerEmbeddingModel that returns deterministic embeddings."""
    mock = MagicMock()
    mock.model_name = "mock-ecapa-tdnn"
    # Return normalized 192-dim random vector
    vec = torch.randn(192)
    vec = vec / torch.norm(vec, p=2)
    mock.extract_embedding.return_value = vec

    with patch.object(server_module, "get_model", return_value=mock):
        yield mock


@pytest.fixture
def client(temp_storage, mock_model):
    """Provides a TestClient with temporary storage and mock model."""
    return TestClient(app)


class TestHealthAndStatic:
    def test_health_endpoint(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "model_name" in data

    def test_root_serves_html(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        assert "VoxKey" in response.text


class TestVolunteerWorkflows:
    def test_empty_volunteers_list(self, client):
        response = client.get("/api/volunteers")
        assert response.status_code == 200
        assert response.json() == {"volunteers": []}

    def test_enrollment_requires_consent(self, client):
        wav_bytes = create_test_wav_bytes(1.5)
        response = client.post(
            "/api/volunteers/enroll",
            data={"demo_id": "VOL-1234", "consent": "false"},
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        )
        assert response.status_code == 400
        assert "consent" in response.json()["detail"].lower()

    def test_enrollment_and_listing_success(self, client, mock_model):
        wav_bytes = create_test_wav_bytes(1.5)
        response = client.post(
            "/api/volunteers/enroll",
            data={"demo_id": "VOL-1234", "consent": "true"},
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["demo_id"] == "VOL-1234"
        assert data["duration_sec"] >= 1.0

        # Verify listed in GET /api/volunteers
        list_res = client.get("/api/volunteers")
        assert list_res.status_code == 200
        vols = list_res.json()["volunteers"]
        assert len(vols) == 1
        assert vols[0]["demo_id"] == "VOL-1234"
        assert vols[0]["num_enrollment_samples"] == 1

    def test_enrollment_multi_sample_success(self, client, mock_model):
        wav_bytes1 = create_test_wav_bytes(1.2)
        wav_bytes2 = create_test_wav_bytes(1.5)
        response = client.post(
            "/api/volunteers/enroll",
            data={"demo_id": "VOL-MULTI", "consent": "true"},
            files=[
                ("files", ("take1.wav", wav_bytes1, "audio/wav")),
                ("files", ("take2.wav", wav_bytes2, "audio/wav")),
            ],
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["demo_id"] == "VOL-MULTI"
        assert data["num_samples"] == 2
        assert data["duration_sec"] >= 2.5

        list_res = client.get("/api/volunteers")
        vols = list_res.json()["volunteers"]
        multi_vol = next(v for v in vols if v["demo_id"] == "VOL-MULTI")
        assert multi_vol["num_enrollment_samples"] == 2

    def test_enrollment_multi_sample_invalid_file_rejected(self, client):
        wav_bytes1 = create_test_wav_bytes(1.2)
        wav_bytes2 = create_test_wav_bytes(0.3)  # Too short (< 0.8s)
        response = client.post(
            "/api/volunteers/enroll",
            data={"demo_id": "VOL-FAIL", "consent": "true"},
            files=[
                ("files", ("take1.wav", wav_bytes1, "audio/wav")),
                ("files", ("take2.wav", wav_bytes2, "audio/wav")),
            ],
        )
        assert response.status_code == 422
        assert "#2" in response.json()["detail"]

    def test_verification_requires_profile(self, client):
        wav_bytes = create_test_wav_bytes(1.5)
        response = client.post(
            "/api/volunteers/verify",
            data={"demo_id": "VOL-NONEXISTENT", "consent": "true", "threshold": 0.25},
            files={"file": ("verify.wav", wav_bytes, "audio/wav")},
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_verification_match_and_nomatch(self, client, mock_model):
        # 1. Enroll
        wav_bytes = create_test_wav_bytes(1.5)
        client.post(
            "/api/volunteers/enroll",
            data={"demo_id": "VOL-8888", "consent": "true"},
            files={"file": ("enroll.wav", wav_bytes, "audio/wav")},
        )

        # 2. Verify with matching embedding (mock returns same vector)
        response = client.post(
            "/api/volunteers/verify",
            data={"demo_id": "VOL-8888", "consent": "true", "threshold": 0.25},
            files={"file": ("candidate.wav", wav_bytes, "audio/wav")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["demo_id"] == "VOL-8888"
        assert data["is_match"] is True
        assert data["similarity"] >= 0.99

        # 3. Verify with orthogonal / non-matching candidate
        diff_vec = torch.randn(192)
        diff_vec = diff_vec / torch.norm(diff_vec, p=2)
        # Ensure it has low cosine similarity
        mock_model.extract_embedding.return_value = diff_vec

        response2 = client.post(
            "/api/volunteers/verify",
            data={"demo_id": "VOL-8888", "consent": "true", "threshold": 0.80},
            files={"file": ("candidate2.wav", wav_bytes, "audio/wav")},
        )
        assert response2.status_code == 200
        data2 = response2.json()
        assert data2["is_match"] is False

    def test_delete_profile(self, client):
        wav_bytes = create_test_wav_bytes(1.5)
        client.post(
            "/api/volunteers/enroll",
            data={"demo_id": "VOL-9999", "consent": "true"},
            files={"file": ("enroll.wav", wav_bytes, "audio/wav")},
        )

        del_res = client.delete("/api/volunteers/VOL-9999")
        assert del_res.status_code == 200
        assert del_res.json()["success"] is True

        # Confirm gone
        list_res = client.get("/api/volunteers")
        assert len(list_res.json()["volunteers"]) == 0

        # Deleting non-existent returns 404
        del_again = client.delete("/api/volunteers/VOL-9999")
        assert del_again.status_code == 404


class TestEvaluationEndpoints:
    def test_validate_dataset_nonexistent(self, client):
        res = client.post(
            "/api/evaluation/validate-dataset",
            json={"path": "C:\\path\\does\\not\\exist\\12345"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["is_valid"] is False
        assert "not exist" in data["message"].lower()

    def test_validate_and_run_dataset_mock(self, client, mock_model):
        # Create a small temporary dataset tree
        temp_corpus = tempfile.mkdtemp()
        try:
            for spk in ["1001", "1002", "1003", "1004", "1005", "1006"]:
                spk_dir = Path(temp_corpus) / spk / "chapter1"
                spk_dir.mkdir(parents=True, exist_ok=True)
                for i in range(2):
                    audio_bytes = create_test_wav_bytes(1.0)
                    with open(spk_dir / f"{spk}-100-{i}.flac", "wb") as f:
                        f.write(audio_bytes)

            val_res = client.post(
                "/api/evaluation/validate-dataset",
                json={"path": temp_corpus},
            )
            assert val_res.status_code == 200
            val_data = val_res.json()
            assert val_data["is_valid"] is True
            assert val_data["num_speakers"] == 6

            # Run evaluation with split
            run_res = client.post(
                "/api/evaluation/run",
                json={
                    "path": temp_corpus,
                    "num_genuine": 4,
                    "num_imposter": 4,
                    "enable_split": True,
                },
            )
            assert run_res.status_code == 200
            run_data = run_res.json()
            assert run_data["success"] is True
            assert "calibrated_threshold" in run_data
            assert "test_far" in run_data
            assert "test_frr" in run_data
            assert "test_accuracy" in run_data
            assert "trial_breakdown" in run_data
            assert "cache_stats" in run_data
            assert run_data["calibration_seed"] == 42
            assert run_data["evaluation_seed"] == 100
            assert "disclaimer" in run_data
            assert "audiobook" in run_data["disclaimer"].lower()
        finally:
            shutil.rmtree(temp_corpus, ignore_errors=True)
