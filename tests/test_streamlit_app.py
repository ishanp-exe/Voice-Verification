"""Tests for Streamlit application."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


def test_streamlit_app_loads_successfully():
    """Verifies that app.py runs cleanly without unhandled exceptions."""
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    assert not at.exception
    assert len(at.tabs) == 4
    # Check that tabs are present
    tab_labels = [tab.label for tab in at.tabs]
    assert any("Step 1" in label for label in tab_labels)
    assert any("Step 2" in label for label in tab_labels)
    assert any("Governance" in label for label in tab_labels)
    assert any("LibriSpeech" in label for label in tab_labels)


def test_volunteer_enrollment_governance_and_deletion_lifecycle():
    """Verifies enrollment, profile listing in governance, and deletion in storage and app rendering."""
    import tempfile
    import shutil
    import torch
    from src.storage import VolunteerStorage
    from src.verification import combine_enrollment_embeddings

    temp_dir = Path(tempfile.mkdtemp())
    try:
        test_storage = VolunteerStorage(base_dir=temp_dir)
        test_id = "VOL-TEST99"

        # 1. Simulate multi-sample embedding extraction & combination
        emb1 = torch.randn(192)
        emb1 = emb1 / torch.norm(emb1, p=2)
        emb2 = torch.randn(192)
        emb2 = emb2 / torch.norm(emb2, p=2)
        combined = combine_enrollment_embeddings([emb1, emb2])

        # 2. Call save_volunteer with exact parameters from app.py
        meta = test_storage.save_volunteer(
            demo_id=test_id,
            embedding=combined,
            duration_sec=3.2,
            original_sample_rate=16000,
            model_name="mock-ecapa",
            num_enrollment_samples=2,
            save_raw_audio=False,
        )

        assert meta.demo_id == test_id
        assert meta.num_enrollment_samples == 2
        assert meta.audio_duration_sec == 3.2

        # 3. Verify profile appears in listing (used by Governance and Verification tabs)
        profiles = test_storage.list_volunteers()
        assert len(profiles) == 1
        p = profiles[0]
        assert p.demo_id == test_id
        assert p.num_enrollment_samples == 2
        # Check compatibility with both attribute and subscript access
        assert p["demo_id"] == test_id
        assert p.get("created_at") is not None
        assert p.enrolled_at_utc is not None

        # 4. Verify embedding load
        loaded_emb = test_storage.load_embedding(test_id)
        assert loaded_emb is not None
        assert torch.allclose(loaded_emb, combined)

        # 5. Verify deletion / purging
        deleted = test_storage.delete_volunteer(test_id)
        assert deleted is True
        assert len(test_storage.list_volunteers()) == 0
        assert test_storage.load_embedding(test_id) is None
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

