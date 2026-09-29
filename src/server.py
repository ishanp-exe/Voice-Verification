"""FastAPI backend server for Customer Voice Authentication prototype.

Serves the modern 21st.dev-inspired web interface, REST endpoints for volunteer
voice enrollment, verification, profile lifecycle management, and corpus evaluation.
"""

from __future__ import annotations

import io
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.audio import AudioValidationError, load_and_validate_audio
from src.dataset_eval import LibriSpeechAdapter, TrialPair
from src.metrics import (
    EvaluatedTrial,
    compute_metric_sweep,
    compute_metrics_at_threshold,
    find_empirical_eer,
)
from src.model import ModelLoadError, SpeakerEmbeddingModel
from src.storage import VolunteerStorage
from src.verification import compute_cosine_similarity, verify_speakers

logger = logging.getLogger(__name__)

# Initialize FastAPI application
app = FastAPI(
    title="Customer Voice Authentication",
    description="Educational prototype for 1:1 speaker verification using SpeechBrain ECAPA-TDNN.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared singletons
storage = VolunteerStorage()
_model_instance: Optional[SpeakerEmbeddingModel] = None
_model_load_error: Optional[str] = None


def get_model() -> SpeakerEmbeddingModel:
    """Lazily loads and returns the speaker embedding model."""
    global _model_instance, _model_load_error
    if _model_instance is not None:
        return _model_instance

    try:
        _model_instance = SpeakerEmbeddingModel.load()
        _model_load_error = None
        return _model_instance
    except Exception as e:
        _model_load_error = str(e)
        logger.error("Failed to load pretrained model: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Pretrained speaker model could not be loaded: {e}",
        ) from e


# Static files and root route
STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def serve_index():
    """Serves the primary single-page application."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return JSONResponse(
        status_code=404,
        content={"error": "Frontend UI file index.html not found in src/static."},
    )


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/health")
async def health_check() -> Dict[str, Any]:
    """Returns application health and model availability."""
    global _model_instance, _model_load_error
    model_loaded = _model_instance is not None
    model_name = "SpeechBrain ECAPA-TDNN (spkrec-ecapa-voxceleb)"
    error_msg = _model_load_error

    if not model_loaded and error_msg is None:
        # Attempt proactive initial load
        try:
            get_model()
            model_loaded = True
        except Exception as e:
            model_loaded = False
            error_msg = str(e)

    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "model_name": model_name,
        "error": error_msg,
    }


@app.get("/api/volunteers")
async def list_volunteers() -> Dict[str, Any]:
    """Lists all enrolled volunteer profiles."""
    volunteers = storage.list_volunteers()
    return {"volunteers": [asdict(v) for v in volunteers]}


@app.post("/api/volunteers/enroll")
async def enroll_volunteer(
    demo_id: str = Form(...),
    consent: str = Form(...),
    file: UploadFile = File(...),
) -> Dict[str, Any]:
    """Enrolls a consenting volunteer by storing an extracted ECAPA-TDNN embedding vector.

    Strict privacy safeguard:
    - Raw audio is discarded by default after embedding extraction.
    - No model parameters are updated or fine-tuned.
    """
    if consent.lower() not in ("true", "1", "yes"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Informed volunteer consent is required to enroll.",
        )

    clean_id = demo_id.strip()
    if not clean_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Demo Speaker ID must not be empty.",
        )

    model = get_model()

    try:
        content = await file.read()
        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded audio file is empty.",
            )

        # Validate and convert audio to 16 kHz mono float32
        audio_stream = io.BytesIO(content)
        audio = load_and_validate_audio(audio_stream, min_duration_sec=0.8)

        # Extract 192-dimensional unit-norm embedding
        embedding = model.extract_embedding(audio.waveform)

        metadata = storage.save_volunteer(
            demo_id=clean_id,
            embedding=embedding,
            duration_sec=audio.duration_sec,
            original_sample_rate=audio.sample_rate,
            model_name=model.model_name,
            save_raw_audio=False,
        )

        return {
            "success": True,
            "demo_id": metadata.demo_id,
            "duration_sec": metadata.audio_duration_sec,
            "message": "Voice profile successfully enrolled.",
        }
    except AudioValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)) from e
    except Exception as e:
        logger.exception("Enrollment failed for demo_id '%s'", clean_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Enrollment failed: {e}",
        ) from e


@app.post("/api/volunteers/verify")
async def verify_volunteer(
    demo_id: str = Form(...),
    threshold: float = Form(0.25),
    consent: str = Form(...),
    file: UploadFile = File(...),
) -> Dict[str, Any]:
    """Performs 1:1 voice verification comparing candidate audio with enrolled embedding.

    Computes cosine similarity between embeddings and compares against threshold.
    """
    if consent.lower() not in ("true", "1", "yes"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Informed volunteer consent is required to perform verification.",
        )

    clean_id = demo_id.strip()
    enrolled_emb = storage.load_embedding(clean_id)
    if enrolled_emb is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Volunteer profile '{clean_id}' was not found in local storage.",
        )

    model = get_model()

    try:
        content = await file.read()
        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Verification audio sample is empty.",
            )

        audio_stream = io.BytesIO(content)
        audio = load_and_validate_audio(audio_stream, min_duration_sec=0.8)

        candidate_emb = model.extract_embedding(audio.waveform)
        result = verify_speakers(
            enrolled_embedding=enrolled_emb,
            candidate_embedding=candidate_emb,
            threshold=threshold,
        )

        return {
            "success": True,
            "demo_id": clean_id,
            "similarity": float(result.similarity),
            "threshold": float(result.threshold),
            "is_match": bool(result.is_match),
            "explanation": result.explanation,
        }
    except AudioValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)) from e
    except Exception as e:
        logger.exception("Verification error for demo_id '%s'", clean_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Verification failed: {e}",
        ) from e


@app.delete("/api/volunteers/{demo_id}")
async def delete_volunteer(demo_id: str) -> Dict[str, Any]:
    """Permanently purges a volunteer profile and all local files from disk."""
    clean_id = demo_id.strip()
    deleted = storage.delete_volunteer(clean_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Profile '{clean_id}' not found or could not be deleted.",
        )
    return {"success": True, "demo_id": clean_id, "message": "Profile permanently purged."}


# ---------------------------------------------------------------------------
# Evaluation Endpoints
# ---------------------------------------------------------------------------


class DatasetPathRequest(BaseModel):
    path: str


class EvaluationRunRequest(BaseModel):
    path: str
    num_genuine: int = 20
    num_imposter: int = 20
    enable_split: bool = True


@app.post("/api/evaluation/validate-dataset")
async def validate_dataset(req: DatasetPathRequest) -> Dict[str, Any]:
    """Validates an external dataset directory without copying files."""
    adapter = LibriSpeechAdapter(req.path)
    summary = adapter.validate()
    return {
        "is_valid": summary.is_valid,
        "message": summary.message,
        "num_speakers": summary.num_speakers,
        "num_recordings": summary.num_recordings,
        "sample_speakers": summary.sample_speakers,
    }


@app.post("/api/evaluation/run")
async def run_evaluation(req: EvaluationRunRequest) -> Dict[str, Any]:
    """Executes empirical verification benchmarking over real audio file pairs."""
    adapter = LibriSpeechAdapter(req.path)
    summary = adapter.validate()
    if not summary.is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid dataset: {summary.message}",
        )

    model = get_model()

    def _evaluate_batch(
        batch: List[TrialPair],
    ) -> tuple[List[EvaluatedTrial], List[dict]]:
        evaluated: List[EvaluatedTrial] = []
        skipped: List[dict] = []
        for t in batch:
            try:
                audio_a = load_and_validate_audio(t.path_a, min_duration_sec=0.5)
                audio_b = load_and_validate_audio(t.path_b, min_duration_sec=0.5)

                emb_a = model.extract_embedding(audio_a.waveform)
                emb_b = model.extract_embedding(audio_b.waveform)

                sim = compute_cosine_similarity(emb_a, emb_b)
                evaluated.append(
                    EvaluatedTrial(
                        path_a=str(t.path_a),
                        path_b=str(t.path_b),
                        speaker_a=t.speaker_a,
                        speaker_b=t.speaker_b,
                        is_genuine=t.is_genuine,
                        similarity=sim,
                    )
                )
            except Exception as e:
                skipped.append({
                    "Speakers": f"{t.speaker_a} vs {t.speaker_b}",
                    "Files": f"{Path(t.path_a).name} vs {Path(t.path_b).name}",
                    "Reason": str(e),
                })
        return evaluated, skipped

    cal_speakers, eval_speakers, is_split_reliable, split_note = (
        adapter.split_speakers_for_calibration(calibration_fraction=0.5)
    )

    if req.enable_split and is_split_reliable:
        # Protocol Branch 1: Disciplined Dev Calibration and Independent Test Split
        cal_trials = adapter.generate_trials(
            num_genuine=req.num_genuine,
            num_imposter=req.num_imposter,
            speaker_subset=cal_speakers,
            seed=42,
        )
        test_trials = adapter.generate_trials(
            num_genuine=req.num_genuine,
            num_imposter=req.num_imposter,
            speaker_subset=eval_speakers,
            seed=100,
        )

        cal_eval, cal_skip = _evaluate_batch(cal_trials)
        test_eval, test_skip = _evaluate_batch(test_trials)

        if not cal_eval or not test_eval:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to evaluate sufficient trials in calibration or test partitions.",
            )

        cal_sweep = compute_metric_sweep(cal_eval, num_steps=60)
        cal_eer = find_empirical_eer(cal_sweep)
        calibrated_threshold = cal_eer.threshold if cal_eer else 0.25

        # Measure performance strictly on held-out test partition at calibrated threshold
        test_metric = compute_metrics_at_threshold(test_eval, threshold=calibrated_threshold)

        return {
            "success": True,
            "protocol": "calibration_split",
            "calibrated_threshold": float(calibrated_threshold),
            "test_far": float(test_metric.far),
            "test_frr": float(test_metric.frr),
            "test_accuracy": float(test_metric.accuracy),
            "total_trials": len(cal_eval) + len(test_eval),
            "skipped_trials": cal_skip + test_skip,
            "note": split_note,
        }

    else:
        # Protocol Branch 2: Development Preview (Unsplit)
        trials = adapter.generate_trials(
            num_genuine=req.num_genuine,
            num_imposter=req.num_imposter,
            speaker_subset=None,
            seed=42,
        )
        evaluated, skipped = _evaluate_batch(trials)

        if not evaluated:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not evaluate any trials from the selected dataset.",
            )

        sweep = compute_metric_sweep(evaluated, num_steps=60)
        eer_metric = find_empirical_eer(sweep)
        operating_threshold = eer_metric.threshold if eer_metric else 0.25

        return {
            "success": True,
            "protocol": "unsplit_dev_preview",
            "calibrated_threshold": float(operating_threshold),
            "test_far": float(eer_metric.far if eer_metric else 0.0),
            "test_frr": float(eer_metric.frr if eer_metric else 0.0),
            "test_accuracy": float(eer_metric.accuracy if eer_metric else 0.0),
            "total_trials": len(evaluated),
            "skipped_trials": skipped,
            "note": "Development Preview Mode: Evaluated across available data without disjoint partition.",
        }
