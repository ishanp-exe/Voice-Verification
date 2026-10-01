"""FastAPI backend server for VoxKey prototype.

Serves the modern dark analytics web interface, REST endpoints for volunteer
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
from src.verification import (
    combine_enrollment_embeddings,
    compute_cosine_similarity,
    verify_speakers,
)

logger = logging.getLogger(__name__)

# Initialize FastAPI application
app = FastAPI(
    title="VoxKey",
    description="Educational voice-verification prototype using SpeechBrain ECAPA-TDNN.",
    version="1.0.0",
)

# Tightened CORS: strictly restricted to local web application origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
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
    files: Optional[List[UploadFile]] = File(None),
    file: Optional[UploadFile] = File(None),
) -> Dict[str, Any]:
    """Enrolls a consenting volunteer by storing an extracted ECAPA-TDNN embedding vector.

    Supports single or multiple separate recordings. When multiple recordings are provided,
    their embeddings are averaged and L2-normalized into a single robust enrollment vector.
    Strict privacy safeguards:
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

    # Collect all uploaded files (handles both single 'file' and multi 'files')
    all_files: List[UploadFile] = []
    if files:
        all_files.extend(files)
    if file:
        all_files.append(file)

    if not all_files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one audio recording file is required for enrollment.",
        )

    model = get_model()
    extracted_embeddings: List[torch.Tensor] = []
    total_duration_sec: float = 0.0
    original_sample_rate: int = 16000

    try:
        for idx, upload in enumerate(all_files):
            content = await upload.read()
            if not content:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Audio recording #{idx + 1} ({upload.filename or 'unnamed'}) is empty.",
                )

            # Validate and convert audio to 16 kHz mono float32
            audio_stream = io.BytesIO(content)
            try:
                audio = load_and_validate_audio(audio_stream, min_duration_sec=0.8)
            except AudioValidationError as e:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Recording #{idx + 1} ({upload.filename or 'sample'}) validation failed: {e}",
                ) from e

            # Extract 192-dimensional unit-norm embedding
            embedding = model.extract_embedding(audio.waveform)
            extracted_embeddings.append(embedding)
            total_duration_sec += audio.duration_sec
            original_sample_rate = audio.sample_rate

        # Combine multiple embeddings into a single unit-normalized vector
        combined_embedding = combine_enrollment_embeddings(extracted_embeddings)

        metadata = storage.save_volunteer(
            demo_id=clean_id,
            embedding=combined_embedding,
            duration_sec=total_duration_sec,
            original_sample_rate=original_sample_rate,
            model_name=model.model_name,
            num_enrollment_samples=len(extracted_embeddings),
            save_raw_audio=False,
        )

        return {
            "success": True,
            "demo_id": metadata.demo_id,
            "duration_sec": metadata.audio_duration_sec,
            "num_samples": metadata.num_enrollment_samples,
            "message": f"Voice profile successfully enrolled with {len(extracted_embeddings)} recording sample(s).",
        }
    except HTTPException:
        raise
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
    num_genuine: int = 100
    num_imposter: int = 100
    enable_split: bool = True
    seed: int = 42


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
    """Executes empirical verification benchmarking over real audio file pairs.

    Utilizes an in-memory embedding cache so audio files appearing in multiple trial pairs
    do not trigger redundant model inferences.
    """
    adapter = LibriSpeechAdapter(req.path)
    summary = adapter.validate()
    if not summary.is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid dataset: {summary.message}",
        )

    model = get_model()

    # Embedding cache: maps canonical audio file path -> 1D embedding tensor
    embedding_cache: Dict[str, torch.Tensor] = {}

    def get_embedding(path: Path) -> torch.Tensor:
        canonical_key = str(path.resolve())
        if canonical_key in embedding_cache:
            return embedding_cache[canonical_key]
        audio = load_and_validate_audio(path, min_duration_sec=0.5)
        emb = model.extract_embedding(audio.waveform)
        embedding_cache[canonical_key] = emb
        return emb

    def _evaluate_batch(
        batch: List[TrialPair],
    ) -> tuple[List[EvaluatedTrial], List[dict]]:
        evaluated: List[EvaluatedTrial] = []
        skipped: List[dict] = []
        for t in batch:
            try:
                emb_a = get_embedding(t.path_a)
                emb_b = get_embedding(t.path_b)
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
            seed=req.seed,
        )
        test_seed = req.seed + 58  # Independent reproducible seed for held-out evaluation
        test_trials = adapter.generate_trials(
            num_genuine=req.num_genuine,
            num_imposter=req.num_imposter,
            speaker_subset=eval_speakers,
            seed=test_seed,
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

        cal_gen_count = sum(1 for t in cal_eval if t.is_genuine)
        cal_imp_count = sum(1 for t in cal_eval if not t.is_genuine)
        test_gen_count = sum(1 for t in test_eval if t.is_genuine)
        test_imp_count = sum(1 for t in test_eval if not t.is_genuine)

        return {
            "success": True,
            "protocol": "calibration_split",
            "calibrated_threshold": float(calibrated_threshold),
            "test_far": float(test_metric.far),
            "test_frr": float(test_metric.frr),
            "test_accuracy": float(test_metric.accuracy),
            "total_trials": len(cal_eval) + len(test_eval),
            "trial_breakdown": {
                "requested_per_partition": {"genuine": req.num_genuine, "imposter": req.num_imposter},
                "calibration_evaluated": {"genuine": cal_gen_count, "imposter": cal_imp_count},
                "test_evaluated": {"genuine": test_gen_count, "imposter": test_imp_count},
            },
            "calibration_seed": req.seed,
            "evaluation_seed": test_seed,
            "seeds": {"calibration_seed": req.seed, "evaluation_seed": test_seed},
            "cache_stats": {"unique_recordings_cached": len(embedding_cache)},
            "skipped_trials": cal_skip + test_skip,
            "note": split_note,
            "disclaimer": "Evaluated on clean audiobook speech (LibriSpeech test-clean). Does not establish performance on telephone banking audio.",
        }

    else:
        # Protocol Branch 2: Development Preview (Unsplit)
        trials = adapter.generate_trials(
            num_genuine=req.num_genuine,
            num_imposter=req.num_imposter,
            speaker_subset=None,
            seed=req.seed,
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

        gen_count = sum(1 for t in evaluated if t.is_genuine)
        imp_count = sum(1 for t in evaluated if not t.is_genuine)

        return {
            "success": True,
            "protocol": "unsplit_dev_preview",
            "calibrated_threshold": float(operating_threshold),
            "test_far": float(eer_metric.far if eer_metric else 0.0),
            "test_frr": float(eer_metric.frr if eer_metric else 0.0),
            "test_accuracy": float(eer_metric.accuracy if eer_metric else 0.0),
            "total_trials": len(evaluated),
            "trial_breakdown": {
                "requested": {"genuine": req.num_genuine, "imposter": req.num_imposter},
                "evaluated": {"genuine": gen_count, "imposter": imp_count},
            },
            "seeds": {"seed": req.seed},
            "cache_stats": {"unique_recordings_cached": len(embedding_cache)},
            "skipped_trials": skipped,
            "note": "Development Preview Mode: Evaluated across available data without disjoint partition.",
            "disclaimer": "Evaluated on clean audiobook speech (LibriSpeech test-clean). Does not establish performance on telephone banking audio.",
        }
