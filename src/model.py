"""Pretrained speaker embedding model module.

Loads SpeechBrain's ECAPA-TDNN model and extracts normalized speaker embeddings.
Note: This system uses a frozen pretrained model. Speaker enrollment extracts and
stores an embedding representation for geometric comparison; it DOES NOT train or
fine-tune model weights.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Default model identifier and local cache directory
DEFAULT_MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
DEFAULT_CACHE_DIR = "speechbrain_cache/spkrec-ecapa-voxceleb"


class ModelLoadError(Exception):
    """Raised when the pretrained model cannot be downloaded or initialized."""
    pass


class SpeakerEmbeddingModel:
    """Wrapper around SpeechBrain's pretrained ECAPA-TDNN model for speaker embedding extraction.

    Important Educational Note:
        Enrolling a speaker stores a fixed-dimensional embedding vector representing
        acoustic characteristics. It does NOT update, train, or fine-tune model parameters.
    """

    def __init__(self, classifier: any, model_name: str = DEFAULT_MODEL_SOURCE):
        self.classifier = classifier
        self.model_name = model_name

    @classmethod
    def load(
        cls,
        source: str = DEFAULT_MODEL_SOURCE,
        savedir: Optional[str] = DEFAULT_CACHE_DIR,
        run_opts: Optional[dict] = None,
    ) -> SpeakerEmbeddingModel:
        """Loads the pretrained ECAPA-TDNN model from SpeechBrain / Hugging Face Hub.

        Args:
            source: Hugging Face repo ID or local checkpoint path.
            savedir: Local directory for caching model weights.
            run_opts: Optional dictionary of runtime options (e.g. {"device": "cpu"}).

        Returns:
            SpeakerEmbeddingModel instance.

        Raises:
            ModelLoadError: With explicit diagnostic instructions if loading fails.
        """
        try:
            # Import SpeechBrain lazily to provide clear error if dependencies are missing
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError as e:
            raise ModelLoadError(
                "SpeechBrain is not installed or could not be imported. "
                "Please ensure dependencies are installed via 'pip install -r requirements.txt'."
            ) from e

        if run_opts is None:
            # Default to CPU for deterministic and portable execution
            device = "cuda" if torch.cuda.is_available() else "cpu"
            run_opts = {"device": device}

        # Resolve cache dir relative to project root
        cache_path = Path(savedir) if savedir else Path("speechbrain_cache")
        cache_path.mkdir(parents=True, exist_ok=True)

        try:
            from speechbrain.utils.fetching import LocalStrategy
            # Use LocalStrategy.COPY on Windows to avoid symlink privilege errors (WinError 1314)
            local_strategy = LocalStrategy.COPY if os.name == "nt" else LocalStrategy.SYMLINK

            logger.info("Loading pretrained speaker model from '%s' into '%s'...", source, cache_path)
            classifier = EncoderClassifier.from_hparams(
                source=source,
                savedir=str(cache_path),
                run_opts=run_opts,
                local_strategy=local_strategy,
            )
            return cls(classifier=classifier, model_name=source)
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load pretrained model '{source}'.\n\n"
                "Troubleshooting Steps:\n"
                "1. Check your Internet connection to allow downloading model weights from Hugging Face.\n"
                "2. Verify write permissions for the local cache directory: "
                f"{cache_path.resolve()}\n"
                "3. If Hugging Face Hub is blocked or offline, ensure local model files are placed in "
                f"'{cache_path}'\n"
                f"Underlying error: {type(e).__name__}: {e}"
            ) from e

    def extract_embedding(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extracts a normalized 1D speaker embedding vector from a waveform tensor.

        Args:
            waveform: Tensor of shape [1, time] or [time] sampled at 16 kHz.

        Returns:
            1D torch.Tensor of normalized embeddings (192 dimensions for ECAPA-TDNN).
        """
        # Ensure shape [batch, time]
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.dim() > 2:
            waveform = waveform.squeeze()
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)

        with torch.no_grad():
            # SpeechBrain's encode_batch outputs [batch, 1, embedding_dim]
            embeddings = self.classifier.encode_batch(waveform)
            # Squeeze down to 1D vector [embedding_dim]
            embedding_1d = embeddings.squeeze()
            # L2-normalize vector to unit sphere
            normalized = F.normalize(embedding_1d, p=2, dim=0)

        return normalized.cpu()
