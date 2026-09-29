# Customer Voice Authentication (Educational Prototype)

> [!WARNING]
> **ACADEMIC & INTERNSHIP PROTOTYPE ONLY**  
> This project is an educational, proof-of-concept prototype for 1:1 voice verification. **It is NOT a production banking security system and must NOT be used to authenticate real bank customers or make financial-access decisions.**

---

## 1. Project Overview & Objectives

Customer Voice Authentication is an internship-level prototype demonstrating 1:1 biometric speaker verification in a simulated banking environment. It compares a candidate audio recording with an enrolled speaker profile to determine whether the two utterances may originate from the same speaker.

### Key Objectives:
- **Modular 1:1 Verification Engine**: Extract fixed-dimensional speaker embeddings using SpeechBrain's pretrained ECAPA-TDNN architecture and compare them via cosine similarity.
- **Modern 21st.dev UI & Local Web App**: Clean, high-contrast, dark-mode single-page application built with FastAPI and vanilla HTML/CSS/JavaScript (no Streamlit, no React, no Node.js build system).
- **Native Browser Microphone Recording**: Custom Web Audio API recorder that encodes 16-bit PCM WAV at 16 kHz directly client-side, allowing recording without requiring system FFmpeg on Windows.
- **Volunteer Demo Mode**: Enable safe, consent-driven voice enrollment and verification with full local data privacy and instant data purging controls.
- **LibriSpeech Evaluation Mode**: In-situ benchmarking on external LibriSpeech audio, with a disciplined calibration vs. independent test split protocol, reporting empirical False Accept Rate (FAR), False Reject Rate (FRR), accuracy, and threshold trade-offs.
- **HCL Dataset Readiness**: Decouple dataset loading through an abstract adapter interface so an official HCL dataset adapter can be integrated seamlessly once requirements are known.

---

## 2. Architecture & File Structure

```text
Voice-Verification/
├── .gitignore                   # Strict exclusion of data, recordings, embeddings, caches, venv
├── requirements.txt             # Dependencies (PyTorch, Torchaudio, SpeechBrain, FastAPI, Uvicorn)
├── pyproject.toml               # Modern build configuration and package metadata
├── start.bat                    # Windows 1-click launcher (starts server & opens browser)
├── start.ps1                    # PowerShell 1-click launcher
├── README.md                    # Project documentation and run guide
├── docs/
│   └── PROJECT_SYNOPSIS.md      # Formal internship project synopsis
├── src/
│   ├── __init__.py
│   ├── server.py                # FastAPI backend app, REST endpoints, and static file mount
│   ├── audio.py                 # Audio ingestion, validation (format/length), 16 kHz mono conversion
│   ├── model.py                 # Pretrained ECAPA-TDNN model wrapper (frozen weights)
│   ├── verification.py          # Cosine similarity computation & threshold decision logic
│   ├── storage.py               # Local volunteer profiles, metadata, and deletion routines
│   ├── dataset_eval.py          # External LibriSpeech adapter & trial pair generator
│   ├── metrics.py               # Empirical FAR, FRR, and EER calculations
│   └── static/                  # 21st.dev inspired modern frontend
│       ├── index.html           # Semantic HTML5 layout with 2 tabs
│       ├── css/
│       │   └── styles.css       # Dark glassmorphism, radial glow accents, metrics cards
│       └── js/
│           ├── recorder.js      # Pure client-side 16 kHz PCM WAV encoder & waveform canvas
│           └── app.js           # Frontend controller for REST API interactions
└── tests/
    ├── test_api.py              # FastAPI endpoint tests using TestClient
    └── test_components.py       # Core audio, verification, storage, and metrics unit tests
```

---

## 3. Quick Start on Windows

### 1. Initial Environment Setup (First time only)
Double-click `setup.bat` in the project root (or run `.\setup.bat` in PowerShell). This script:
- Verifies Python 3.10 is installed on the system.
- Checks and creates `.venv` (preserving any existing environment without destructive deletion).
- Automatically installs and upgrades dependencies from `requirements.txt` and `pyproject.toml`.

### 2. Launch the Application
Double-click `start.bat` in the project root, or execute from PowerShell:
```powershell
.\start.bat
```
This launcher:
1. Validates the `.venv` Python virtual environment.
2. Automatically launches your default web browser to `http://127.0.0.1:8000`.
3. Runs the local FastAPI server with Uvicorn.

Alternatively, for PowerShell:
```powershell
.\start.ps1
```

---

## 4. Manual Installation & Setup Instructions

### Prerequisites
- Python 3.10 (Recommended for prebuilt PyTorch and SpeechBrain wheel compatibility)
- Git

### Step-by-Step Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/ishanp-exe/Voice-Verification.git
   cd Voice-Verification
   ```

2. **Create and activate a virtual environment**:
   - **Windows (PowerShell)**:
     ```powershell
     py -3.10 -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - **Linux / macOS**:
     ```bash
     python3.10 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install dependencies & project package**:
   ```powershell
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install -e .
   ```

4. **Start the application manually**:
   ```powershell
   .\.venv\Scripts\uvicorn src.server:app --host 127.0.0.1 --port 8000
   ```
   Open your browser to `http://127.0.0.1:8000`.

---

## 5. Usage Guide

### Mode 1: Volunteer Demo Mode
Designed for interactive, privacy-preserving testing with volunteers:
1. **Informed Consent**: Review the privacy notice and check the consent box to activate controls.
2. **Speaker Enrollment**:
   - Generate a random Demo ID (e.g., `VOL-7281`).
   - Use the **Browser Microphone** to record live audio (with real-time waveform visualizer), or upload an audio file (`.wav`, `.flac`, `.mp3`, `.ogg`) of at least 1.0 second duration.
   - Click **Enroll Voice Profile**. The system extracts and stores a 192-dimensional vector. *Note: Enrolling does not train or modify the neural network.*
3. **1:1 Verification**:
   - Select the enrolled Demo ID from the dropdown.
   - Record or upload a new candidate audio sample.
   - Adjust the **Decision Threshold** slider.
   - Click **Verify Candidate Voice** to view the computed cosine similarity score, threshold pin, decision margin, and match verdict.
4. **Data Purging**:
   - Under **Enrolled Profiles & Data Governance**, click **Purge** to permanently remove all stored representations and metadata for a profile from disk.

### Mode 2: LibriSpeech Evaluation Mode
Designed for benchmarking verification error metrics on external corpus data:
1. **Temporary Dataset Notice**: LibriSpeech contains clean audiobook narrations and does not represent telephone banking audio. The official dataset will be provided by HCL later.
2. **Directory Selection**:
   - Provide the path to your extracted `LibriSpeech` or `test-clean` folder located **outside** this repository (e.g., `C:\Users\ishan\Downloads\LibriSpeech\test-clean`).
   - The app reads files in-place and **will not move or copy** the dataset.
3. **Trial Configuration & Disciplined Protocol**:
   - Specify the desired count of Genuine (same-speaker) and Imposter (different-speaker) trial pairs.
   - **Calibration & Test Split**: When enabled on a dataset with sufficient speakers ($\ge 6$ speakers with $\ge 4$ multi-sample speakers), the system partitions data into disjoint sets:
     - **Stage 1 (Calibration)**: The operating threshold $\tau_{cal}$ is determined strictly on the calibration partition where $|FAR - FRR|$ is minimized.
     - **Stage 2 (Held-Out Test Evaluation)**: Unbiased final performance metrics (Test FAR, Test FRR, Test Accuracy) are measured on the unseen evaluation partition using $\tau_{cal}$.
     - *Methodological Discipline*: The system explicitly avoids calculating an "EER threshold" on the evaluation set itself, preventing optimistic post-hoc test bias.
   - **Development Preview**: If the dataset has too few speakers or the split is disabled, the system clearly labels output as a Development Preview across available trials.
   - **Transparent Error Reporting**: Any unreadable or corrupted audio trials are logged and displayed with filenames and error reasons rather than silently skipped.

---

## 6. Privacy & Data Handling Practices
- **Strictly Local Processing**: Audio files and embeddings remain on the local machine. No data is transmitted to cloud services or external APIs.
- **Minimal Retention**: By default, raw audio is discarded immediately following embedding extraction. Only mathematical embedding vectors and non-PII metadata (`metadata.json`) are stored.
- **Git Safety**: The `data/` directory, model cache, audio files, and virtual environments are explicitly excluded via `.gitignore` to prevent accidental commits of biometric data.
- **Instant Deletion**: Volunteers have full control to permanently remove their enrolled templates at any time.

---

## 7. Understanding Scores & Thresholds
- **Cosine Similarity is NOT a Probability**: Cosine similarity measures geometric alignment in $[-1.0, 1.0]$. A score of `0.65` does not represent a "65% chance of being the same person."
- **Configurable Decision Threshold**: Different security postures require different operating points. High thresholds minimize False Accepts (imposter breach), while low thresholds minimize False Rejects (customer friction). No threshold should be considered "typical" without empirical calibration on target channel audio.

---

## 8. Limitations & Scope Boundaries
- **Prototype Status**: Designed exclusively for internship-level education and research.
- **No Anti-Spoofing / Presentation Attack Detection (PAD)**: Vulnerable to replays, text-to-speech synthesis, and deepfakes.
- **No 1:N Identification**: Evaluates one-to-one identity claims only.
- **Domain Discrepancy**: Pretrained models trained on studio/audiobook audio may experience reduced performance on narrow-band (8 kHz) telephony audio.
- **HCL Dataset**: The official project dataset will be delivered by HCL later. No assumptions about HCL data structure or licensing are made in this release.

---

## 9. Running Automated Tests

To run the automated unit and API test suite from the repository root in PowerShell:
```powershell
.\.venv\Scripts\pytest -v
```
All 28 tests utilize mocked embedding outputs to verify code paths, validation logic, metrics formulas, REST endpoints, and storage operations in test isolation without requiring GPU hardware or Internet access.