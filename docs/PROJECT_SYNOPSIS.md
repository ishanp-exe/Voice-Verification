# Formal Project Synopsis

## 1. Project Title
**Customer Voice Authentication: An Educational Prototype for Biometric Speaker Verification in Banking Workflows**

---

## 2. Abstract
Customer identity verification is a cornerstone of modern financial services, where conventional knowledge-based authentication methods (such as passwords, OTPs, and security questions) increasingly suffer from vulnerabilities to phishing, credential stuffing, and social engineering. Voice biometrics offers an intuitive, friction-reducing modality for speaker verification. This project details the design and implementation of an educational, proof-of-concept voice authentication system modeled for a simulated banking context.

Engineered as an internship-level learning prototype, the system leverages deep representation learning via SpeechBrain's pretrained ECAPA-TDNN architecture, executing within a PyTorch/Torchaudio framework, served by a local FastAPI backend, and controlled through a modern 21st.dev-inspired vanilla web interface. The architecture extracts fixed-dimensional speaker embeddings from acoustic utterances and computes pairwise cosine similarity against an operational decision threshold. The prototype implements two core workflows: an in-person, consent-governed **Volunteer Demo Mode** for live speaker enrollment (via browser microphone or file upload), verification, and on-demand data purging, and an offline **LibriSpeech Evaluation Mode** for measuring biometric error curves across genuine and imposter trial pairs. 

**Critical Boundary**: This project is strictly an internship-level academic prototype; it does not constitute a production banking security system and must **never** be deployed to authenticate real banking customers or authorize financial transactions.

---

## 3. Problem Statement
Remote banking channels (including telephone customer support, interactive voice response units, and mobile applications) require rapid and reliable identity confirmation. Knowledge-based credentials can be compromised, shared, or coerced, while hardware tokens introduce operational friction. Biometric voice verification provides a potential secondary layer of authentication by assessing individual vocal characteristics.

However, studying and integrating speaker-embedding pipelines involves significant technical and conceptual challenges:
- Establishing a robust acoustic preprocessing pipeline that converts variable-duration audio into standardized, noise-resistant speaker embeddings.
- Formulating an interpretable verification decision framework based on metric vector geometry rather than misunderstood "probability" percentages.
- Characterizing the operational trade-offs between False Accept Rate (FAR) and False Reject Rate (FRR) across differing decision thresholds.
- Enforcing strict privacy and data governance standards when capturing and persisting human biometric data.

---

## 4. Objectives
The key objectives of this project are:
1. **Develop an Educational 1:1 Verification Engine**: Build a modular Python application leveraging PyTorch, Torchaudio, and SpeechBrain’s pretrained ECAPA-TDNN model to extract speaker embeddings and perform cosine similarity comparisons.
2. **Implement an Intuitive, Responsive Web Interface**: Provide a polished, dark-mode Streamlit application (styled with 21st.dev design principles including hero spotlight, glassmorphic stat cards, and scroll indicator) that visualizes similarity scores alongside user-adjustable decision thresholds, outputting transparent match/no-match decisions.
3. **Establish a Safe Volunteer Demonstration Workflow**: Allow consenting volunteers to enroll via single or multi-sample browser audio or uploaded files under randomized demo IDs, test candidate samples, inspect stored metadata, and permanently purge their data from the local machine.
4. **Build a Standardized Evaluation Harness**: Provide an evaluation adapter for an external LibriSpeech `test-clean` dataset to programmatically construct genuine (same-speaker) and imposter (different-speaker) trial pairs using the identical verification pipeline with in-memory embedding caching.
5. **Empirical Performance Measurement**: Calculate and graph empirical verification metrics (FAR, FRR, accuracy, and empirical EER) strictly from actual trial runs without inventing or fabricating results.
6. **Modular Architecture for Future Domain Datasets**: Isolate dataset ingestion from verification logic to allow straightforward integration of specialized enterprise datasets once schema, licensing, and handling guidelines are provided.

---

## 5. Proposed System
The system is constructed with a decoupled, modular architecture where both user-driven demonstrations and batch evaluations utilize the identical core verification pipeline.

```text
+---------------------------------------------------------------------------------+
|                         Streamlit Modern Web Application                        |
|   +------------------------------------+------------------------------------+   |
|   |        Volunteer Demo Mode         |     LibriSpeech Evaluation Mode    |   |
|   | - Informed consent verification    | - External path selection (in-situ)|   |
|   | - Multi-take mic (WAV) / upload    | - Speaker trial generation         |   |
|   | - Candidate verification           | - Separate calibration/eval split  |   |
|   | - Complete profile & data deletion | - In-memory embedding cache        |   |
|   +------------------------------------+------------------------------------+   |
+----------------------------------------+----------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|                            Unified Verification Core                            |
|   1. Audio Ingestion & Resampling (src/audio.py):                               |
|      - Mono conversion, validation (min 0.8s), resample to 16 kHz               |
|   2. Pretrained ECAPA-TDNN Embedding Extraction (src/model.py):                 |
|      - Frozen weights, 192-dim normalized vector; NO model fine-tuning         |
|   3. Multi-Take Embedding Combination (src/verification.py):                    |
|      - Element-wise arithmetic mean across takes with unit L2 normalization     |
|   4. Vector Geometry & Threshold Comparator (src/verification.py):              |
|      - Cosine similarity: cos(u, v) = (u . v) / (||u|| * ||v||)                 |
|      - Decision: Score >= Threshold -> MATCH; else NO MATCH                     |
+---------------------------------------------------------------------------------+
```

### Operational Modes:
- **Volunteer Demo Mode**: Prioritizes privacy and informed consent. Volunteers enroll under anonymized IDs (`VOL-XXXX`). The system stores the numerical embedding vector and minimal non-PII metadata. Raw audio is discarded by default. Users can inspect or permanently delete their profile and files at any point.
- **LibriSpeech Evaluation Mode**: Designed for offline experimental benchmarking. Analyzes external LibriSpeech `test-clean` audio in-place without copying files, generating balanced genuine and imposter trial pairs.

---

## 6. Methodology

1. **Acoustic Ingestion & Validation**:
   Audio streams (WAV, FLAC, OGG, MP3) are loaded via `soundfile` / `torchaudio`. The signal is checked for corruption, emptiness, and minimum duration (minimum 1.0s required). Multi-channel signals are averaged to mono, and the sample rate is standardized to 16,000 Hz.

2. **Embedding Extraction via Pretrained ECAPA-TDNN**:
   The waveform is forwarded through SpeechBrain’s pretrained `speechbrain/spkrec-ecapa-voxceleb` model. The network applies 1D dilated convolutions with Squeeze-and-Excitation channel attention and attentive statistical pooling, producing a 192-dimensional vector. The vector is L2-normalized onto the unit hypersphere:
   $$\hat{\mathbf{e}} = \frac{\mathbf{e}}{\|\mathbf{e}\|_2}$$
   *Note*: Enrolling a speaker stores this mathematical vector; it does not train or fine-tune neural network weights.

3. **Metric Comparison & Threshold Decision**:
   Cosine similarity evaluates the directional alignment of candidate embedding $\mathbf{v}$ against enrolled embedding $\mathbf{u}$:
   $$\text{Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$
   If $\text{Similarity} \ge \tau$, the candidate identity is accepted (Match); otherwise, it is rejected (No Match). The interface highlights that similarity is a geometric metric, not an identity probability.

4. **Biometric Error Rate Formulation & Disciplined Evaluation Protocol**:
   Evaluation trials are classified into four empirical quadrants:
   - **True Accept (TA)**: Genuine pair with $\text{Score} \ge \tau$
   - **False Reject (FR)**: Genuine pair with $\text{Score} < \tau$
   - **False Accept (FA)**: Imposter pair with $\text{Score} \ge \tau$
   - **True Reject (TR)**: Imposter pair with $\text{Score} < \tau$

   Error rates are computed empirically across candidate threshold values:
   $$\text{FAR}(\tau) = \frac{\text{FA}}{\text{FA} + \text{TR}}, \quad \text{FRR}(\tau) = \frac{\text{FR}}{\text{FR} + \text{TA}}$$

   **Calibration & Test Independence Protocol**:
   - When dataset scale permits ($\ge 6$ speakers with balanced multi-sample speakers), speakers are partitioned into disjoint **Calibration** and **Evaluation** subsets.
   - The decision threshold $\tau_{cal}$ is determined strictly on the calibration trials by identifying the point minimizing $|\text{FAR}_{cal} - \text{FRR}_{cal}|$.
   - Final test performance ($\text{FAR}_{test}$, $\text{FRR}_{test}$, and accuracy) is evaluated on the unseen evaluation partition using $\tau_{cal}$.
   - To maintain experimental validity, no post-hoc "EER threshold" is tuned or claimed directly on the test set.
   - Any trials containing corrupt or unreadable audio are explicitly logged and reported rather than silently suppressed.

---

## 7. Tools and Technologies
- **Programming Language**: Python 3.10+
- **Deep Learning Framework**: PyTorch and Torchaudio
- **Pretrained Biometric Models**: SpeechBrain (`speechbrain/spkrec-ecapa-voxceleb`)
- **Web Interface & Local Server**: FastAPI with vanilla HTML5, CSS3, and modern JavaScript (no external frontend build systems)
- **Audio I/O & Signal Processing**: SoundFile, NumPy, SciPy, Web Audio API (in-browser 16 kHz PCM WAV encoding)
- **Data Analysis & Visualization**: Pandas, Matplotlib
- **Quality Assurance**: Pytest (utilizing mocked model embeddings for test isolation)

---

## 8. Dataset Strategy & Enterprise Dataset Roadmap

### Current Development Dataset: LibriSpeech test-clean
- **Source**: Public domain LibriVox audiobooks.
- **Role**: Development dataset for implementing and testing the trial generation and evaluation workflow.
- **Handling**: Stored externally to the repository; accessed in-place without automated copying.
- **Acoustic Reality Check**: LibriSpeech consists of clean, narrated audiobook recordings captured under quiet conditions. It **does not** represent telephony banking audio (which exhibits 8 kHz band-pass filtering, codec compression, packet loss, and ambient acoustic noise).

### Future Target Dataset: Enterprise Banking Audio
- Specialized domain datasets can be integrated in subsequent phases.
- **Strict Boundary**: This prototype makes zero assumptions regarding target domain datasets' directory structure, audio encoding, speaker labeling, licensing, or handling protocols.
- The dataset loading logic is isolated behind an abstract adapter interface (`BaseSpeakerDatasetAdapter`), ensuring custom adapters can be cleanly integrated once formal documentation and access terms are released.
- No external enterprise data will be processed until its specifications and governance requirements are established.

---

## 9. Expected Outcomes
Upon full execution and benchmark evaluation, the project delivers:
- A responsive, locally run web application demonstrating interactive speaker enrollment, verification, and privacy-preserving data purging.
- A functional batch evaluation harness that computes actual FAR, FRR, accuracy, and empirical EER curves on user-selected external trial sets.
- Complete separation of concerns across audio processing, model inference, verification, local storage, dataset adapters, and metrics computation.
- Comprehensive unit test coverage verifying error handling, mathematical correctness, and profile deletion.

---

## 10. Limitations
1. **Prototype Status**: This system is strictly an academic learning tool and is **not suitable for production banking authentication or financial-access decisions**.
2. **Lack of Anti-Spoofing & Liveness**: The prototype does not include Presentation Attack Detection (PAD). It cannot detect pre-recorded replays, synthesized voices, or deepfake voice clones.
3. **No 1:N Identification**: The system is restricted to 1:1 verification (claim verification) and does not support open-set 1:N speaker identification.
4. **Domain Mismatch**: Pretrained models trained on high-bandwidth audio experience degraded accuracy when exposed to low-bandwidth telephone networks (8 kHz G.711 codecs).
5. **Score Non-Probabilistic Nature**: Raw cosine similarity values depend on embedding space characteristics and cannot be interpreted as calibrated posterior probabilities without statistical calibration (e.g., PLDA or logistic regression).

---

## 11. Future Scope
Subsequent project phases may investigate:
- **Presentation Attack Detection (PAD)**: Integrating anti-spoofing models (e.g. ASVspoof challenge baselines) to reject synthetic or replayed audio.
- **Liveness Challenge-Response**: Incorporating dynamic spoken passcodes or prompt-response mechanisms to verify speaker presence.
- **Acoustic Channel Adaptation**: Simulating telephony band-pass filters and testing noise-robust front-ends.
- **Domain Dataset Integration**: Developing specialized adapters when target enterprise data and guidelines are provided.
- **Score Calibration**: Fitting Probabilistic Linear Discriminant Analysis (PLDA) models to output log-likelihood ratios.

---

## 12. Conclusion
The Customer Voice Authentication prototype offers a structured, transparent exploration of biometric speaker verification. By leveraging SpeechBrain's ECAPA-TDNN model within an accessible FastAPI and modern vanilla web application, the project highlights key operational concepts—including embedding representation, metric comparison, decision boundaries, and error trade-offs. By prioritizing volunteer consent, local data residency, and clear architectural boundaries, the project provides a solid, responsible foundation for academic study and future dataset integration.
