# Graph Report - Voice-Verification  (2026-10-22)

## Corpus Check
- Corpus is ~31,532 words - fits in a single context window. You may not need a graph.

## Summary
- 395 nodes · 796 edges · 19 communities (13 shown, 6 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 76 edges (avg confidence: 0.89)
- Token cost: 187,677 input · 0 output

## Community Hubs (Navigation)
- Benchmark & Trial Scoring
- FastAPI Server & Audio IO
- Streamlit App & Dataset Eval
- Volunteer Storage
- Project Concepts & Docs
- API Tests
- Dataset Cleaning & EDA
- ECAPA Embedding Model
- Environment Setup Check
- Web UI Frontend Logic
- Hero Carousel Component
- Dataset Adapter Interface
- Browser WAV Recorder
- DET Curve Results
- Score Distribution Results
- Repo Review Tests
- Project Metadata

## God Nodes (most connected - your core abstractions)
1. `load_and_validate_audio()` - 24 edges
2. `EvaluatedTrial` - 22 edges
3. `run()` - 20 edges
4. `LibriSpeechAdapter` - 20 edges
5. `VolunteerStorage` - 20 edges
6. `compute_cosine_similarity()` - 18 edges
7. `run_evaluation()` - 17 edges
8. `combine_enrollment_embeddings()` - 16 edges
9. `compute_metric_sweep()` - 15 edges
10. `compute_metrics_at_threshold()` - 14 edges

## Surprising Connections (you probably didn't know these)
- `Calibration & Test Independence Protocol` --semantically_similar_to--> `Calibration vs Test Speaker Split Protocol`  [INFERRED] [semantically similar]
  docs/PROJECT_SYNOPSIS.md → README.md
- `eval_batch()` --uses--> `TrialPair`  [INFERRED]
  app.py → src/dataset_eval.py
- `eval_batch()` --uses--> `EvaluatedTrial`  [INFERRED]
  app.py → src/metrics.py
- `run()` --uses--> `SpeakerEmbeddingModel`  [INFERRED]
  scripts/run_benchmark.py → src/model.py
- `temp_storage()` --uses--> `VolunteerStorage`  [INFERRED]
  tests/test_api.py → src/storage.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Speaker Verification Pipeline (embed, combine, score, threshold)** — readme_ecapa_tdnn, docs_project_synopsis_l2_normalized_embedding, readme_multi_sample_enrollment, readme_cosine_similarity, readme_decision_threshold [EXTRACTED 1.00]
- **Disciplined LibriSpeech Evaluation Protocol** — readme_dataset_validation, readme_calibration_test_split, docs_project_synopsis_error_quadrants, readme_far_frr_metrics, readme_benchmark_results [INFERRED 0.85]
- **Volunteer Consent-Enroll-Verify-Purge Workflow** — src_static_index_consent_gate, src_static_index_enrollment_card, src_static_index_verification_card, src_static_index_governance_table [EXTRACTED 1.00]
- **FAR/FRR trade-off at calibrated threshold** — docs_results_det_curve_far, docs_results_det_curve_frr, docs_results_det_curve_calibrated_threshold [EXTRACTED 1.00]
- **Threshold separates genuine vs imposter score distributions** — docs_results_score_distribution_threshold, docs_results_score_distribution_genuine_scores, docs_results_score_distribution_imposter_scores [INFERRED 0.85]

## Communities (19 total, 6 thin omitted)

### Community 0 - "Benchmark & Trial Scoring"
Cohesion: 0.08
Nodes (24): main(), _plot_det(), _plot_scores(), run(), embed(), score(), summarize(), _write_trials_csv() (+16 more)

### Community 1 - "FastAPI Server & Audio IO"
Cohesion: 0.06
Nodes (15): AudioData, AudioValidationError, load_and_validate_audio(), DatasetPathRequest, delete_volunteer(), enroll_volunteer(), EvaluationRunRequest, get_model() (+7 more)

### Community 2 - "Streamlit App & Dataset Eval"
Cohesion: 0.07
Nodes (8): eval_batch(), get_cached_emb(), combine_enrollment_embeddings(), compute_cosine_similarity(), VerificationResult, verify_speakers(), mock_model(), TestVerification

### Community 3 - "Volunteer Storage"
Cohesion: 0.08
Nodes (7): VolunteerMetadata, VolunteerStorage, TestVolunteerStorage, test_consent_withdrawal_apptest_lifecycle(), test_duplicate_id_shows_server_wording(), test_streamlit_app_loads_successfully(), test_volunteer_enrollment_governance_and_deletion_lifecycle()

### Community 4 - "Project Concepts & Docs"
Cohesion: 0.07
Nodes (32): CPU-only PyTorch Install Step, CI pytest Job (tests workflow), Enterprise Banking Dataset Roadmap, Biometric Error Quadrants (TA/FR/FA/TR), 192-dim L2-Normalized Speaker Embedding, PLDA Score Calibration (future scope), Presentation Attack Detection (PAD) / Anti-Spoofing, Telephony Domain Mismatch (8 kHz codecs) (+24 more)

### Community 5 - "API Tests"
Cohesion: 0.09
Nodes (6): client(), create_test_wav_bytes(), temp_storage(), TestEvaluationEndpoints, TestHealthAndStatic, TestVolunteerWorkflows

### Community 6 - "Dataset Cleaning & EDA"
Cohesion: 0.15
Nodes (6): AudioRecord, compute_sha256(), main(), validate_dataset(), canvas(), font()

### Community 7 - "ECAPA Embedding Model"
Cohesion: 0.13
Nodes (4): get_model(), ModelLoadError, SpeakerEmbeddingModel, TestModelInference

### Community 8 - "Environment Setup Check"
Cohesion: 0.20
Nodes (5): check_data_directory_access(), check_python_version(), check_required_packages(), check_virtual_env(), main()

### Community 9 - "Web UI Frontend Logic"
Cohesion: 0.24
Nodes (8): checkHealth(), deleteVolunteer(), fetchVolunteers(), renderEvaluationResults(), renderStagedSamples(), renderVolunteersList(), showToast(), updateEnrollSubmitState()

### Community 10 - "Hero Carousel Component"
Cohesion: 0.24
Nodes (3): get_hero_carousel_html(), render_hero_carousel(), TestHeroCarousel

### Community 13 - "DET Curve Results"
Cohesion: 0.40
Nodes (6): DET Curve (held-out test partition), Calibrated Threshold tau_cal = 0.312 (FAR 1.20%, FRR 3.40%), Equal Error Rate (FAR = FRR line), False Accept Rate, False Reject Rate, Held-out Test Partition

### Community 14 - "Score Distribution Results"
Cohesion: 0.53
Nodes (6): Held-out Test Score Distributions (chart), Cosine Similarity Score, Genuine Trial Scores (n=500), Held-out Test Partition (LibriSpeech), Imposter Trial Scores (n=500), Calibrated Threshold tau_cal = 0.312

## Ambiguous Edges - Review These
- `Configurable Decision Threshold` → `Threshold Slider (default 0.25)`  [AMBIGUOUS]
  src/static/index.html · relation: implements

## Knowledge Gaps
- **16 isolated node(s):** `customer-voice-authentication`, `Biometric Error Quadrants (TA/FR/FA/TR)`, `192-dim L2-Normalized Speaker Embedding`, `Presentation Attack Detection (PAD) / Anti-Spoofing`, `PLDA Score Calibration (future scope)` (+11 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 146 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **6 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `VolunteerStorage` connect `Volunteer Storage` to `Benchmark & Trial Scoring`, `FastAPI Server & Audio IO`, `Streamlit App & Dataset Eval`, `API Tests`?**
  _High betweenness centrality (0.092) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `EvaluatedTrial` (e.g. with `eval_batch()` and `_plot_det()`) actually correct?**
  _`EvaluatedTrial` has 6 INFERRED edges - model-reasoned connections that need verification._
- **What connects `customer-voice-authentication`, `Biometric Error Quadrants (TA/FR/FA/TR)`, `192-dim L2-Normalized Speaker Embedding` to the rest of the system?**
  _16 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Benchmark & Trial Scoring` be split into smaller, more focused modules?**
  _Cohesion score 0.08415300546448087 - nodes in this community are weakly interconnected._
- **Why does `LibriSpeechAdapter` connect `Benchmark & Trial Scoring` to `FastAPI Server & Audio IO`, `Streamlit App & Dataset Eval`, `Dataset Adapter Interface`?**
  _High betweenness centrality (0.059) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `run()` (e.g. with `.generate_trials()` and `.split_speakers_for_calibration()`) actually correct?**
  _`run()` has 4 INFERRED edges - model-reasoned connections that need verification._

### Low-confidence Hints
_AMBIGUOUS edges — the extractor was unsure. Verify before acting on these._

- **What is the exact relationship between `Configurable Decision Threshold` and `Threshold Slider (default 0.25)`?**
  _Edge tagged AMBIGUOUS (relation: implements) - confidence is low._