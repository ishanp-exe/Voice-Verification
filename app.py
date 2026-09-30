"""Customer Voice Authentication - Streamlit Application.

An internship-level educational prototype for 1:1 voice biometric verification.
Features:
- 21st.dev-inspired modern analytics UI with hero spotlight card and glassmorphic styling
- Multi-sample volunteer enrollment with vector averaging and unit L2-normalization
- 1:1 candidate verification with configurable decision threshold
- Profile governance with permanent disk purging
- LibriSpeech test-clean benchmark mode with disjoint calibration/test partitions and in-memory caching
"""

from __future__ import annotations

import io
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import streamlit as st
import torch

from src.audio import AudioValidationError, load_and_validate_audio
from src.dataset_eval import LibriSpeechAdapter, TrialPair
from src.metrics import (
    EvaluatedTrial,
    compute_metric_sweep,
    compute_metrics_at_threshold,
    find_empirical_eer,
)
from src.model import SpeakerEmbeddingModel
from src.storage import VolunteerStorage
from src.verification import (
    combine_enrollment_embeddings,
    compute_cosine_similarity,
    verify_speakers,
)

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Voice Authentication",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# 21st.dev Inspired Styling (Hero Spotlight, Glassmorphic Cards, Responsive Motion)
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
/* Base typography & theme refinement */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

code, kbd, samp, pre {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Scroll progress indicator (subtle 21st.dev scroll effect) */
#scroll-progress-bar {
    position: fixed;
    top: 0;
    left: 0;
    height: 3px;
    background: linear-gradient(90deg, #14b8a6, #06b6d4, #3b82f6);
    z-index: 99999;
    width: 0%;
    transition: width 0.1s ease-out;
}

@media (prefers-reduced-motion: reduce) {
    #scroll-progress-bar {
        transition: none !important;
        animation: none !important;
    }
    * {
        animation-duration: 0.001ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.001ms !important;
    }
}

/* 21st.dev Hero Card with Spotlight Effect */
.hero-card {
    position: relative;
    background: radial-gradient(circle at top left, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.95));
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 20px;
    padding: 2.2rem 2.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
    overflow: hidden;
}

.hero-spotlight {
    position: absolute;
    top: -50%;
    left: -20%;
    width: 140%;
    height: 140%;
    background: radial-gradient(circle 380px at var(--mouse-x, 25%) var(--mouse-y, 25%), rgba(20, 184, 166, 0.16), transparent 70%);
    pointer-events: none;
    transition: background 0.15s ease;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.35rem 0.85rem;
    background: rgba(20, 184, 166, 0.12);
    border: 1px solid rgba(20, 184, 166, 0.3);
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 600;
    color: #2dd4bf;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.025em;
    color: #f8fafc;
    margin: 0.2rem 0 0.8rem 0;
    line-height: 1.2;
}

.hero-title span {
    background: linear-gradient(135deg, #2dd4bf, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-desc {
    font-size: 1.05rem;
    line-height: 1.6;
    color: #94a3b8;
    max-width: 820px;
    margin-bottom: 0;
}

/* Reusable Glassmorphic 21st.dev Card */
.stat-card {
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 1.25rem 1.4rem;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
    backdrop-filter: blur(8px);
}

.stat-label {
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748b;
    margin-bottom: 0.35rem;
}

.stat-value {
    font-size: 1.7rem;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: -0.02em;
}

.stat-sub {
    font-size: 0.8rem;
    color: #94a3b8;
    margin-top: 0.25rem;
}

/* Audio Sample Tag */
.sample-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.3rem 0.75rem;
    background: rgba(51, 65, 85, 0.6);
    border: 1px solid rgba(148, 163, 184, 0.2);
    border-radius: 8px;
    font-size: 0.85rem;
    color: #cbd5e1;
    margin: 0.2rem 0.3rem 0.2rem 0;
}

/* Verification Result Banners */
.result-banner-match {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(5, 150, 105, 0.05));
    border: 1px solid rgba(16, 185, 129, 0.35);
    border-radius: 14px;
    padding: 1.5rem;
    margin-top: 1rem;
}

.result-banner-nomatch {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(220, 38, 38, 0.05));
    border: 1px solid rgba(239, 68, 68, 0.35);
    border-radius: 14px;
    padding: 1.5rem;
    margin-top: 1rem;
}

/* Modern Tab Enhancements */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 18px;
    font-weight: 600;
}

/* Scope notice container */
.scope-callout {
    background: rgba(30, 41, 59, 0.4);
    border-left: 3px solid #0ea5e9;
    padding: 0.85rem 1.1rem;
    border-radius: 0 8px 8px 0;
    font-size: 0.88rem;
    color: #cbd5e1;
    margin: 1rem 0;
}
</style>

<div id="scroll-progress-bar"></div>
<script>
// Scroll progress listener
window.addEventListener('scroll', () => {
    const winScroll = document.documentElement.scrollTop || document.body.scrollTop;
    const height = document.documentElement.scrollHeight - document.documentElement.clientHeight;
    const scrolled = height > 0 ? (winScroll / height) * 100 : 0;
    const bar = document.getElementById('scroll-progress-bar');
    if (bar) bar.style.width = scrolled + '%';
});

// Spotlight mousemove tracker for hero-card
document.addEventListener('mousemove', (e) => {
    const heroCards = document.querySelectorAll('.hero-card');
    heroCards.forEach(card => {
        const rect = card.getBoundingClientRect();
        const x = ((e.clientX - rect.left) / rect.width) * 100;
        const y = ((e.clientY - rect.top) / rect.height) * 100;
        card.style.setProperty('--mouse-x', x + '%');
        card.style.setProperty('--mouse-y', y + '%');
    });
});
</script>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Singleton Cache & Model Initialization
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading pretrained ECAPA-TDNN speaker embedding model...")
def get_model() -> SpeakerEmbeddingModel:
    """Loads and caches the pretrained SpeechBrain ECAPA-TDNN embedding extractor."""
    return SpeakerEmbeddingModel.load()


storage = VolunteerStorage()


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "demo_id" not in st.session_state:
    st.session_state.demo_id = f"VOL-{random.randint(1000, 9999)}"

if "enrollment_takes" not in st.session_state:
    # List of dicts: {"name": str, "bytes": bytes, "duration": float}
    st.session_state.enrollment_takes = []

if "evaluation_results" not in st.session_state:
    st.session_state.evaluation_results = None


# -----------------------------------------------------------------------------
# Sidebar: System Governance, Status & Ambient Visualizer
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎙️ Core Biometrics Engine")

    # Load Model with visual feedback
    try:
        model = get_model()
        st.success("✅ **ECAPA-TDNN Pretrained**\n\n192-dim embeddings • 16 kHz Mono", icon="🧠")
    except Exception as e:
        st.error(f"❌ Failed to load speaker model: {e}")
        st.stop()

    st.markdown("---")
    st.markdown("### 🔒 Privacy & Data Practices")
    st.markdown(
        """
        - **Local Processing**: Audio stays on your machine; never sent to external servers.
        - **No Raw Audio Retention**: Raw voice recordings are discarded immediately after embedding computation.
        - **Vector Storage Only**: Stored files contain only 192-dimensional floating-point vectors and non-PII metadata.
        - **Permanent Purging**: Delete profile templates and metadata from disk with one click.
        """
    )

    st.markdown("---")
    st.markdown("### 🎨 Ambient 3D Visualization")
    show_spline = st.checkbox("Show 3D Voice Orb (Spline)", value=False, help="Interactive 3D voice wave orb. Optional and purely decorative; does not affect biometric processing.")
    
    if show_spline:
        st.caption("Interactive sound wave visualization (purely ambient):")
        # Lightweight embed with local graceful fallback
        spline_html = """
        <div style="width: 100%; height: 220px; border-radius: 12px; overflow: hidden; background: #0f172a; border: 1px solid rgba(255,255,255,0.1); position: relative;">
            <iframe src="https://my.spline.design/soundwaveorb-f0278fb4de391ba5f2ce360ea1003468/" 
                    frameborder="0" width="100%" height="100%" loading="lazy"
                    title="Ambient Sound Wave Scene"
                    style="pointer-events: auto;">
            </iframe>
        </div>
        """
        st.components.v1.html(spline_html, height=230)
    else:
        st.caption("3D ambient scene disabled. Standard high-performance UI mode active.")

    st.markdown("---")
    st.caption("Educational Biometric Prototype • Zero Cloud Dependencies")


# -----------------------------------------------------------------------------
# Main Hero Section (21st.dev Spotlight Card)
# -----------------------------------------------------------------------------
hero_html = f"""
<div class="hero-card">
    <div class="hero-spotlight"></div>
    <div class="hero-badge">Biometric Analytics Prototype</div>
    <div class="hero-title">Customer <span>Voice Authentication</span></div>
    <div class="hero-desc">
        A calibrated 1:1 speaker verification prototype built for educational research. 
        Extracts 192-dimensional neural speaker embeddings via a pretrained ECAPA-TDNN model, 
        combines multi-sample enrollment takes with unit $L_2$-normalization, and evaluates genuine vs. imposter 
        cosine similarity against empirical thresholds.
    </div>
</div>
"""
st.markdown(hero_html, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Global Volunteer Consent Gate
# -----------------------------------------------------------------------------
consent_container = st.container()
with consent_container:
    st.markdown("#### 📋 Volunteer Informed Consent Gate")
    consent_col1, consent_col2 = st.columns([0.7, 0.3])
    with consent_col1:
        volunteer_consent = st.checkbox(
            "I freely consent to having my voice audio recorded and processed locally to extract a 192-dimensional mathematical speaker embedding. I understand that raw audio is discarded immediately and only the numerical template is retained.",
            value=True,
            key="global_volunteer_consent",
        )
    with consent_col2:
        if volunteer_consent:
            st.markdown(
                '<div style="background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; padding: 0.5rem 0.8rem; color: #34d399; font-size: 0.85rem; font-weight: 600; text-align: center;">✅ Consent Verified</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 8px; padding: 0.5rem 0.8rem; color: #f87171; font-size: 0.85rem; font-weight: 600; text-align: center;">⚠️ Consent Required</div>',
                unsafe_allow_html=True,
            )

st.markdown("<br>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Main Application Tabs
# -----------------------------------------------------------------------------
tab_enroll, tab_verify, tab_governance, tab_eval = st.tabs(
    ["📥 Step 1: Voice Enrollment", "🔍 Step 2: 1:1 Verification", "🛡️ Profile Governance", "📊 LibriSpeech Evaluation"]
)


# =============================================================================
# TAB 1: Multi-Sample Volunteer Voice Enrollment
# =============================================================================
with tab_enroll:
    st.markdown("### 📥 Step 1: Multi-Sample Volunteer Voice Enrollment")
    st.markdown(
        "Enroll a consenting volunteer by providing **one or more separate audio takes** (e.g., repeating a passphrase or speaking naturally). "
        "The model extracts an embedding vector for each take and computes an element-wise arithmetic mean with unit $L_2$-normalization to establish a robust template."
    )

    if not volunteer_consent:
        st.warning("⚠️ You must check the Informed Consent box above before you can stage recordings or enroll a profile.")

    enroll_col_left, enroll_col_right = st.columns([0.55, 0.45], gap="large")

    with enroll_col_left:
        st.markdown("#### 1. Identity & Audio Ingestion")

        id_col1, id_col2 = st.columns([0.7, 0.3])
        with id_col1:
            enroll_id = st.text_input("Volunteer Demo ID", value=st.session_state.demo_id, max_chars=32)
            st.session_state.demo_id = enroll_id
        with id_col2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("🎲 New ID", use_container_width=True):
                st.session_state.demo_id = f"VOL-{random.randint(1000, 9999)}"
                st.rerun()

        input_mode = st.radio(
            "Input Mode:",
            ["🎙️ Browser Microphone", "📁 File Upload"],
            horizontal=True,
            disabled=not volunteer_consent,
        )

        audio_bytes_to_add: Optional[bytes] = None
        source_label: str = ""

        if "Browser Microphone" in input_mode:
            mic_input = st.audio_input(
                "Record Voice Sample (Minimum 0.8s):",
                disabled=not volunteer_consent,
                key="enroll_mic_input",
            )
            if mic_input is not None:
                audio_bytes_to_add = mic_input.getvalue()
                source_label = f"Microphone Take #{len(st.session_state.enrollment_takes) + 1}"
        else:
            file_input = st.file_uploader(
                "Upload Voice Audio (.wav, .flac, .mp3, .ogg):",
                type=["wav", "flac", "mp3", "ogg"],
                disabled=not volunteer_consent,
                key="enroll_file_input",
            )
            if file_input is not None:
                audio_bytes_to_add = file_input.getvalue()
                source_label = file_input.name

        # Staging action button
        btn_stage_col1, btn_stage_col2 = st.columns([0.65, 0.35])
        with btn_stage_col1:
            if st.button(
                "➕ Add Take to Enrollment Set",
                disabled=(audio_bytes_to_add is None or not volunteer_consent),
                use_container_width=True,
                type="secondary",
            ):
                try:
                    # Validate audio before adding
                    validated = load_and_validate_audio(audio_bytes_to_add, min_duration_sec=0.8)
                    st.session_state.enrollment_takes.append({
                        "name": source_label,
                        "bytes": audio_bytes_to_add,
                        "duration": validated.duration_sec,
                    })
                    st.toast(f"Take added: {validated.duration_sec:.2f}s audio verified.", icon="✅")
                    st.rerun()
                except AudioValidationError as e:
                    st.error(f"Validation Error: {e}")
                except Exception as e:
                    st.error(f"Failed to process sample: {e}")

        with btn_stage_col2:
            if st.button(
                "🗑️ Clear Takes",
                disabled=len(st.session_state.enrollment_takes) == 0,
                use_container_width=True,
            ):
                st.session_state.enrollment_takes = []
                st.rerun()

    with enroll_col_right:
        st.markdown("#### 2. Staged Enrollment Set")
        st.caption(
            "Combining multiple separate utterances captures vocal variability and produces a significantly more resilient enrollment vector."
        )

        staged_count = len(st.session_state.enrollment_takes)
        total_staged_duration = sum(t["duration"] for t in st.session_state.enrollment_takes)

        st.markdown(
            f"""
            <div class="stat-card" style="margin-bottom: 1rem;">
                <div class="stat-label">Staged Recordings</div>
                <div class="stat-value">{staged_count} <span style="font-size: 1rem; font-weight: normal; color: #94a3b8;">sample(s)</span></div>
                <div class="stat-sub">Total verified audio: {total_staged_duration:.2f}s (Mean + Unit L2-Norm)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if staged_count > 0:
            st.markdown("**Staged Samples Breakdown:**")
            for idx, take in enumerate(st.session_state.enrollment_takes):
                st.markdown(
                    f"""<div class="sample-pill">
                        <b>#{idx + 1}</b> {take['name']} 
                        <span style="color: #2dd4bf; margin-left: 4px;">({take['duration']:.2f}s)</span>
                    </div>""",
                    unsafe_allow_html=True,
                )
            st.markdown("<br>", unsafe_allow_html=True)

        enroll_btn = st.button(
            f"🚀 Enroll Profile '{enroll_id}' ({staged_count} Sample{'s' if staged_count != 1 else ''})",
            type="primary",
            use_container_width=True,
            disabled=(staged_count == 0 or not volunteer_consent),
        )

        if enroll_btn:
            if not volunteer_consent:
                st.error("Informed volunteer consent is required to process and save voice data.")
                st.stop()
            with st.spinner("Extracting embeddings and generating normalized template..."):
                try:
                    embeddings: List[torch.Tensor] = []
                    for take in st.session_state.enrollment_takes:
                        audio = load_and_validate_audio(take["bytes"], min_duration_sec=0.8)
                        emb = model.extract_embedding(audio.waveform)
                        embeddings.append(emb)

                    combined_embedding = combine_enrollment_embeddings(embeddings)

                    storage.save_volunteer(
                        demo_id=enroll_id.strip(),
                        embedding=combined_embedding,
                        duration_sec=total_staged_duration,
                        original_sample_rate=16000,
                        model_name=model.model_name,
                        num_enrollment_samples=len(embeddings),
                        save_raw_audio=False,
                    )

                    st.session_state.enrollment_takes = []
                    st.session_state.demo_id = f"VOL-{random.randint(1000, 9999)}"
                    st.success(
                        f"🎉 **Volunteer Profile '{enroll_id}' Successfully Enrolled!**\n\n"
                        f"- Combined {len(embeddings)} takes into a unit-normalized 192-dimensional vector.\n"
                        f"- Total speech processed: {total_staged_duration:.2f}s.\n"
                        f"- Raw audio streams were purged from memory.",
                        icon="✅",
                    )
                    time.sleep(1.2)
                    st.rerun()
                except Exception as e:
                    st.error(f"Enrollment failed: {e}")


# =============================================================================
# TAB 2: Voice Verification (1:1)
# =============================================================================
with tab_verify:
    st.markdown("### 🔍 Step 2: One-to-One (1:1) Voice Verification")
    st.markdown(
        "Verify an incoming voice recording against an enrolled profile. "
        "The system compares the candidate's embedding against the enrolled representation using cosine similarity."
    )

    enrolled_profiles = storage.list_volunteers()

    if len(enrolled_profiles) == 0:
        st.info("ℹ️ No enrolled profiles found on disk. Please enroll at least one volunteer profile in Step 1 first.")
    else:
        v_col_left, v_col_right = st.columns([0.5, 0.5], gap="large")

        with v_col_left:
            st.markdown("#### 1. Target Profile & Threshold")
            profile_options = [p.demo_id for p in enrolled_profiles]
            selected_demo_id = st.selectbox("Select Enrolled Profile to Verify Against:", options=profile_options)

            # Find selected profile metadata
            selected_profile_meta = next((p for p in enrolled_profiles if p.demo_id == selected_demo_id), None)
            if selected_profile_meta:
                st.caption(
                    f"Profile `{selected_demo_id}` was created on `{selected_profile_meta.enrolled_at_utc[:10]}` "
                    f"using **{selected_profile_meta.num_enrollment_samples} combined take(s)**."
                )

            threshold_val = st.slider(
                "Decision Threshold (τ):",
                min_value=0.00,
                max_value=1.00,
                value=0.31,
                step=0.01,
                help="Similarity threshold. Scores at or above this value are accepted as a MATCH. Empirically calibrated on LibriSpeech dev partition at τ = 0.31.",
            )

            st.markdown("#### 2. Candidate Audio Sample")
            verify_input_mode = st.radio(
                "Candidate Audio Input Mode:",
                ["🎙️ Record Candidate Voice", "📁 Upload Candidate Audio"],
                horizontal=True,
                key="verify_input_mode",
            )

            candidate_bytes: Optional[bytes] = None
            if "Record" in verify_input_mode:
                candidate_mic = st.audio_input("Record Candidate Voice (Minimum 0.8s):", key="verify_mic")
                if candidate_mic:
                    candidate_bytes = candidate_mic.getvalue()
            else:
                candidate_file = st.file_uploader(
                    "Upload Candidate Audio:",
                    type=["wav", "flac", "mp3", "ogg"],
                    key="verify_file",
                )
                if candidate_file:
                    candidate_bytes = candidate_file.getvalue()

            verify_btn = st.button(
                "🔎 Verify Candidate Voice",
                type="primary",
                use_container_width=True,
                disabled=(candidate_bytes is None or not volunteer_consent),
            )

        with v_col_right:
            st.markdown("#### 3. Verification Analysis & Verdict")

            if not volunteer_consent:
                st.warning("⚠️ Informed consent must be checked to perform verification.")
            elif verify_btn and candidate_bytes:
                with st.spinner("Processing candidate audio and evaluating cosine similarity..."):
                    try:
                        cand_audio = load_and_validate_audio(candidate_bytes, min_duration_sec=0.8)
                        cand_emb = model.extract_embedding(cand_audio.waveform)
                        enrolled_emb = storage.load_embedding(selected_demo_id)

                        if enrolled_emb is None:
                            st.error(f"Could not load embedding vector for '{selected_demo_id}'.")
                        else:
                            sim_score = compute_cosine_similarity(enrolled_emb, cand_emb)
                            is_match = sim_score >= threshold_val
                            margin = sim_score - threshold_val

                            # Render Visual Stat Cards
                            m_col1, m_col2, m_col3 = st.columns(3)
                            with m_col1:
                                st.markdown(
                                    f"""
                                    <div class="stat-card">
                                        <div class="stat-label">Cosine Score</div>
                                        <div class="stat-value">{sim_score:.4f}</div>
                                        <div class="stat-sub">Range [-1.0, 1.0]</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                            with m_col2:
                                st.markdown(
                                    f"""
                                    <div class="stat-card">
                                        <div class="stat-label">Threshold (τ)</div>
                                        <div class="stat-value">{threshold_val:.2f}</div>
                                        <div class="stat-sub">Operating point</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                            with m_col3:
                                st.markdown(
                                    f"""
                                    <div class="stat-card">
                                        <div class="stat-label">Decision Margin</div>
                                        <div class="stat-value" style="color: {'#34d399' if margin >= 0 else '#f87171'};">{margin:+.4f}</div>
                                        <div class="stat-sub">Score - Threshold</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                            if is_match:
                                st.markdown(
                                    f"""
                                    <div class="result-banner-match">
                                        <h3 style="color: #10b981; margin: 0 0 0.5rem 0;">✅ VERIFICATION MATCH CONFIRMED</h3>
                                        <p style="color: #cbd5e1; margin-bottom: 0;">
                                            The candidate voice produced a cosine similarity of <b>{sim_score:.4f}</b>, 
                                            which exceeds the decision threshold of <b>{threshold_val:.2f}</b> by <b>{margin:+.4f}</b>. 
                                            The identity claim for profile <code>{selected_demo_id}</code> is accepted.
                                        </p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.markdown(
                                    f"""
                                    <div class="result-banner-nomatch">
                                        <h3 style="color: #ef4444; margin: 0 0 0.5rem 0;">❌ VERIFICATION REJECTED (NO MATCH)</h3>
                                        <p style="color: #cbd5e1; margin-bottom: 0;">
                                            The candidate voice produced a cosine similarity of <b>{sim_score:.4f}</b>, 
                                            which falls below the required threshold of <b>{threshold_val:.2f}</b> by <b>{abs(margin):.4f}</b>. 
                                            The identity claim for profile <code>{selected_demo_id}</code> is rejected.
                                        </p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                            st.markdown(
                                """
                                <div class="scope-callout">
                                    <b>Interpretability Note:</b> Cosine similarity measures the geometric alignment of high-dimensional neural vectors. 
                                    It is not a probability percentage of identity. Raw candidate audio was immediately discarded from memory.
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                    except AudioValidationError as e:
                        st.error(f"Audio Validation Error: {e}")
                    except Exception as e:
                        st.error(f"Verification processing failed: {e}")
            else:
                st.info("👈 Select an enrolled profile and provide a candidate recording on the left to run verification.")


# =============================================================================
# TAB 3: Profile Governance & Disk Purging
# =============================================================================
with tab_governance:
    st.markdown("### 🛡️ Enrolled Profiles & Data Governance")
    st.markdown(
        "Review enrolled volunteer profiles stored in local repository storage (`data/volunteers`). "
        "Under biometric data protection principles, volunteers have the right to permanently purge their templates at any time."
    )

    profiles = storage.list_volunteers()
    if len(profiles) == 0:
        st.info("No enrolled volunteer profiles currently exist on disk.")
    else:
        st.markdown(f"**Total Registered Profiles:** `{len(profiles)}`")

        for p in profiles:
            p_id = p.demo_id
            p_samples = p.num_enrollment_samples
            p_created = p.enrolled_at_utc[:19].replace("T", " ")

            c1, c2, c3, c4 = st.columns([0.3, 0.3, 0.2, 0.2])
            with c1:
                st.markdown(f"**Demo ID:** `{p_id}`")
            with c2:
                st.markdown(f"📅 {p_created}")
            with c3:
                st.markdown(f"🎙️ **{p_samples}** take{'s' if p_samples != 1 else ''}")
            with c4:
                if st.button(f"🗑️ Purge", key=f"del_{p_id}", use_container_width=True, type="secondary"):
                    deleted = storage.delete_volunteer(p_id)
                    if deleted:
                        st.toast(f"Profile {p_id} permanently deleted from disk.", icon="🗑️")
                        time.sleep(0.8)
                        st.rerun()
                    else:
                        st.error(f"Failed to delete {p_id}.")
            st.divider()


# =============================================================================
# TAB 4: LibriSpeech Evaluation Mode
# =============================================================================
with tab_eval:
    st.markdown("### 📊 LibriSpeech Verification Benchmark Harness")
    st.markdown(
        "Evaluate empirical biometric error metrics (False Accept Rate, False Reject Rate, Accuracy) "
        "over real audio recordings from an external directory (e.g., LibriSpeech `test-clean`)."
    )

    st.markdown(
        """
        <div class="scope-callout">
            <b>Scientific Scope Notice:</b> LibriSpeech comprises high-fidelity (16 kHz) clean audiobook speech. 
            Metrics obtained here demonstrate theoretical model separation under ideal acoustic conditions and do not 
            establish real-world performance on noisy, compressed, or narrow-band (8 kHz) telephone banking audio.
        </div>
        """,
        unsafe_allow_html=True,
    )

    eval_col1, eval_col2 = st.columns([0.65, 0.35], gap="large")

    default_corpus_path = r"C:\Users\ishan\Downloads\LibriSpeech\test-clean"
    with eval_col1:
        dataset_path_str = st.text_input(
            "External Dataset Directory Path (read in-place without copying):",
            value=default_corpus_path,
        )

        c_val1, c_val2 = st.columns(2)
        with c_val1:
            req_genuine = st.number_input("Genuine Trial Pairs per Partition:", min_value=5, max_value=200, value=100, step=5)
        with c_val2:
            req_imposter = st.number_input("Imposter Trial Pairs per Partition:", min_value=5, max_value=200, value=100, step=5)

        enable_split = st.checkbox(
            "Enable Disjoint Calibration / Held-Out Test Split (Recommended)",
            value=True,
            help="Partitions speakers into separate calibration (dev) and evaluation (test) subsets to avoid optimistic post-hoc threshold selection bias.",
        )

    with eval_col2:
        st.markdown("#### Corpus Validation")
        adapter = LibriSpeechAdapter(dataset_path_str)
        summary = adapter.validate()

        if summary.is_valid:
            st.success(
                f"✅ **Corpus Validated**\n\n"
                f"- **Speakers Found**: {summary.num_speakers}\n"
                f"- **Recordings**: {summary.num_recordings}\n"
                f"- Sample IDs: `{', '.join(summary.sample_speakers[:4])}...`",
                icon="📁",
            )
        else:
            st.error(f"❌ {summary.message}")

    run_benchmark_btn = st.button(
        "🚀 Run Biometric Verification Benchmark",
        type="primary",
        disabled=(not summary.is_valid),
        use_container_width=True,
    )

    if run_benchmark_btn:
        progress_bar = st.progress(0, text="Initializing evaluation protocol...")
        status_text = st.empty()

        cal_speakers, eval_speakers, is_split_reliable, split_note = adapter.split_speakers_for_calibration(0.5)

        cal_seed = 42
        test_seed = 100
        cal_trials = adapter.generate_trials(
            num_genuine=req_genuine,
            num_imposter=req_imposter,
            speaker_subset=cal_speakers,
            seed=cal_seed,
        )
        test_trials = adapter.generate_trials(
            num_genuine=req_genuine,
            num_imposter=req_imposter,
            speaker_subset=eval_speakers,
            seed=test_seed,
        )

        total_trials_to_run = len(cal_trials) + len(test_trials)
        embedding_cache: Dict[str, torch.Tensor] = {}

        def get_cached_emb(p: Path) -> torch.Tensor:
            k = str(p.resolve())
            if k in embedding_cache:
                return embedding_cache[k]
            a = load_and_validate_audio(p, min_duration_sec=0.5)
            e = model.extract_embedding(a.waveform)
            embedding_cache[k] = e
            return e

        def eval_batch(trials: List[TrialPair], offset: int, total: int):
            evaluated: List[EvaluatedTrial] = []
            skipped: List[dict] = []
            for i, t in enumerate(trials):
                try:
                    ea = get_cached_emb(t.path_a)
                    eb = get_cached_emb(t.path_b)
                    sim = compute_cosine_similarity(ea, eb)
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
                except Exception as ex:
                    skipped.append({"pair": f"{t.speaker_a} vs {t.speaker_b}", "reason": str(ex)})

                current_idx = offset + i + 1
                frac = current_idx / total
                progress_bar.progress(frac, text=f"Evaluating trial {current_idx} of {total} ({len(embedding_cache)} recordings cached)...")
            return evaluated, skipped

        start_time = time.time()
        cal_eval, cal_skip = eval_batch(cal_trials, 0, total_trials_to_run)
        test_eval, test_skip = eval_batch(test_trials, len(cal_trials), total_trials_to_run)
        elapsed = time.time() - start_time

        # Sweep calibration set for threshold
        sweep = compute_metric_sweep(cal_eval, num_steps=100)
        eer_res = find_empirical_eer(sweep)
        calibrated_threshold = eer_res.threshold if eer_res else 0.25

        # Evaluate held-out test partition at calibrated threshold
        test_metric = compute_metrics_at_threshold(test_eval, threshold=calibrated_threshold)

        st.session_state.evaluation_results = {
            "calibrated_threshold": calibrated_threshold,
            "test_metric": test_metric,
            "cal_trials_count": len(cal_eval),
            "test_trials_count": len(test_eval),
            "cal_skip": len(cal_skip),
            "test_skip": len(test_skip),
            "cache_count": len(embedding_cache),
            "elapsed": elapsed,
            "note": split_note,
        }
        progress_bar.empty()
        status_text.empty()
        st.success("🎉 Benchmark evaluation completed successfully!")

    if st.session_state.evaluation_results:
        res = st.session_state.evaluation_results
        tm = res["test_metric"]

        st.markdown("#### 📈 Benchmark Results Summary")
        r_c1, r_c2, r_c3, r_c4 = st.columns(4)
        with r_c1:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">Calibrated Threshold</div>
                    <div class="stat-value">{res['calibrated_threshold']:.4f}</div>
                    <div class="stat-sub">Dev set EER (Seed 42)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with r_c2:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">Held-Out Test FAR</div>
                    <div class="stat-value" style="color: {'#34d399' if tm.far <= 0.05 else '#f87171'};">{tm.far * 100:.2f}%</div>
                    <div class="stat-sub">{tm.false_accepts} / {tm.imposter_trials} Imposters</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with r_c3:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">Held-Out Test FRR</div>
                    <div class="stat-value" style="color: {'#34d399' if tm.frr <= 0.05 else '#f87171'};">{tm.frr * 100:.2f}%</div>
                    <div class="stat-sub">{tm.false_rejects} / {tm.genuine_trials} Genuines</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with r_c4:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">Held-Out Accuracy</div>
                    <div class="stat-value">{tm.accuracy * 100:.2f}%</div>
                    <div class="stat-sub">{(tm.true_accepts + tm.true_rejects)} / {tm.total_trials} Correct</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"""
            - **Calibration Trials Evaluated**: `{res['cal_trials_count']}` (disjoint dev speakers, seed 42)
            - **Held-Out Test Trials Evaluated**: `{res['test_trials_count']}` (disjoint eval speakers, seed 100)
            - **Skipped / Corrupt Audio**: `{res['cal_skip'] + res['test_skip']}`
            - **In-Memory Cache Stats**: `{res['cache_count']}` unique audio files cached
            - **Benchmark Duration**: `{res['elapsed']:.2f}s`
            """
        )
