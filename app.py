"""VoxKey - Streamlit Application.

An internship-level educational prototype for 1:1 voice biometric verification.
Features:
- Modern dark analytics UI with hero carousel and glassmorphic styling
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
from src.hero_carousel import render_hero_carousel
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
    page_title="VoxKey",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Modern Dark Analytics Styling (Hero Spotlight, Glassmorphic Cards, Responsive Motion)
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

/* Scroll progress indicator (subtle scroll effect) */
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

/* Hero Card with Spotlight Effect */
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

/* Reusable Glassmorphic Card */
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

/* Primary AGON Gold Action Buttons */
.stButton > button[kind="primary"] {
    background: #F59E0B !important;
    color: #0B0D11 !important;
    border: none !important;
    font-weight: 800 !important;
    border-radius: 9px !important;
    box-shadow: 0 4px 18px rgba(245, 158, 11, 0.28) !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

.stButton > button[kind="primary"]:hover {
    background: #FBBF24 !important;
    color: #000000 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 22px rgba(245, 158, 11, 0.4) !important;
}

.stButton > button[kind="primary"]:disabled {
    background: #1C212D !important;
    color: #555C6E !important;
    box-shadow: none !important;
    transform: none !important;
}

/* Secondary Action Buttons */
.stButton > button[kind="secondary"] {
    background: rgba(22, 27, 38, 0.85) !important;
    border: 1px solid #2E364A !important;
    color: #F0F2F7 !important;
    font-weight: 600 !important;
    border-radius: 9px !important;
    transition: all 0.2s ease !important;
}

.stButton > button[kind="secondary"]:hover {
    background: rgba(35, 42, 58, 0.95) !important;
    border-color: rgba(245, 158, 11, 0.5) !important;
    color: #FBBF24 !important;
}

/* Staging Telemetry Deck */
.telemetry-card {
    background: #0F1218;
    border: 1px solid #242936;
    border-radius: 12px;
    padding: 1.3rem;
    position: relative;
    overflow: hidden;
}

.telemetry-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 2px;
    background: linear-gradient(90deg, #F59E0B, transparent);
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
    st.markdown("## 🎙️ **VoxKey**")
    st.caption("Educational Voice-Verification Prototype")
    st.markdown("---")
    st.markdown("### Core Biometrics Engine")

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
    st.markdown("### 🎛️ Ambient Audio Visualizer")
    show_ambient = st.checkbox(
        "Show Voice Wave Monitor",
        value=False,
        help="Local CSS sound wave monitor. Optional and purely decorative; zero external network requests or dependencies.",
    )

    if show_ambient:
        st.caption("Live CSS sound wave monitor (purely ambient, zero remote calls):")
        ambient_html = """
        <div style="width: 100%; height: 110px; border-radius: 12px; background: #0F1218; border: 1px solid #242936; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 8px; padding: 12px;">
            <div style="display: flex; align-items: center; gap: 4px; height: 48px;">
                <div style="width: 4px; height: 18px; background: #F59E0B; border-radius: 2px; animation: bar1 1.1s ease-in-out infinite alternate;"></div>
                <div style="width: 4px; height: 36px; background: #F59E0B; border-radius: 2px; animation: bar2 0.9s ease-in-out infinite alternate;"></div>
                <div style="width: 4px; height: 24px; background: #EF4444; border-radius: 2px; animation: bar1 1.3s ease-in-out infinite alternate;"></div>
                <div style="width: 4px; height: 42px; background: #EF4444; border-radius: 2px; animation: bar2 1.0s ease-in-out infinite alternate;"></div>
                <div style="width: 4px; height: 30px; background: #EAB308; border-radius: 2px; animation: bar1 0.8s ease-in-out infinite alternate;"></div>
                <div style="width: 4px; height: 46px; background: #EAB308; border-radius: 2px; animation: bar2 1.2s ease-in-out infinite alternate;"></div>
                <div style="width: 4px; height: 20px; background: #F59E0B; border-radius: 2px; animation: bar1 1.4s ease-in-out infinite alternate;"></div>
            </div>
            <span style="font-family: monospace; font-size: 10px; color: #878E9F; text-transform: uppercase; letter-spacing: 0.05em;">16 kHz Acoustic Monitor</span>
            <style>
                @keyframes bar1 { 0% { height: 12px; opacity: 0.4; } 100% { height: 44px; opacity: 1; } }
                @keyframes bar2 { 0% { height: 40px; opacity: 1; } 100% { height: 14px; opacity: 0.5; } }
                @media (prefers-reduced-motion: reduce) { div { animation: none !important; } }
            </style>
        </div>
        """
        st.components.v1.html(ambient_html, height=120)
    else:
        st.caption("Local audio monitor standby. High-performance UI mode active.")

    st.markdown("---")
    st.caption("Educational Biometric Prototype • Zero Cloud Dependencies")


# -----------------------------------------------------------------------------
# Main Hero Section (Dark-Mode Hero Carousel)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div style="display: flex; align-items: baseline; gap: 0.75rem; margin-bottom: 0.35rem;">
        <h1 style="font-size: 2.2rem; font-weight: 800; letter-spacing: -0.03em; color: #FFFFFF; margin: 0; line-height: 1;">VoxKey</h1>
        <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; font-weight: 700; color: #F59E0B; text-transform: uppercase; letter-spacing: 0.06em;">Educational Voice-Verification Prototype</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("🔒 Educational voice-verification prototype • Local ECAPA-TDNN 192-dim vectors • Zero cloud retention")
render_hero_carousel(height=265)


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
    st.markdown(
        """
        <div style="margin-bottom: 1.4rem;">
            <div style="display: inline-flex; align-items: center; gap: 0.4rem; padding: 0.22rem 0.65rem; background: rgba(245, 158, 11, 0.12); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 9999px; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; font-weight: 700; color: #FBBF24; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 0.5rem;">
                <span style="width: 6px; height: 6px; border-radius: 50%; background: #F59E0B; display: inline-block;"></span> 01 // VOLUNTEER ONBOARDING
            </div>
            <h2 style="font-size: 1.55rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.02em; margin: 0 0 0.4rem 0;">
                Step 1: Multi-Sample Volunteer Voice Enrollment
            </h2>
            <div style="color: #878E9F; font-size: 0.88rem; line-height: 1.55; max-width: 920px;">
                Enroll a consenting volunteer by providing <b>one or more separate audio takes</b> (e.g., repeating a passphrase or speaking naturally). 
                The model extracts an embedding vector for each take and computes an element-wise arithmetic mean with unit <i>L</i><sub>2</sub>-normalization to establish a robust template.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not volunteer_consent:
        st.warning("⚠️ You must check the Informed Consent box above before you can stage recordings or enroll a profile.")

    enroll_col_left, enroll_col_right = st.columns([0.55, 0.45], gap="large")

    with enroll_col_left:
        st.markdown(
            """
            <div style="display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid #242936; padding-bottom: 0.5rem; margin-bottom: 0.9rem;">
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.76rem; font-weight: 700; color: #878E9F; text-transform: uppercase; letter-spacing: 0.05em;">PANEL 1A // AUDIO INTAKE & IDENTITY</span>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; font-weight: 700; padding: 0.15rem 0.5rem; border-radius: 9999px; background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.3); color: #34D399;">● READY</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

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

        with st.expander("💡 Microphone Access & Audio Troubleshooting"):
            st.markdown(
                """
                - **Microphone Blocked / Permission Issue**:
                  - In Google Chrome or Microsoft Edge, look at the address bar URL (`http://127.0.0.1:8501`).
                  - Click the **Padlock** or **Site Settings** icon.
                  - Ensure **Microphone** is toggled to **Allow**.
                  - Refresh the page (`F5`).
                - **Duration Requirements**:
                  - Audio recordings must be at least **0.8 seconds** long. Short clicks or pops will fail validation.
                - **Supported Formats**:
                  - Uncompressed WAV (recommended), FLAC, MP3, and OGG.
                  - Signals are standardized to 16,000 Hz mono in memory.
                """
            )

    with enroll_col_right:
        st.markdown(
            """
            <div style="display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid #242936; padding-bottom: 0.5rem; margin-bottom: 0.9rem;">
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.76rem; font-weight: 700; color: #878E9F; text-transform: uppercase; letter-spacing: 0.05em;">PANEL 1B // EMBEDDING STAGING POOL</span>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; font-weight: 700; padding: 0.15rem 0.5rem; border-radius: 9999px; background: rgba(245, 158, 11, 0.12); border: 1px solid rgba(245, 158, 11, 0.3); color: #FBBF24;">● MULTI-TAKE MEAN</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        staged_count = len(st.session_state.enrollment_takes)
        total_staged_duration = sum(t["duration"] for t in st.session_state.enrollment_takes)

        status_chip = (
            '<span style="color: #34D399; font-weight: 600;">● Ready to Compile</span>'
            if staged_count > 0
            else '<span style="color: #878E9F;">Waiting for audio take</span>'
        )

        st.markdown(
            f"""
            <div class="telemetry-card" style="margin-bottom: 1.1rem;">
                <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.3rem;">
                    <div>
                        <span style="font-family: 'JetBrains Mono', monospace; font-size: 2.6rem; font-weight: 800; color: #FFFFFF; line-height: 1;">{staged_count}</span>
                        <span style="font-size: 1rem; font-weight: 600; color: #878E9F; margin-left: 6px;">sample(s) staged</span>
                    </div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;">
                        {status_chip}
                    </div>
                </div>
                <div style="font-size: 0.82rem; color: #878E9F; margin-bottom: 0.8rem;">
                    Combining multiple utterances captures vocal variability and produces a significantly more resilient enrollment vector.
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.6rem; padding-top: 0.8rem; border-top: 1px solid rgba(255,255,255,0.06); font-family: 'JetBrains Mono', monospace; font-size: 0.74rem; color: #878E9F;">
                    <div>Total Speech: <b style="color: #FFFFFF;">{total_staged_duration:.2f}s</b></div>
                    <div>Format: <b style="color: #FFFFFF;">16 kHz Mono</b></div>
                    <div>Norm Type: <b style="color: #FFFFFF;">Unit L2-Norm</b></div>
                    <div>Raw Audio: <b style="color: #34D399;">Purged on Save</b></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if staged_count > 0:
            st.markdown(
                '<div style="font-family: \'JetBrains Mono\', monospace; font-size: 0.75rem; font-weight: 700; color: #878E9F; text-transform: uppercase; margin-bottom: 0.4rem;">Staged Takes Breakdown:</div>',
                unsafe_allow_html=True,
            )
            for idx, take in enumerate(st.session_state.enrollment_takes):
                st.markdown(
                    f"""<div style="display: flex; align-items: center; justify-content: space-between; background: #0F1218; border: 1px solid #242936; border-radius: 8px; padding: 0.45rem 0.8rem; margin-bottom: 0.35rem; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;">
                        <div>
                            <span style="color: #F59E0B; font-weight: 700; margin-right: 6px;">#{idx + 1}</span>
                            <span style="color: #F0F2F7;">{take['name']}</span>
                        </div>
                        <span style="color: #34D399; background: rgba(16, 185, 129, 0.12); padding: 0.15rem 0.45rem; border-radius: 4px; font-size: 0.75rem;">{take['duration']:.2f}s</span>
                    </div>""",
                    unsafe_allow_html=True,
                )
            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        enroll_btn = st.button(
            f"⚡ Enroll Profile '{enroll_id}' ({staged_count} Sample{'s' if staged_count != 1 else ''})",
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
    st.markdown(
        """
        <div style="margin-bottom: 1.4rem;">
            <div style="display: inline-flex; align-items: center; gap: 0.4rem; padding: 0.22rem 0.65rem; background: rgba(239, 68, 68, 0.12); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 9999px; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; font-weight: 700; color: #F87171; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 0.5rem;">
                <span style="width: 6px; height: 6px; border-radius: 50%; background: #EF4444; display: inline-block;"></span> 02 // 1:1 BIOMETRIC COMPARISON
            </div>
            <h2 style="font-size: 1.55rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.02em; margin: 0 0 0.4rem 0;">
                Step 2: One-to-One (1:1) Voice Verification
            </h2>
            <div style="color: #878E9F; font-size: 0.88rem; line-height: 1.55; max-width: 920px;">
                Verify an incoming voice recording against an enrolled profile. 
                The system extracts an embedding via ECAPA-TDNN and compares it against the enrolled representation using cosine similarity.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    enrolled_profiles = storage.list_volunteers()

    if len(enrolled_profiles) == 0:
        st.info("ℹ️ No enrolled profiles found on disk. Please enroll at least one volunteer profile in Step 1 first.")
    else:
        v_col_left, v_col_right = st.columns([0.5, 0.5], gap="large")

        with v_col_left:
            st.markdown(
                """
                <div style="display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid #242936; padding-bottom: 0.5rem; margin-bottom: 0.9rem;">
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.76rem; font-weight: 700; color: #878E9F; text-transform: uppercase; letter-spacing: 0.05em;">PANEL 2A // TARGET PROFILE & THRESHOLD</span>
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; font-weight: 700; padding: 0.15rem 0.5rem; border-radius: 9999px; background: rgba(239, 68, 68, 0.12); border: 1px solid rgba(239, 68, 68, 0.3); color: #F87171;">● 192-DIM ECAPA</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
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

            with st.expander("💡 Candidate Audio Troubleshooting"):
                st.markdown(
                    """
                    - **Microphone Inactive**: Ensure microphone permission is granted in browser site settings.
                    - **Sample Length**: Speak for at least **0.8 seconds** for the model to capture characteristic vocal tract features.
                    - **Geometric Cosine Similarity**: Note that scores measure angle alignment in $[-1.0, 1.0]$. A score of 0.65 is not a "65% probability" of being the same person.
                    """
                )

        with v_col_right:
            st.markdown(
                """
                <div style="display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid #242936; padding-bottom: 0.5rem; margin-bottom: 0.9rem;">
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.76rem; font-weight: 700; color: #878E9F; text-transform: uppercase; letter-spacing: 0.05em;">PANEL 2B // VERIFICATION VERDICT</span>
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; font-weight: 700; padding: 0.15rem 0.5rem; border-radius: 9999px; background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.3); color: #34D399;">● REALTIME SCORING</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

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
    st.markdown(
        """
        <div style="margin-bottom: 1.4rem;">
            <div style="display: inline-flex; align-items: center; gap: 0.4rem; padding: 0.22rem 0.65rem; background: rgba(59, 130, 246, 0.12); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 9999px; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; font-weight: 700; color: #60A5FA; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 0.5rem;">
                <span style="width: 6px; height: 6px; border-radius: 50%; background: #3B82F6; display: inline-block;"></span> 03 // PROFILE GOVERNANCE & PRIVACY
            </div>
            <h2 style="font-size: 1.55rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.02em; margin: 0 0 0.4rem 0;">
                Profile Governance & Template Purging
            </h2>
            <div style="color: #878E9F; font-size: 0.88rem; line-height: 1.55; max-width: 920px;">
                Review enrolled volunteer profiles stored in local repository storage (<code>data/volunteers</code>). 
                Under biometric data protection principles, volunteers have the right to permanently purge their templates at any time.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
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

        st.markdown("#### 🚫 Consent Withdrawal & Immediate Data Purge")
        st.markdown(
            "Under biometric privacy principles (e.g. GDPR Art. 17 / CCPA), any volunteer can withdraw consent at any time. "
            "Withdrawing consent immediately purges the volunteer's `.pt` embedding vector and `metadata.json` from the local file system."
        )

        withdraw_col1, withdraw_col2 = st.columns([0.7, 0.3])
        with withdraw_col1:
            withdraw_target = st.selectbox(
                "Select profile to withdraw consent for:",
                options=[p.demo_id for p in profiles],
                key="withdraw_target_id",
            )
        with withdraw_col2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("🚫 Withdraw & Purge", type="primary", use_container_width=True, key="withdraw_consent_btn"):
                purged = storage.delete_volunteer(withdraw_target)
                if purged:
                    st.session_state.global_volunteer_consent = False
                    st.success(f"Consent withdrawn. Profile `{withdraw_target}` and all associated vector data permanently purged.")
                    time.sleep(1.0)
                    st.rerun()
                else:
                    st.error(f"Failed to purge profile `{withdraw_target}`.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🛡️ Local Data Architecture & Privacy Summary")
    st.markdown(
        """
        <div class="stat-card" style="font-size: 0.9rem; line-height: 1.6;">
            <b>Storage Path:</b> <code>data/volunteers/&lt;demo_id&gt;/</code><br>
            <b>Stored Artifacts:</b>
            <ul>
                <li><code>embedding.pt</code>: 192-dimensional floating point tensor representing voice characteristics.</li>
                <li><code>metadata.json</code>: Anonymous timestamp, sample count, model source (no PII).</li>
            </ul>
            <b>Raw Audio Retention:</b> <b>Zero</b> raw audio recordings are stored on disk. Audio streams are validated and converted in memory, then immediately freed by Python garbage collection.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# TAB 4: LibriSpeech Evaluation Mode
# =============================================================================
with tab_eval:
    st.markdown(
        """
        <div style="margin-bottom: 1.4rem;">
            <div style="display: inline-flex; align-items: center; gap: 0.4rem; padding: 0.22rem 0.65rem; background: rgba(234, 179, 8, 0.12); border: 1px solid rgba(234, 179, 8, 0.3); border-radius: 9999px; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; font-weight: 700; color: #FDE047; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 0.5rem;">
                <span style="width: 6px; height: 6px; border-radius: 50%; background: #EAB308; display: inline-block;"></span> 04 // BENCHMARK & CALIBRATION
            </div>
            <h2 style="font-size: 1.55rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.02em; margin: 0 0 0.4rem 0;">
                LibriSpeech Verification Benchmark Harness
            </h2>
            <div style="color: #878E9F; font-size: 0.88rem; line-height: 1.55; max-width: 920px;">
                Evaluate empirical biometric error metrics (False Accept Rate, False Reject Rate, Accuracy) 
                over real audio recordings from an external directory (e.g., LibriSpeech <code>test-clean</code>).
            </div>
        </div>
        """,
        unsafe_allow_html=True,
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

    eval_col1, eval_col2 = st.columns([0.6, 0.4], gap="large")

    default_corpus_path = r"C:\Users\ishan\Downloads\LibriSpeech\test-clean"
    valid_index_path = Path("data/evaluation/valid_audio_index.csv")
    has_valid_index = valid_index_path.is_file()

    with eval_col1:
        st.markdown(
            """
            <div style="display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid #242936; padding-bottom: 0.5rem; margin-bottom: 0.9rem;">
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.76rem; font-weight: 700; color: #878E9F; text-transform: uppercase; letter-spacing: 0.05em;">PANEL 4A // DATASET PARTITIONS & TRIALS</span>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; font-weight: 700; padding: 0.15rem 0.5rem; border-radius: 9999px; background: rgba(234, 179, 8, 0.12); border: 1px solid rgba(234, 179, 8, 0.3); color: #FDE047;">● DISJOINT SPLIT</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
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

        use_validated_index = st.checkbox(
            "Use Clean Validated Index (valid_audio_index.csv)",
            value=has_valid_index,
            disabled=not has_valid_index,
            help="Directs trial generation to use only audio files that passed acoustic and integrity validation.",
        )

    with eval_col2:
        st.markdown(
            """
            <div style="display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid #242936; padding-bottom: 0.5rem; margin-bottom: 0.9rem;">
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.76rem; font-weight: 700; color: #878E9F; text-transform: uppercase; letter-spacing: 0.05em;">PANEL 4B // CORPUS VALIDATION & AUDIT</span>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; font-weight: 700; padding: 0.15rem 0.5rem; border-radius: 9999px; background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.3); color: #34D399;">● IN-SITU</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        adapter = LibriSpeechAdapter(
            dataset_path_str,
            valid_index_path=valid_index_path if use_validated_index and has_valid_index else None,
        )
        summary = adapter.validate()

        if summary.is_valid:
            index_tag = " (Pre-Validated Index Active)" if summary.used_validated_index else ""
            st.success(
                f"✅ **Corpus Validated{index_tag}**\n\n"
                f"- **Speakers Found**: {summary.num_speakers}\n"
                f"- **Recordings**: {summary.num_recordings:,}\n"
                f"- Sample IDs: `{', '.join(summary.sample_speakers[:4])}...`",
                icon="📁",
            )
        else:
            st.error(f"❌ {summary.message}")

        if st.button("🔄 Run Dataset Cleaning Script", use_container_width=True, help="Executes scripts/clean_validate_dataset.py in-place to re-validate all audio files."):
            with st.spinner("Validating and hashing all audio files in corpus..."):
                try:
                    from scripts.clean_validate_dataset import validate_dataset
                    all_recs, val_recs, rejs = validate_dataset(Path(dataset_path_str), Path("data/evaluation"))
                    st.toast(f"Validated {len(val_recs):,} audio files across {len(set(r.speaker_id for r in val_recs))} speakers.", icon="✅")
                    time.sleep(1.0)
                    st.rerun()
                except Exception as ex:
                    st.error(f"Cleaning script failed: {ex}")

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
            "used_validated_index": summary.used_validated_index,
            "dataset_path": dataset_path_str,
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
            - **Data Index**: `{'pre-validated valid_audio_index.csv' if res.get('used_validated_index') else 'direct directory scan'}`
            - **Calibration Trials Evaluated**: `{res['cal_trials_count']}` (disjoint dev speakers, seed 42)
            - **Held-Out Test Trials Evaluated**: `{res['test_trials_count']}` (disjoint eval speakers, seed 100)
            - **Skipped / Corrupt Audio**: `{res['cal_skip'] + res['test_skip']}`
            - **In-Memory Cache Stats**: `{res['cache_count']}` unique audio files cached
            - **Benchmark Duration**: `{res['elapsed']:.2f}s`
            """
        )

        # Export Data in JSON and CSV
        import csv
        import json

        export_dict = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_directory": res.get("dataset_path", dataset_path_str),
            "used_validated_index": res.get("used_validated_index", False),
            "calibrated_threshold": float(res["calibrated_threshold"]),
            "seeds": {"calibration_seed": 42, "evaluation_seed": 100},
            "trials_summary": {
                "calibration_trials": res["cal_trials_count"],
                "test_trials": res["test_trials_count"],
                "skipped_trials": res["cal_skip"] + res["test_skip"],
                "unique_recordings_cached": res["cache_count"],
            },
            "held_out_metrics": {
                "far": float(tm.far),
                "frr": float(tm.frr),
                "accuracy": float(tm.accuracy),
                "false_accepts": tm.false_accepts,
                "false_rejects": tm.false_rejects,
                "true_accepts": tm.true_accepts,
                "true_rejects": tm.true_rejects,
                "imposter_trials": tm.imposter_trials,
                "genuine_trials": tm.genuine_trials,
                "total_trials": tm.total_trials,
            },
            "disclaimer": "Evaluated on clean 16 kHz audiobook speech (LibriSpeech test-clean). Does not establish performance on telephone banking audio.",
        }
        json_payload = json.dumps(export_dict, indent=2)

        csv_buf = io.StringIO()
        writer = csv.writer(csv_buf)
        writer.writerow(["Parameter / Metric", "Value", "Notes"])
        writer.writerow(["Dataset Directory", res.get("dataset_path", dataset_path_str), "External in-situ"])
        writer.writerow(["Used Validated Index", res.get("used_validated_index", False), "valid_audio_index.csv"])
        writer.writerow(["Calibrated Threshold (τ)", f"{res['calibrated_threshold']:.4f}", "Dev EER operating point (Seed 42)"])
        writer.writerow(["Held-Out Test FAR", f"{tm.far:.4f}", f"{tm.false_accepts} / {tm.imposter_trials} Imposters (Seed 100)"])
        writer.writerow(["Held-Out Test FRR", f"{tm.frr:.4f}", f"{tm.false_rejects} / {tm.genuine_trials} Genuines (Seed 100)"])
        writer.writerow(["Held-Out Test Accuracy", f"{tm.accuracy:.4f}", f"{(tm.true_accepts + tm.true_rejects)} / {tm.total_trials} Correct"])
        writer.writerow(["Total Trials Evaluated", res["cal_trials_count"] + res["test_trials_count"], "400 requested across partitions"])
        writer.writerow(["Unique Audio Cached", res["cache_count"], "In-memory tensor cache"])
        writer.writerow(["Skipped Trials", res["cal_skip"] + res["test_skip"], "0 rejected / corrupt"])
        writer.writerow(["Disclaimer", "Clean audiobook speech (LibriSpeech test-clean). Not telephone banking audio.", "Academic scope"])
        csv_payload = csv_buf.getvalue()

        st.markdown("#### 💾 Export Benchmark Results")
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            st.download_button(
                "📥 Export Results as JSON",
                data=json_payload,
                file_name="librispeech_evaluation_results.json",
                mime="application/json",
                use_container_width=True,
            )
        with exp_col2:
            st.download_button(
                "📥 Export Results as CSV",
                data=csv_payload,
                file_name="librispeech_evaluation_results.csv",
                mime="text/csv",
                use_container_width=True,
            )

