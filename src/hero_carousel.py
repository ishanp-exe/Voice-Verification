"""Hero Carousel Component for VoxKey.

Interactive multi-slide carousel designed for Streamlit with:
- Dark-mode-only charcoal palette (#0B0D11, #13161D, #191D26, #242936)
- Changing accents per slide: Golden-Yellow (#F59E0B / #D4A359) and Crimson Red (#EF4444 / #F43F5E)
- Focused center slide with partially visible, peeking neighboring slides
- Smooth 60fps CSS transitions and subtle hover effects
- Locally generated CSS audio waveform visuals (NO remote images, NO external services, NO Spline)
- Compact footprint (~265px height) preventing displacement of consent gate and controls
- Full keyboard (ArrowLeft / ArrowRight), touch swipe, and click controls
- Non-blocking page scrolling (zero wheel capture or trapping)
- Reduced-motion accessibility (@media prefers-reduced-motion: reduce)
"""

from __future__ import annotations

import streamlit.components.v1 as components


def get_hero_carousel_html() -> str:
    """Generates the self-contained HTML/CSS/JS string for the hero carousel."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&family=JetBrains+Mono:wght@500;600;700&display=swap');

:root {
  --bg-obsidian: #0B0D11;
  --bg-card-primary: #13161D;
  --bg-card-secondary: #0F1218;
  --border-subtle: #242936;
  --text-primary: #F0F2F7;
  --text-muted: #878E9F;
  --text-dim: #555C6E;

  /* Slide 1 Accent: Golden-Yellow */
  --accent-gold: #F59E0B;
  --accent-gold-glow: rgba(245, 158, 11, 0.22);
  --accent-gold-border: rgba(245, 158, 11, 0.45);

  /* Slide 2 Accent: Crimson Red */
  --accent-red: #EF4444;
  --accent-red-glow: rgba(239, 68, 68, 0.22);
  --accent-red-border: rgba(239, 68, 68, 0.45);

  /* Slide 3 Accent: Warm Amber / Gold */
  --accent-amber: #EAB308;
  --accent-amber-glow: rgba(234, 179, 8, 0.22);
  --accent-amber-border: rgba(234, 179, 8, 0.45);
}

* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
  -webkit-font-smoothing: antialiased;
}

body {
  background: transparent;
  color: var(--text-primary);
  font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
  overflow: hidden;
  user-select: none;
}

/* Outer Carousel Viewport */
.carousel-viewport {
  position: relative;
  width: 100%;
  height: 255px;
  background: radial-gradient(circle at 50% 0%, #171B24 0%, var(--bg-obsidian) 75%);
  border: 1px solid var(--border-subtle);
  border-radius: 16px;
  overflow: hidden;
  box-shadow: 0 16px 36px -12px rgba(0, 0, 0, 0.65), inset 0 1px 0 rgba(255, 255, 255, 0.05);
}

/* Subtle background telemetry grid */
.carousel-viewport::before {
  content: '';
  position: absolute;
  inset: 0;
  background-image: 
    linear-gradient(to right, rgba(255, 255, 255, 0.02) 1px, transparent 1px),
    linear-gradient(to bottom, rgba(255, 255, 255, 0.02) 1px, transparent 1px);
  background-size: 32px 32px;
  pointer-events: none;
}

/* Slides Deck */
.deck {
  position: relative;
  width: 100%;
  height: 100%;
}

/* Individual Slide Card */
.slide {
  position: absolute;
  top: 12px;
  bottom: 44px;
  left: 50%;
  width: 74%;
  max-width: 860px;
  background: linear-gradient(135deg, rgba(25, 29, 38, 0.95) 0%, rgba(15, 18, 25, 0.98) 100%);
  border: 1px solid var(--border-subtle);
  border-radius: 14px;
  padding: 1.15rem 1.6rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1.5rem;
  backdrop-filter: blur(12px);
  cursor: default;
  transition: transform 0.45s cubic-bezier(0.16, 1, 0.3, 1),
              opacity 0.4s ease,
              border-color 0.4s ease,
              box-shadow 0.4s ease,
              filter 0.4s ease;
  will-change: transform, opacity;
}

/* Center / Focused Slide */
.slide.active {
  transform: translateX(-50%) scale(1);
  opacity: 1;
  z-index: 10;
  pointer-events: auto;
  box-shadow: 0 12px 32px -8px var(--active-glow, rgba(245, 158, 11, 0.2)), inset 0 1px 0 rgba(255, 255, 255, 0.08);
}

.slide[data-slide="0"].active {
  border-color: var(--accent-gold-border);
  box-shadow: 0 12px 32px -8px var(--accent-gold-glow), inset 0 1px 0 rgba(255, 255, 255, 0.08);
}

.slide[data-slide="1"].active {
  border-color: var(--accent-red-border);
  box-shadow: 0 12px 32px -8px var(--accent-red-glow), inset 0 1px 0 rgba(255, 255, 255, 0.08);
}

.slide[data-slide="2"].active {
  border-color: var(--accent-amber-border);
  box-shadow: 0 12px 32px -8px var(--accent-amber-glow), inset 0 1px 0 rgba(255, 255, 255, 0.08);
}

/* Left Neighbor (Peeking) */
.slide.prev-peek {
  transform: translateX(-144%) scale(0.91);
  opacity: 0.32;
  filter: blur(0.4px);
  z-index: 4;
  cursor: pointer;
  pointer-events: auto;
}

.slide.prev-peek:hover {
  opacity: 0.55;
  transform: translateX(-142%) scale(0.93);
}

/* Right Neighbor (Peeking) */
.slide.next-peek {
  transform: translateX(44%) scale(0.91);
  opacity: 0.32;
  filter: blur(0.4px);
  z-index: 4;
  cursor: pointer;
  pointer-events: auto;
}

.slide.next-peek:hover {
  opacity: 0.55;
  transform: translateX(42%) scale(0.93);
}

/* Hidden / Far off */
.slide.hidden-far {
  transform: translateX(-50%) scale(0.8);
  opacity: 0;
  pointer-events: none;
  z-index: 1;
}

/* Slide Content Layout */
.slide-content {
  flex: 1 1 60%;
  min-width: 0;
  display: flex;
  flex-direction: column;
  justify-content: center;
}

.slide-badge-row {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.35rem;
}

.slide-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.2rem 0.55rem;
  border-radius: 9999px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.badge-gold {
  background: rgba(245, 158, 11, 0.12);
  border: 1px solid rgba(245, 158, 11, 0.3);
  color: #FBBF24;
}

.badge-red {
  background: rgba(239, 68, 68, 0.12);
  border: 1px solid rgba(239, 68, 68, 0.3);
  color: #F87171;
}

.badge-amber {
  background: rgba(234, 179, 8, 0.12);
  border: 1px solid rgba(234, 179, 8, 0.3);
  color: #FDE047;
}

.status-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  display: inline-block;
  animation: pulse-dot 2s infinite ease-in-out;
}

.badge-gold .status-dot { background: #F59E0B; box-shadow: 0 0 6px #F59E0B; }
.badge-red .status-dot { background: #EF4444; box-shadow: 0 0 6px #EF4444; }
.badge-amber .status-dot { background: #EAB308; box-shadow: 0 0 6px #EAB308; }

@keyframes pulse-dot {
  0%, 100% { transform: scale(1); opacity: 0.8; }
  50% { transform: scale(1.35); opacity: 1; }
}

.slide-title {
  font-size: 1.45rem;
  font-weight: 800;
  letter-spacing: -0.02em;
  color: #FFFFFF;
  margin-bottom: 0.25rem;
  line-height: 1.2;
}

.slide-desc {
  font-size: 0.83rem;
  line-height: 1.45;
  color: var(--text-muted);
  margin-bottom: 0.55rem;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.slide-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem;
}

.tag-pill {
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
  padding: 0.18rem 0.5rem;
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 6px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.7rem;
  color: #CBD5E1;
}

/* Right Side: Local CSS Visualizer */
.slide-visual {
  flex: 0 0 36%;
  height: 130px;
  background: rgba(11, 13, 17, 0.7);
  border: 1px solid rgba(255, 255, 255, 0.06);
  border-radius: 10px;
  padding: 0.75rem;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  position: relative;
  overflow: hidden;
}

.visual-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.65rem;
  font-weight: 600;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: var(--text-muted);
  margin-top: 0.4rem;
  text-align: center;
}

/* Wave 1: Golden Equalizer */
.wave-container {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 3px;
  height: 60px;
  width: 100%;
}

.wave-bar-gold {
  width: 3.5px;
  border-radius: 2px;
  background: linear-gradient(180deg, #FDE68A 0%, #D97706 100%);
  animation: wavePulseGold 1.2s ease-in-out infinite alternate;
}

@keyframes wavePulseGold {
  0% { height: 10%; opacity: 0.35; }
  100% { height: 95%; opacity: 1; }
}

/* Wave 2: Crimson Dual Stream */
.dual-wave-container {
  display: flex;
  flex-direction: column;
  gap: 6px;
  width: 100%;
  align-items: center;
}

.stream-row {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 3px;
  height: 26px;
  width: 100%;
}

.wave-bar-red {
  width: 3.5px;
  border-radius: 2px;
  background: linear-gradient(180deg, #FCA5A5 0%, #DC2626 100%);
  animation: wavePulseRed 1.1s ease-in-out infinite alternate;
}

@keyframes wavePulseRed {
  0% { height: 15%; opacity: 0.4; }
  100% { height: 90%; opacity: 1; }
}

/* Wave 3: Amber Benchmark Spectrum */
.spectrum-container {
  display: flex;
  align-items: flex-end;
  justify-content: center;
  gap: 2.5px;
  height: 55px;
  width: 100%;
  border-bottom: 1px dashed rgba(245, 158, 11, 0.35);
  padding-bottom: 2px;
}

.wave-bar-amber {
  width: 3.5px;
  border-radius: 2px 2px 0 0;
  background: linear-gradient(180deg, #FEF08A 0%, #B45309 100%);
  animation: wavePulseAmber 1.4s ease-in-out infinite alternate;
}

@keyframes wavePulseAmber {
  0% { height: 18%; opacity: 0.3; }
  100% { height: 88%; opacity: 1; }
}

/* Navigation Arrow Controls */
.nav-arrow {
  position: absolute;
  top: calc(50% - 16px);
  width: 34px;
  height: 34px;
  border-radius: 50%;
  background: rgba(18, 22, 30, 0.82);
  border: 1px solid rgba(255, 255, 255, 0.12);
  color: var(--text-primary);
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  z-index: 25;
  backdrop-filter: blur(8px);
  transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.35);
}

.nav-arrow:hover {
  background: rgba(30, 36, 48, 0.95);
  border-color: rgba(255, 255, 255, 0.3);
  color: #FFFFFF;
  transform: scale(1.08);
}

.nav-arrow:active {
  transform: scale(0.96);
}

.nav-arrow.prev { left: 12px; }
.nav-arrow.next { right: 12px; }

/* Bottom Dock (Counter & Indicator Pills) */
.dock {
  position: absolute;
  bottom: 9px;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  align-items: center;
  gap: 8px;
  background: rgba(13, 16, 22, 0.9);
  border: 1px solid rgba(255, 255, 255, 0.08);
  padding: 4px 12px;
  border-radius: 9999px;
  z-index: 25;
  backdrop-filter: blur(8px);
}

.dock-counter {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.72rem;
  font-weight: 700;
  color: var(--text-muted);
  margin-right: 4px;
  letter-spacing: 0.05em;
}

.indicator-track {
  display: flex;
  align-items: center;
  gap: 6px;
}

.pill-dot {
  height: 5px;
  border-radius: 3px;
  background: rgba(255, 255, 255, 0.2);
  cursor: pointer;
  transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
  width: 8px;
}

.pill-dot:hover {
  background: rgba(255, 255, 255, 0.45);
}

.pill-dot.active {
  width: 22px;
}

.pill-dot[data-dot="0"].active {
  background: var(--accent-gold);
  box-shadow: 0 0 8px var(--accent-gold);
}

.pill-dot[data-dot="1"].active {
  background: var(--accent-red);
  box-shadow: 0 0 8px var(--accent-red);
}

.pill-dot[data-dot="2"].active {
  background: var(--accent-amber);
  box-shadow: 0 0 8px var(--accent-amber);
}

/* Reduced Motion Accessibility */
@media (prefers-reduced-motion: reduce) {
  .slide, .nav-arrow, .pill-dot {
    transition: none !important;
  }
  .wave-bar-gold, .wave-bar-red, .wave-bar-amber, .status-dot {
    animation: none !important;
  }
}

/* Responsive adjustments */
@media (max-width: 768px) {
  .carousel-viewport {
    height: 275px;
  }
  .slide {
    width: 88%;
    padding: 1rem 1.1rem;
    flex-direction: column;
    align-items: flex-start;
    gap: 0.75rem;
    bottom: 42px;
  }
  .slide-visual {
    width: 100%;
    height: 70px;
    flex: 0 0 70px;
  }
  .slide-title {
    font-size: 1.2rem;
  }
  .slide-desc {
    font-size: 0.78rem;
    -webkit-line-clamp: 2;
  }
  .nav-arrow {
    display: none;
  }
}
</style>
</head>
<body>

<div class="carousel-viewport" id="carousel" role="region" aria-label="Biometric Architecture Showcase">
  <div class="deck">
    <!-- SLIDE 1: Enroll a voice -->
    <div class="slide active" data-slide="0" onclick="handleSlideClick(0)">
      <div class="slide-content">
        <div class="slide-badge-row">
          <span class="slide-badge badge-gold">
            <span class="status-dot"></span> 01 / Consent & Multi-Sample
          </span>
        </div>
        <div class="slide-title">Enroll a Voice</div>
        <div class="slide-desc">
          Voluntary biometric registration with strict informed consent. Combines multiple vocal takes into a single 
          robust 192-dim embedding template using element-wise mean and unit L₂-normalization.
        </div>
        <div class="slide-tags">
          <span class="tag-pill">🛡️ Consent Gate</span>
          <span class="tag-pill">🎙️ Multi-Take L₂ Mean</span>
          <span class="tag-pill">🗑️ Zero Audio Retained</span>
        </div>
      </div>
      <div class="slide-visual">
        <div class="wave-container" id="gold-wave"></div>
        <div class="visual-label">16 kHz Mono Intake • L₂ Normalization</div>
      </div>
    </div>

    <!-- SLIDE 2: Verify a speaker -->
    <div class="slide next-peek" data-slide="1" onclick="handleSlideClick(1)">
      <div class="slide-content">
        <div class="slide-badge-row">
          <span class="slide-badge badge-red">
            <span class="status-dot"></span> 02 / 1:1 Voice Comparison
          </span>
        </div>
        <div class="slide-title">Verify a Speaker</div>
        <div class="slide-desc">
          Instantaneous 1:1 candidate scoring. Extracts test utterance embeddings via SpeechBrain ECAPA-TDNN and 
          computes cosine similarity against the enrolled profile template with calibrated decision thresholding.
        </div>
        <div class="slide-tags">
          <span class="tag-pill">🧠 ECAPA-TDNN 192-d</span>
          <span class="tag-pill">📐 Cosine Similarity</span>
          <span class="tag-pill">⚖️ Calibrated Threshold (θ)</span>
        </div>
      </div>
      <div class="slide-visual">
        <div class="dual-wave-container">
          <div class="stream-row" id="red-wave-1"></div>
          <div class="stream-row" id="red-wave-2"></div>
        </div>
        <div class="visual-label">1:1 Biometric Match vs Imposter</div>
      </div>
    </div>

    <!-- SLIDE 3: Evaluate the model -->
    <div class="slide prev-peek" data-slide="2" onclick="handleSlideClick(2)">
      <div class="slide-content">
        <div class="slide-badge-row">
          <span class="slide-badge badge-amber">
            <span class="status-dot"></span> 03 / LibriSpeech Benchmark
          </span>
        </div>
        <div class="slide-title">Evaluate the Model</div>
        <div class="slide-desc">
          In-situ empirical validation across 2,620 LibriSpeech recordings. Enforces disjoint calibration vs. 
          test splits, in-memory embedding caching, and honest academic audiobook scope limitations.
        </div>
        <div class="slide-tags">
          <span class="tag-pill">📚 LibriSpeech test-clean</span>
          <span class="tag-pill">🔀 Disjoint Dev / Eval</span>
          <span class="tag-pill">📈 Empirical FAR / FRR</span>
        </div>
      </div>
      <div class="slide-visual">
        <div class="spectrum-container" id="amber-spectrum"></div>
        <div class="visual-label">Disjoint Split & Empirical EER</div>
      </div>
    </div>
  </div>

  <!-- Navigation Arrows -->
  <button class="nav-arrow prev" onclick="prevSlide()" aria-label="Previous slide">&#8249;</button>
  <button class="nav-arrow next" onclick="nextSlide()" aria-label="Next slide">&#8250;</button>

  <!-- Bottom Indicator Dock -->
  <div class="dock">
    <span class="dock-counter" id="slide-counter">01 / 03</span>
    <div class="indicator-track">
      <div class="pill-dot active" data-dot="0" onclick="goToSlide(0)" title="Slide 1: Enroll a voice"></div>
      <div class="pill-dot" data-dot="1" onclick="goToSlide(1)" title="Slide 2: Verify a speaker"></div>
      <div class="pill-dot" data-dot="2" onclick="goToSlide(2)" title="Slide 3: Evaluate the model"></div>
    </div>
  </div>
</div>

<script>
// State management
let currentSlide = 0;
const totalSlides = 3;

// Generate local CSS wave bars dynamically
function createWaveBars(containerId, count, barClass, animDelayBase) {
  const container = document.getElementById(containerId);
  if (!container) return;
  container.innerHTML = '';
  for (let i = 0; i < count; i++) {
    const bar = document.createElement('div');
    bar.className = barClass;
    // Varied natural heights and delays
    const delay = (i * animDelayBase) % 1.2;
    bar.style.animationDelay = `${delay.toFixed(2)}s`;
    // Preset harmonic height profile
    const progress = i / count;
    const sinHeight = Math.sin(progress * Math.PI) * 75 + 15;
    bar.style.height = `${Math.max(12, Math.min(95, sinHeight))}%`;
    container.appendChild(bar);
  }
}

// Initialize wave visuals
createWaveBars('gold-wave', 18, 'wave-bar-gold', 0.08);
createWaveBars('red-wave-1', 14, 'wave-bar-red', 0.09);
createWaveBars('red-wave-2', 14, 'wave-bar-red', 0.11);
createWaveBars('amber-spectrum', 20, 'wave-bar-amber', 0.07);

function updateCarousel() {
  const slides = document.querySelectorAll('.slide');
  const dots = document.querySelectorAll('.pill-dot');
  const counter = document.getElementById('slide-counter');

  if (counter) {
    counter.textContent = `0${currentSlide + 1} / 0${totalSlides}`;
  }

  slides.forEach((slide) => {
    const index = parseInt(slide.getAttribute('data-slide'), 10);
    slide.classList.remove('active', 'prev-peek', 'next-peek', 'hidden-far');

    // Calculate relative cyclic position
    const diff = (index - currentSlide + totalSlides) % totalSlides;

    if (diff === 0) {
      slide.classList.add('active');
    } else if (diff === totalSlides - 1) {
      // Immediate predecessor (left peek)
      slide.classList.add('prev-peek');
    } else if (diff === 1) {
      // Immediate successor (right peek)
      slide.classList.add('next-peek');
    } else {
      slide.classList.add('hidden-far');
    }
  });

  dots.forEach((dot) => {
    const dotIndex = parseInt(dot.getAttribute('data-dot'), 10);
    if (dotIndex === currentSlide) {
      dot.classList.add('active');
    } else {
      dot.classList.remove('active');
    }
  });
}

function goToSlide(idx) {
  currentSlide = (idx + totalSlides) % totalSlides;
  updateCarousel();
}

function nextSlide() {
  goToSlide(currentSlide + 1);
}

function prevSlide() {
  goToSlide(currentSlide - 1);
}

function handleSlideClick(idx) {
  if (idx !== currentSlide) {
    goToSlide(idx);
  }
}

// Keyboard arrow navigation
window.addEventListener('keydown', (e) => {
  if (e.key === 'ArrowRight') {
    nextSlide();
  } else if (e.key === 'ArrowLeft') {
    prevSlide();
  }
});

// Touch swipe navigation for mobile
let touchStartX = 0;
let touchEndX = 0;
const carouselEl = document.getElementById('carousel');

if (carouselEl) {
  carouselEl.addEventListener('touchstart', (e) => {
    touchStartX = e.changedTouches[0].screenX;
  }, { passive: true });

  carouselEl.addEventListener('touchend', (e) => {
    touchEndX = e.changedTouches[0].screenX;
    const swipeDist = touchEndX - touchStartX;
    if (Math.abs(swipeDist) > 40) {
      if (swipeDist < 0) {
        nextSlide();
      } else {
        prevSlide();
      }
    }
  }, { passive: true });
}

// Initial draw
updateCarousel();
</script>
</body>
</html>
"""


def render_hero_carousel(height: int = 265) -> None:
    """Renders the dark-mode hero carousel component in Streamlit."""
    components.html(get_hero_carousel_html(), height=height)
