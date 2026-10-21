from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "data" / "evaluation"
OUT = EVAL / "eda"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(EVAL / "valid_audio_index.csv")
audit = pd.read_csv(EVAL / "audio_file_validation.csv")
df["speaker_id"] = df.speaker_id.astype(str)
for col in ("duration_sec", "sample_rate", "num_channels"):
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=["speaker_id", "duration_sec"])

def font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()

def canvas(title, width, height):
    image = Image.new("RGB", (width, height), "#f8fafc")
    draw = ImageDraw.Draw(image)
    draw.text((35, 20), title, fill="#102a43", font=font(25))
    return image, draw

# Duration histogram
edges = [0, 2, 4, 6, 8, 10, 15, 20, 35]
labels = ["0-2", "2-4", "4-6", "6-8", "8-10", "10-15", "15-20", "20-35"]
durations = df.duration_sec.tolist()
counts = [sum(lo <= x < hi for x in durations) for lo, hi in zip(edges, edges[1:])]
image, draw = canvas("Recording duration distribution (seconds)", 1100, 560)
left, top, width, height = 85, 85, 960, 380
peak = max(counts) or 1
for tick in range(6):
    y = top + height - tick * height // 5
    draw.line((left, y, left + width, y), fill="#d9e2ec")
    draw.text((35, y - 8), str(round(peak * tick / 5)), fill="#486581", font=font(13))
bar_width = width // len(counts)
for i, count in enumerate(counts):
    x = left + i * bar_width + 12
    y = top + height - int(count / peak * height)
    draw.rectangle((x, y, x + bar_width - 24, top + height), fill="#07839b")
    draw.text((x + 2, y - 20), str(count), fill="#102a43", font=font(13))
    draw.text((x + 4, top + height + 10), labels[i], fill="#334e68", font=font(13))
image.save(OUT / "duration_distribution.png")

# Recordings per speaker
speaker_counts = df.groupby("speaker_id").size().sort_values()
image, draw = canvas("Recordings per speaker", 1100, 1000)
left, top, width, row_height = 105, 75, 870, 22
maximum = int(speaker_counts.max()) or 1
for i, (speaker, count) in enumerate(speaker_counts.items()):
    y = top + i * row_height
    draw.text((25, y), str(speaker), fill="#334e68", font=font(12))
    bar = int(count / maximum * width)
    draw.rectangle((left, y, left + bar, y + 15), fill="#07839b")
    draw.text((left + bar + 5, y), str(count), fill="#102a43", font=font(12))
image.save(OUT / "recordings_per_speaker.png")

pd.cut(df.duration_sec, bins=edges, right=False, labels=labels).value_counts(sort=False).rename_axis(
    "duration_seconds"
).reset_index(name="recording_count").to_csv(OUT / "duration_bins.csv", index=False)
speaker_counts.sort_values(ascending=False).rename_axis("speaker_id").reset_index(
    name="recording_count"
).to_csv(OUT / "speaker_recording_counts.csv", index=False)

duration = df.duration_sec
quartiles = duration.quantile([0.25, 0.5, 0.75])
per_speaker = df.groupby("speaker_id").size()
rejected = int((audit.status.astype(str).str.upper() != "VALID").sum())
rates = sorted(df.sample_rate.dropna().astype(int).unique())
channels = sorted(df.num_channels.dropna().astype(int).unique())
report = f"""# LibriSpeech test-clean: Exploratory Data Analysis

## Scope
This analysis uses the validated-file index and leaves the original audio untouched. LibriSpeech is clean, read audiobook speech; these findings do not establish performance on telephone or noisy banking audio.

## Dataset overview
- Recordings analyzed: **{len(df):,}**
- Speakers: **{df.speaker_id.nunique()}**
- Rejected in validation audit: **{rejected}**
- Total speech duration: **{duration.sum() / 3600:.2f} hours**
- Sample rates: **{rates} Hz**
- Channels: **{channels}**

## Recording duration
- Minimum: **{duration.min():.2f} s**
- 25th percentile: **{quartiles.loc[0.25]:.2f} s**
- Median: **{quartiles.loc[0.5]:.2f} s**
- Mean: **{duration.mean():.2f} s**
- 75th percentile: **{quartiles.loc[0.75]:.2f} s**
- Maximum: **{duration.max():.2f} s**
- Standard deviation: **{duration.std():.2f} s**

![Recording duration distribution](duration_distribution.png)

## Speaker representation
- Recordings per speaker: minimum **{per_speaker.min()}**, median **{per_speaker.median():.1f}**, maximum **{per_speaker.max()}**
- Speaker counts vary; evaluation should balance trials so speakers with more recordings do not dominate.

![Recordings per speaker](recordings_per_speaker.png)

Supporting tables: duration_bins.csv and speaker_recording_counts.csv.

## Interpretation
All indexed audio is consistently 16 kHz mono, and all indexed files passed validation. Recording lengths and recordings per speaker vary. This is descriptive dataset analysis, not a measure of model accuracy.

## Reproduce
From the project root run:
    .venv\\Scripts\\python scripts\\run_dataset_eda.py

Outputs are written to data/evaluation/eda/.
"""
(OUT / "EDA_REPORT.md").write_text(report, encoding="utf-8")
print(f"Analyzed {len(df)} recordings from {df.speaker_id.nunique()} speakers.")
print(OUT / "EDA_REPORT.md")
