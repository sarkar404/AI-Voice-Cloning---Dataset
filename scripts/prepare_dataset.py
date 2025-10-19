"""
prepare_dataset.py
------------------
Creates a ready-to-use dataset folder from RBP-DSP processed data.

It collects:
 - Speech segments (VAD outputs)
 - Feature files (.npz from acoustic features)
 - Metadata (segments.csv, sources.csv)
and exports them in a standardized structure suitable for ML/DL teams.

Output:  dataset_ready/
├── audio/
├── features/
├── labels.csv
├── dataset_summary.txt
└── dataset_statistics.png

Usage:
    python scripts/prepare_dataset.py
"""

import os
import shutil
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_SEGMENTS = ROOT_DIR / "processed_data" / "2_signal_separation_chain" / "vad_segments"
FEATURES_DIR = ROOT_DIR / "processed_data" / "3_phonetic_linguistic_features" / "acoustic"
PROSODY_DIR = ROOT_DIR / "processed_data" / "3_phonetic_linguistic_features" / "phonetic"
META_DIR = ROOT_DIR / "metadata"
OUT_DIR = ROOT_DIR / "dataset_ready"

AUDIO_OUT = OUT_DIR / "audio"
FEATURE_OUT = OUT_DIR / "features"

for d in [AUDIO_OUT, FEATURE_OUT]:
    os.makedirs(d, exist_ok=True)

# -----------------------------------------------------------------------------
# Load metadata
# -----------------------------------------------------------------------------
segments_csv = META_DIR / "segments.csv"
sources_csv = META_DIR / "sources.csv"

df_segments = pd.read_csv(segments_csv) if segments_csv.exists() else pd.DataFrame()
df_sources = pd.read_csv(sources_csv) if sources_csv.exists() else pd.DataFrame()

print(f"[INFO] Loaded {len(df_segments)} segments and {len(df_sources)} sources.")

# -----------------------------------------------------------------------------
# Copy audio + features into unified structure
# -----------------------------------------------------------------------------
dataset_rows = []
print("\n[STAGE 1] Collecting files and organizing dataset...")

for seg_file in tqdm(sorted(RAW_SEGMENTS.glob("*.wav"))):
    seg_id = seg_file.stem
    audio_dest = AUDIO_OUT / seg_file.name
    feat_src = FEATURES_DIR / f"{seg_id}.npz"
    feat_dest = FEATURE_OUT / f"{seg_id}.npz"

    try:
        shutil.copy2(seg_file, audio_dest)
    except Exception as e:
        print(f"[WARN] Could not copy audio {seg_file.name}: {e}")
        continue

    if feat_src.exists():
        shutil.copy2(feat_src, feat_dest)

    # link metadata
    seg_row = df_segments[df_segments["segment_id"] == seg_id].to_dict("records")
    meta = seg_row[0] if seg_row else {}
    dataset_rows.append({
        "segment_id": seg_id,
        "source_file": meta.get("source_file", ""),
        "t_start_s": meta.get("t_start_s", 0.0),
        "t_end_s": meta.get("t_end_s", 0.0),
        "duration_s": round(meta.get("t_end_s", 0) - meta.get("t_start_s", 0), 3)
    })

# -----------------------------------------------------------------------------
# Merge with feature-based metadata if available
# -----------------------------------------------------------------------------
prosody_summary = PROSODY_DIR / "phonetic_prosody_summary.csv"
if prosody_summary.exists():
    df_prosody = pd.read_csv(prosody_summary)
    df_prosody.rename(columns={"segment": "segment_id"}, inplace=True)
else:
    df_prosody = pd.DataFrame()

df_main = pd.DataFrame(dataset_rows)
if not df_prosody.empty:
    df_main = pd.merge(df_main, df_prosody, on="segment_id", how="left")

# -----------------------------------------------------------------------------
# Save labels CSV
# -----------------------------------------------------------------------------
labels_csv = OUT_DIR / "labels.csv"
df_main.to_csv(labels_csv, index=False, encoding="utf-8")
print(f"[INFO] Saved labels -> {labels_csv}")

# -----------------------------------------------------------------------------
# Optional: Compute dataset-level statistics
# -----------------------------------------------------------------------------
print("\n[STAGE 2] Computing dataset statistics...")

durations = df_main["duration_s"].dropna().values if "duration_s" in df_main else []
n_segments = len(df_main)
n_audio = len(list(AUDIO_OUT.glob("*.wav")))
n_feat = len(list(FEATURE_OUT.glob("*.npz")))

summary_txt = OUT_DIR / "dataset_summary.txt"
with open(summary_txt, "w", encoding="utf-8") as f:
    f.write("=== RBP-DSP Dataset Summary ===\n")
    f.write(f"Total segments: {n_segments}\n")
    f.write(f"Audio files: {n_audio}\n")
    f.write(f"Feature files: {n_feat}\n")
    f.write(f"Average segment duration: {np.mean(durations):.2f} s\n")
    f.write(f"Median duration: {np.median(durations):.2f} s\n")
    f.write(f"Total speech hours: {np.sum(durations)/3600:.2f} h\n")

# -----------------------------------------------------------------------------
# Plot duration distribution
# -----------------------------------------------------------------------------
if len(durations) > 0:
    plt.figure(figsize=(7, 4))
    sns.histplot(durations, bins=40, color="skyblue", edgecolor="black")
    plt.xlabel("Segment duration (s)")
    plt.ylabel("Count")
    plt.title("Segment Duration Distribution")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "dataset_statistics.png", dpi=160)
    plt.close()

print(f"[INFO] Saved summary and statistics -> {summary_txt}")

# -----------------------------------------------------------------------------
# Optional: Normalize feature vectors globally
# -----------------------------------------------------------------------------
print("\n[STAGE 3] Normalizing features (optional)...")
try:
    from sklearn.preprocessing import StandardScaler
    feats = []
    for npz in FEATURE_OUT.glob("*.npz"):
        data = np.load(npz)
        if "mfcc" in data:
            feats.append(data["mfcc"].mean(axis=1))
    if feats:
        feats = np.stack(feats)
        scaler = StandardScaler()
        scaler.fit(feats)
        np.save(OUT_DIR / "feature_scaler_mean.npy", scaler.mean_)
        np.save(OUT_DIR / "feature_scaler_std.npy", scaler.scale_)
        print(f"[INFO] Saved feature normalization stats (mean/std).")
except Exception as e:
    print(f"[WARN] Feature normalization skipped: {e}")

# -----------------------------------------------------------------------------
# Final message
# -----------------------------------------------------------------------------
print("\n✅ Dataset preparation complete!")
print(f"Dataset ready for ML/DL at: {OUT_DIR}")
print("Contents:")
print(f" - Audio files:     {n_audio}")
print(f" - Feature vectors: {n_feat}")
print(f" - Labels CSV:      {labels_csv}")
print(f" - Summary:         {summary_txt}")
print(f" - Statistics:      {OUT_DIR/'dataset_statistics.png'}")
