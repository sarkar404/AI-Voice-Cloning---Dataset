"""
evaluation.py
--------------
Scientific comparison and visualization of input (raw) vs processed audio.

Generates:
  processed_data/5_validation_verification/reports/scientific_evaluation.csv
  processed_data/5_validation_verification/reports/evaluation_summary.txt
  processed_data/5_validation_verification/reports/evaluation.log
  processed_data/5_validation_verification/figures/*.png

Usage:
    python scripts/evaluation.py
"""

import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from scipy.stats import pearsonr
import traceback
import warnings

# --------------------------------------------------------------------------
# Import helper functions from dsp_helpers
# --------------------------------------------------------------------------
from scripts.dsp_helpers import (
    ensure_dir, load_audio, snr_proxy, si_sdr,
    stoi_score, pesq_score, lsd_db, f0_corr
)

warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------------------------------------------------------
# Setup paths
# --------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "raw_data"
PROC_DIR = ROOT_DIR / "processed_data" / "2_signal_separation_chain"
OUT_DIR = ROOT_DIR / "processed_data" / "5_validation_verification"
META_FILE = ROOT_DIR / "metadata" / "sources.csv"

ensure_dir(OUT_DIR / "figures")
ensure_dir(OUT_DIR / "reports")

LOG_FILE = OUT_DIR / "reports" / "evaluation.log"
sr_target = 16000

# --------------------------------------------------------------------------
# Safe plotting utilities
# --------------------------------------------------------------------------
def plot_waveforms(x, y, sr, name, out_dir):
    try:
        t = np.linspace(0, len(x) / sr, len(x))
        plt.figure(figsize=(10, 3))
        plt.plot(t, x, label="Raw", alpha=0.7)
        plt.plot(t, y, label="Processed", alpha=0.7)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.title(f"Waveform Comparison: {name}")
        plt.tight_layout()
        plt.savefig(out_dir / f"{name}_waveform.png", dpi=160)
        plt.close()
    except Exception:
        pass


def plot_spectrograms(x, y, sr, name, out_dir):
    try:
        n_fft = 1024
        X = librosa.amplitude_to_db(np.abs(librosa.stft(x, n_fft=n_fft)), ref=np.max)
        Y = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=n_fft)), ref=np.max)
        diff = np.abs(X - Y)

        fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        for ax, data, title in zip(axs, [X, Y, diff], ["Raw", "Processed", "Δ|Raw–Proc|"]):
            im = ax.imshow(data, aspect="auto", origin="lower", cmap="magma")
            ax.set_title(title)
        fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.6)
        plt.suptitle(f"Spectrogram Comparison: {name}")
        plt.tight_layout()
        plt.savefig(out_dir / f"{name}_spectrograms.png", dpi=160)
        plt.close()
    except Exception:
        pass


def plot_pitch_energy(x, y, sr, name, out_dir):
    try:
        hop = int(sr * 0.01)
        f0x = librosa.yin(x, 50, 500, sr=sr, hop_length=hop)
        f0y = librosa.yin(y, 50, 500, sr=sr, hop_length=hop)
        ex = librosa.feature.rms(y=x, hop_length=hop).flatten()
        ey = librosa.feature.rms(y=y, hop_length=hop).flatten()
        t = np.arange(len(f0x)) * hop / sr

        plt.figure(figsize=(10, 4))
        plt.subplot(2, 1, 1)
        plt.plot(t, f0x, label="Raw F0", color="C0")
        plt.plot(t, f0y, label="Processed F0", color="C1")
        plt.ylabel("Pitch (Hz)")
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(t, ex, label="Raw Energy", color="C0")
        plt.plot(t, ey, label="Processed Energy", color="C1")
        plt.xlabel("Time (s)")
        plt.ylabel("Energy")
        plt.legend()

        plt.suptitle(f"Pitch & Energy Contours: {name}")
        plt.tight_layout()
        plt.savefig(out_dir / f"{name}_pitch_energy.png", dpi=160)
        plt.close()
    except Exception:
        pass


# --------------------------------------------------------------------------
# Evaluation routine
# --------------------------------------------------------------------------
def evaluate_all():
    if not META_FILE.exists():
        print("❌ Missing metadata/sources.csv — run main pipeline first.")
        return

    df_sources = pd.read_csv(META_FILE)
    metrics_rows = []
    errors = []

    print("\n[Evaluation] Comparing input vs processed outputs...")
    for _, row in tqdm(df_sources.iterrows(), total=len(df_sources)):
        try:
            name = Path(row["source_file"]).stem
            raw_path = RAW_DIR / row["source_file"]

            # locate processed version
            proc_path = None
            for sub in ["normalized", "denoised", "dereverbed", "dehum_corrected"]:
                candidates = list((PROC_DIR / sub).glob(f"{name}*.wav"))
                if candidates:
                    proc_path = candidates[0]
                    break
            if not proc_path:
                continue

            x, _ = load_audio(raw_path, sr_target)
            y, _ = load_audio(proc_path, sr_target)
            n = min(len(x), len(y))
            if n < sr_target * 0.5:
                continue  # skip too short

            x, y = x[:n], y[:n]

            # --- Compute metrics ---
            snr = snr_proxy(x, y)
            sisdr = si_sdr(x, y)
            stoi = stoi_score(x, y, sr_target)
            pesq = pesq_score(x, y, sr_target)
            lsd = lsd_db(x, y)
            f0c = f0_corr(x, y, sr_target)

            metrics_rows.append({
                "file": name,
                "SNR_dB": round(snr, 2),
                "SI_SDR_dB": round(sisdr, 2),
                "STOI": round(stoi, 3) if stoi == stoi else np.nan,
                "PESQ": round(pesq, 3) if pesq == pesq else np.nan,
                "LSD_dB": round(lsd, 3),
                "F0_corr": round(f0c, 3) if f0c == f0c else np.nan,
            })

            # --- Plots ---
            fig_dir = OUT_DIR / "figures"
            plot_waveforms(x, y, sr_target, name, fig_dir)
            plot_spectrograms(x, y, sr_target, name, fig_dir)
            plot_pitch_energy(x, y, sr_target, name, fig_dir)

        except Exception as e:
            errors.append(f"{row['source_file']}: {e}")
            with open(LOG_FILE, "a", encoding="utf-8") as log:
                log.write(f"[ERROR] {row['source_file']}:\n{traceback.format_exc()}\n\n")

    # --- Save metrics ---
    df = pd.DataFrame(metrics_rows)
    out_csv = OUT_DIR / "reports" / "scientific_evaluation.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n✅ Metrics saved -> {out_csv}")

    # --- Correlation plot ---
    if not df.empty:
        plt.figure(figsize=(6, 5))
        sns.heatmap(df.drop(columns=["file"]).corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation among Objective Metrics")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "figures" / "metric_correlation_heatmap.png", dpi=160)
        plt.close()

        # --- Summary ---
        summary_txt = OUT_DIR / "reports" / "evaluation_summary.txt"
        with open(summary_txt, "w", encoding="utf-8") as f:
            f.write("RBP-DSP Evaluation Summary\n")
            f.write("=" * 40 + "\n\n")
            f.write(df.describe().to_string())
            f.write("\n\nAverage SNR Gain (dB): %.2f\n" % df["SNR_dB"].mean())
            if len(df) > 1 and df["PESQ"].notna().sum() > 1:
                corr, _ = pearsonr(df["SNR_dB"].fillna(0), df["PESQ"].fillna(0))
                f.write(f"Pearson(SNR,PESQ): {corr:.3f}\n")
        print(f"📊 Summary saved -> {summary_txt}")

    if errors:
        print(f"\n⚠️ Some files failed ({len(errors)}). Check {LOG_FILE} for details.")

    print("\n✅ Evaluation completed successfully.")
    print(f"Results stored in {OUT_DIR}")


# --------------------------------------------------------------------------
if __name__ == "__main__":
    evaluate_all()
