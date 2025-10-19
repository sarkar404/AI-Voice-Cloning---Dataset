"""
rbp_dsp_master_pipeline.py
--------------------------
Portable, CPU-friendly end-to-end DSP pipeline:
Acquisition → Separation Chain → Features → Annotation → Verification.

It auto-creates all required folders and writes structured outputs.

Run:
    python rbp_dsp_master_pipeline.py
or
    python scripts/rbp_dsp_master_pipeline.py
"""

import os, glob, time, sys, logging, warnings
from typing import Dict, Any
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from pathlib import Path
from contextlib import contextmanager

# ---------------------------------------------------------------------
# 0) PORTABLE PROJECT ROOT DETECTION
# ---------------------------------------------------------------------
current_path = Path(__file__).resolve()
if (current_path.parent / "cfg").exists():
    ROOT_DIR = current_path.parent
else:
    ROOT_DIR = current_path.parent.parent

scripts_path = ROOT_DIR / "scripts"
if str(scripts_path) not in sys.path:
    sys.path.insert(0, str(scripts_path))

# ---------------------------------------------------------------------
# 1) IMPORT DSP HELPERS
# ---------------------------------------------------------------------
from scripts.dsp_helpers import (
    load_yaml, ensure_dir, load_audio, save_wav, save_jsonl, sha256_file,
    multi_notch_dehum, wpe_dereverb, spectral_subtraction_mmse_lsa,
    loudness_normalize_lufs, vad_segments_webrtc, compute_mels_mfcc,
    lpc_coeffs, prosody_f0_energy, phonetic_measures, write_textgrid_scaffold,
    snr_proxy, si_sdr, stoi_score, pesq_score, lsd_db, f0_corr, plot_spectrogram_diff
)

# ---------------------------------------------------------------------
# 2) CONFIG PATHS & LOGGING
# ---------------------------------------------------------------------
CFG_CORE = ROOT_DIR / "cfg" / "core_config.yml"
CFG_KPI = ROOT_DIR / "cfg" / "kpi_thresholds.yml"

if not CFG_CORE.exists() or not CFG_KPI.exists():
    print(f"❌ Configuration files not found.\nExpected:\n - {CFG_CORE}\n - {CFG_KPI}")
    raise FileNotFoundError("Missing configuration files in cfg/")

LOG_PATH = ROOT_DIR / "rbp_dsp.log"
ensure_dir(str(ROOT_DIR))  # just in case

# Configure logging: console + file
logger = logging.getLogger("RBP-DSP")
logger.setLevel(logging.INFO)
logger.handlers.clear()

_fmt = logging.Formatter(
    fmt="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
fh.setFormatter(_fmt)
fh.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(_fmt)
ch.setLevel(logging.INFO)

logger.addHandler(fh)
logger.addHandler(ch)

# Send warnings to logging
def _warn_to_log(message, category, filename, lineno, file=None, line=None):
    logger.warning(f"{category.__name__}: {message} ({Path(filename).name}:{lineno})")
warnings.showwarning = _warn_to_log  # route warnings into our logger

logger.info(f"Using configuration files:\n - {CFG_CORE}\n - {CFG_KPI}")

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
@contextmanager
def stage(name: str):
    logger.info(f"▶ START: {name}")
    t0 = time.time()
    try:
        yield
    finally:
        dt = time.time() - t0
        logger.info(f"✓ END: {name} (elapsed: {dt:.2f}s)")

def build_tree(cfg: Dict[str, Any]) -> Dict[str, str]:
    root_proc = cfg["paths"]["processed"]
    meta_dir = cfg["paths"]["metadata"]

    # master processed folders per stage
    stage_dirs = {
        "acq": os.path.join(root_proc, "1_acquisition_preprocessing"),
        "sep": os.path.join(root_proc, "2_signal_separation_chain"),
        "feat": os.path.join(root_proc, "3_phonetic_linguistic_features"),
        "ann": os.path.join(root_proc, "4_annotation_alignment"),
        "ver": os.path.join(root_proc, "5_validation_verification"),
    }
    for d in stage_dirs.values():
        ensure_dir(d)

    # subfolders for the separation chain
    sep_sub = {
        "dehum": os.path.join(stage_dirs["sep"], "dehum_corrected"),
        "derev": os.path.join(stage_dirs["sep"], "dereverbed"),
        "denoi": os.path.join(stage_dirs["sep"], "denoised"),
        "norm": os.path.join(stage_dirs["sep"], "normalized"),
        "vad": os.path.join(stage_dirs["sep"], "vad_segments"),
        "sepstems": os.path.join(stage_dirs["sep"], "separated_sources")
    }
    # features
    feat_sub = {
        "acoustic": os.path.join(stage_dirs["feat"], "acoustic"),
        "prosodic": os.path.join(stage_dirs["feat"], "prosodic"),
        "phonetic": os.path.join(stage_dirs["feat"], "phonetic"),
        "ling": os.path.join(stage_dirs["feat"], "linguistic")
    }
    # annotation
    ann_sub = {
        "textgrids": os.path.join(stage_dirs["ann"], "textgrids"),
        "transcripts": os.path.join(stage_dirs["ann"], "transcripts"),
        "alignments": os.path.join(stage_dirs["ann"], "alignments")
    }
    # verification
    ver_sub = {
        "reports": os.path.join(stage_dirs["ver"], "reports"),
        "figures": os.path.join(stage_dirs["ver"], "figures")
    }

    # metadata
    ensure_dir(meta_dir)
    for d in [*sep_sub.values(), *feat_sub.values(), *ann_sub.values(), *ver_sub.values()]:
        ensure_dir(d)

    return {
        "root_proc": root_proc, "meta_dir": meta_dir,
        **stage_dirs,
        **{f"sep_{k}": v for k, v in sep_sub.items()},
        **{f"feat_{k}": v for k, v in feat_sub.items()},
        **{f"ann_{k}": v for k, v in ann_sub.items()},
        **{f"ver_{k}": v for k, v in ver_sub.items()},
    }

# ---------------------------------------------------------------------
# 4) MAIN PIPELINE
# ---------------------------------------------------------------------
def main():
    print(f"[INFO] Using configuration files:\n - {CFG_CORE}\n - {CFG_KPI}")
    print(f"[INFO] Working directory set to: {ROOT_DIR}")
    logger.info(f"Working directory: {ROOT_DIR}")

    cfg = load_yaml(str(CFG_CORE))
    kpi = load_yaml(str(CFG_KPI))
    sr = cfg["project"]["samplerate"]
    paths_cfg = cfg["paths"]
    dirs = build_tree(cfg)

    raw_dir = paths_cfg["raw"]
    meta_dir = paths_cfg["metadata"]
    sources_csv = os.path.join(meta_dir, "sources.csv")
    segments_csv = os.path.join(meta_dir, "segments.csv")
    features_idx_csv = os.path.join(meta_dir, "features_index.csv")
    log_jsonl = os.path.join(meta_dir, "processing_log.jsonl")

    # ----------------------- DISCOVER INPUT FILES -----------------------
    in_files = []
    for ext in ("*.wav", "*.mp3", "*.flac", "*.aac", "*.m4a"):
        in_files += glob.glob(os.path.join(raw_dir, ext))
    in_files = sorted(in_files)
    logger.info(f"Discovered {len(in_files)} input file(s) in {raw_dir}")

    if not in_files:
        msg = f"No audio files found in '{raw_dir}'. Please place .wav or .mp3 files there."
        print(f"\n⚠️ {msg}")
        logger.warning(msg)
        return

    # ---------------- 1) ACQUISITION & PREPROCESSING -------------------
    with stage("1) Acquisition & Pre-processing"):
        rows_sources = []
        for f in tqdm(in_files, desc="Acquisition", unit="file"):
            x, _sr = load_audio(f, target_sr=cfg["dsp"]["resample_rate"])
            out = os.path.join(dirs["acq"], os.path.splitext(os.path.basename(f))[0] + "_resampled.wav")
            save_wav(out, x, _sr)
            row = {
                "source_file": os.path.basename(f),
                "sha256": sha256_file(f),
                "sr_out": _sr,
                "duration_s": round(len(x)/_sr, 3),
                "acq_path": out
            }
            rows_sources.append(row)
            save_jsonl(log_jsonl, {"stage": "acquisition_preprocessing", "file": f, "acq_out": out, "t": int(time.time())})
        pd.DataFrame(rows_sources).to_csv(sources_csv, index=False, encoding="utf-8")
        logger.info(f"Wrote sources metadata -> {sources_csv}")

    # ---------------- 2) SIGNAL SEPARATION CHAIN -----------------------
    with stage("2) Signal Separation Chain"):
        rows_segments = []
        for r in tqdm(rows_sources, desc="Separation", unit="file"):
            x, _ = load_audio(r["acq_path"], target_sr=sr)
            # De-hum
            if cfg["dsp"]["dehum"]["enable"]:
                x = multi_notch_dehum(x, sr, cfg["dsp"]["dehum"]["mains_hz"],
                                      cfg["dsp"]["dehum"]["harmonics"], cfg["dsp"]["dehum"]["pole_r"])
                out_dehum = os.path.join(dirs["sep_dehum"], os.path.basename(r["acq_path"]).replace("_resampled", "_dehum"))
                save_wav(out_dehum, x, sr)

            # Dereverb (WPE)
            if cfg["dsp"]["dereverb"]["enable"]:
                x = wpe_dereverb(x, sr, cfg["dsp"]["dereverb"]["taps"],
                                 cfg["dsp"]["dereverb"]["delay"], cfg["dsp"]["dereverb"]["iterations"])
                out_derev = os.path.join(dirs["sep_derev"], os.path.basename(r["acq_path"]).replace("_resampled", "_dereverb"))
                save_wav(out_derev, x, sr)

            # Denoise
            if cfg["dsp"]["denoise"]["enable"]:
                x = spectral_subtraction_mmse_lsa(x, sr, cfg["dsp"]["denoise"]["noise_floor_db"])
                out_denoi = os.path.join(dirs["sep_denoi"], os.path.basename(r["acq_path"]).replace("_resampled", "_denoise"))
                save_wav(out_denoi, x, sr)

            # Loudness
            if cfg["dsp"]["loudness"]["enable"]:
                x = loudness_normalize_lufs(x, sr, cfg["dsp"]["loudness"]["target_lufs"], cfg["dsp"]["loudness"]["truepeak_dbfs"])
                out_norm = os.path.join(dirs["sep_norm"], os.path.basename(r["acq_path"]).replace("_resampled", "_normalized"))
                save_wav(out_norm, x, sr)

            # VAD → segments
            if cfg["dsp"]["vad"]["enable"]:
                segs = vad_segments_webrtc(x, sr, cfg["dsp"]["vad"]["mode"], cfg["dsp"]["vad"]["frame_ms"],
                                           cfg["dsp"]["vad"]["min_sec"], cfg["dsp"]["vad"]["max_sec"], cfg["dsp"]["vad"]["pad_sec"])
                base = os.path.splitext(os.path.basename(r["acq_path"]))[0].replace("_resampled", "")
                for i, (s, e) in enumerate(segs):
                    s = int(s); e = int(e)
                    seg = x[s:e]
                    seg_id = f"{base}_{i:04d}"
                    seg_path = os.path.join(dirs["sep_vad"], seg_id + ".wav")
                    save_wav(seg_path, seg, sr)
                    rows_segments.append({
                        "segment_id": seg_id, "source_file": r["source_file"],
                        "t_start_s": round(s/sr, 3), "t_end_s": round(e/sr, 3), "sr_hz": sr
                    })
            save_jsonl(log_jsonl, {"stage": "signal_separation_chain", "file": r["source_file"], "t": int(time.time())})
        pd.DataFrame(rows_segments).to_csv(segments_csv, index=False, encoding="utf-8")
        logger.info(f"Wrote segments index -> {segments_csv}")

    # ---------------- 3) FEATURE EXTRACTION ----------------------------
    with stage("3) Feature Extraction"):
        seg_files = sorted(glob.glob(os.path.join(dirs["sep_vad"], "*.wav")))
        feat_rows = []
        for seg_file in tqdm(seg_files, desc="Features", unit="seg"):
            x, _ = load_audio(seg_file, target_sr=sr)

            acc = compute_mels_mfcc(x, sr,
                                    n_fft=cfg["features"]["acoustic"]["n_fft"],
                                    win_ms=cfg["features"]["acoustic"]["win_ms"],
                                    hop_ms=cfg["features"]["acoustic"]["hop_ms"],
                                    n_mels=cfg["features"]["acoustic"]["n_mels"],
                                    n_mfcc=cfg["features"]["acoustic"]["n_mfcc"],
                                    fmin=cfg["features"]["acoustic"]["fmin"],
                                    fmax=cfg["features"]["acoustic"]["fmax"])
            npz_path = os.path.join(dirs["feat_acoustic"], os.path.basename(seg_file).replace(".wav", ".npz"))
            np.savez_compressed(npz_path, **acc)

            lpc = lpc_coeffs(x, cfg["features"]["acoustic"]["lpc_order"])
            lpc_path = os.path.join(dirs["feat_ling"], os.path.basename(seg_file).replace(".wav", "_lpc.npy"))
            np.save(lpc_path, lpc)

            pro = prosody_f0_energy(x, sr,
                                    cfg["features"]["prosodic"]["f0_min_hz"],
                                    cfg["features"]["prosodic"]["f0_max_hz"],
                                    cfg["features"]["prosodic"]["hop_ms"])
            phn = phonetic_measures(seg_file)
            row = {"segment": os.path.basename(seg_file), **pro, **phn}
            feat_rows.append(row)

        phon_csv = os.path.join(dirs["feat_phonetic"], "phonetic_prosody_summary.csv")
        pd.DataFrame(feat_rows).to_csv(phon_csv, index=False, encoding="utf-8")
        logger.info(f"Wrote phonetic/prosody summary -> {phon_csv}")

    # ---------------- 4) ANNOTATION & ALIGNMENT -----------------------
    with stage("4) Annotation (TextGrid scaffolds)"):
        seg_files = sorted(glob.glob(os.path.join(dirs["sep_vad"], "*.wav")))
        for seg_file in tqdm(seg_files, desc="TextGrids", unit="seg"):
            dur = librosa.get_duration(path=seg_file)
            tg_path = os.path.join(dirs["ann_textgrids"], os.path.basename(seg_file).replace(".wav", ".TextGrid"))
            write_textgrid_scaffold(dur, cfg["annotation"]["textgrid_tiers"], tg_path)

    # ---------------- 5) VALIDATION & VERIFICATION --------------------
    with stage("5) Validation & Verification"):
        ver_rows = []
        fig_dir = dirs["ver_figures"]
        ensure_dir(fig_dir)
        for r in tqdm(rows_sources, desc="Verification", unit="file"):
            raw_path = os.path.join(cfg["paths"]["raw"], r["source_file"])
            norm_path = os.path.join(dirs["sep_norm"], os.path.basename(r["acq_path"]).replace("_resampled", "_normalized"))
            if not os.path.exists(norm_path):
                candidates = ["_denoise", "_dereverb", "_dehum", "_resampled"]
                found = None
                for tag in candidates:
                    p = os.path.join(dirs["sep_denoi"], os.path.basename(r["acq_path"]).replace("_resampled", tag))
                    if os.path.exists(p):
                        found = p; break
                if found is None: found = r["acq_path"]
                norm_path = found

            x, _ = load_audio(raw_path, target_sr=sr)
            y, _ = load_audio(norm_path, target_sr=sr)
            n = min(len(x), len(y))
            x, y = x[:n], y[:n]

            snr = snr_proxy(x, y)
            sisdr = si_sdr(x, y)
            stoi = stoi_score(x, y, sr)
            pesq = pesq_score(x, y, sr)
            lsd = lsd_db(x, y, n_fft=1024)
            f0c = f0_corr(x, y, sr)

            diff_png = os.path.join(fig_dir, os.path.splitext(r["source_file"])[0] + "_diff.png")
            plot_spectrogram_diff(x, y, sr, diff_png)

            ver_rows.append({
                "file": r["source_file"],
                "snr_proxy_db": round(snr, 2),
                "si_sdr_db": round(sisdr, 2),
                "stoi": round(stoi, 3) if stoi == stoi else "",
                "pesq": round(pesq, 3) if pesq == pesq else "",
                "lsd_db": round(lsd, 3),
                "f0_corr": round(f0c, 3) if f0c == f0c else ""
            })
        ver_df = pd.DataFrame(ver_rows)
        ver_csv = os.path.join(dirs["ver_reports"], "verification_summary.csv")
        ensure_dir(dirs["ver_reports"])
        ver_df.to_csv(ver_csv, index=False, encoding="utf-8")
        logger.info(f"Wrote verification summary -> {ver_csv}")

        # KPI checks
        k = kpi["kpi"]
        passed = []
        for _, row in ver_df.iterrows():
            ok = True
            if row["snr_proxy_db"] < k["snr_gain_db_min"]: ok = False
            if row["si_sdr_db"] < k["si_sdr_min"]: ok = False
            if isinstance(row["stoi"], float) and row["stoi"] < k["stoi_min"]: ok = False
            if isinstance(row["pesq"], float) and row["pesq"] < k["pesq_min"]: ok = False
            if row["lsd_db"] > k["lsd_db_max"]: ok = False
            if isinstance(row["f0_corr"], float) and row["f0_corr"] < k["f0_corr_min"]: ok = False
            passed.append(ok)
        ver_df["pass"] = passed
        ver_df.to_csv(ver_csv, index=False, encoding="utf-8")
        logger.info("KPI flags appended to verification summary.")

    # Feature index (acoustic npz list)
    with stage("Feature Index"):
        feat_idx_rows = [{"segment_npz": os.path.basename(f)} for f in sorted(glob.glob(os.path.join(dirs["feat_acoustic"], "*.npz")))]
        pd.DataFrame(feat_idx_rows).to_csv(features_idx_csv, index=False, encoding="utf-8")
        logger.info(f"Wrote feature index -> {features_idx_csv}")

    print("\nVerification summary ->", ver_csv)
    print("\n✅ Pipeline completed successfully.")
    print("Outputs:")
    print(" - processed_data/1_acquisition_preprocessing/")
    print(" - processed_data/2_signal_separation_chain/")
    print(" - processed_data/3_phonetic_linguistic_features/")
    print(" - processed_data/4_annotation_alignment/")
    print(" - processed_data/5_validation_verification/")
    print(" - metadata/*.csv and processing_log.jsonl")
    print(f" - logs at: {LOG_PATH}")

# ---------------------------------------------------------------------
# 5) RUN PIPELINE
# ---------------------------------------------------------------------
if __name__ == "__main__":
    os.chdir(ROOT_DIR)
    print(f"[INFO] Working directory set to: {ROOT_DIR}")
    logger.info(f"Working directory set to: {ROOT_DIR}")
    main()
