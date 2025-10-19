"""
dsp_helpers.py
Reusable, CPU-friendly DSP + feature + verification helpers.
Accurate algorithms with safe fallbacks where heavy libs are optional.
"""

import os, io, math, json, yaml, hashlib, warnings
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from scipy import signal, special
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Optional dependencies (safe imports)
# ----------------------------------------------------------------------
try:
    import pyroomacoustics as pra
except Exception:
    pra = None

try:
    import pyloudnorm as pyln
except Exception:
    pyln = None

try:
    from pesq import pesq as pesq_fn
except Exception:
    pesq_fn = None

try:
    from pystoi import stoi as stoi_fn
except Exception:
    stoi_fn = None

try:
    import webrtcvad
except Exception:
    webrtcvad = None

try:
    import parselmouth
    from parselmouth.praat import call
except Exception:
    parselmouth = None
    call = None


# ======================================================================
# Basic Utilities
# ======================================================================

def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_jsonl(path: str, obj: dict):
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def sha256_file(path: str, block_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def load_audio(path: str, target_sr: int = None) -> Tuple[np.ndarray, int]:
    """
    Loads audio file with SoundFile and resamples using Librosa (falls back to polyphase).
    Applies a mild peak safety limit to avoid clipping in later DSP.
    """
    x, sr = sf.read(path, always_2d=False)
    if x.ndim > 1:
        x = np.mean(x, axis=1)
    if target_sr is not None and sr != target_sr:
        try:
            x = librosa.resample(x, orig_sr=sr, target_sr=target_sr, res_type="kaiser_best")
        except Exception:
            x = signal.resample_poly(x, target_sr, sr)
        sr = target_sr
    m = np.max(np.abs(x)) + 1e-12
    if m > 0.999:
        x = 0.999 * x / m
    return x.astype(np.float32), sr

def save_wav(path: str, x: np.ndarray, sr: int, subtype="PCM_16"):
    ensure_dir(os.path.dirname(path))
    sf.write(path, x, sr, subtype=subtype)


# ======================================================================
# Signal Processing Functions
# ======================================================================

def multi_notch_dehum(x: np.ndarray, sr: int, mains_hz=50, harmonics=5, pole_r=0.985) -> np.ndarray:
    """
    Multi-notch IIR dehum filtering for mains interference (and harmonics).
    """
    y = x.copy()
    for k in range(1, harmonics + 1):
        f0 = k * mains_hz
        if f0 >= sr / 2:
            break
        w0 = 2 * math.pi * f0 / sr
        b = [1, -2 * math.cos(w0), 1]
        a = [1, -2 * pole_r * math.cos(w0), pole_r**2]
        y = signal.lfilter(b, a, y)
    return y.astype(np.float32)

def wpe_dereverb(x: np.ndarray, sr: int, taps=10, delay=3, iterations=3) -> np.ndarray:
    """
    Weighted Prediction Error (WPE) dereverberation with broad compatibility.
    """
    if pra is None:
        warnings.warn("pyroomacoustics not installed. Skipping WPE dereverberation.")
        return x

    X = librosa.stft(x, n_fft=1024, hop_length=256, win_length=512, window="hann")
    try:
        if hasattr(pra, "denoise") and hasattr(pra.denoise, "wpe") and hasattr(pra.denoise.wpe, "wpe"):
            Y = pra.denoise.wpe.wpe(X, taps=taps, delay=delay, iterations=iterations)
        elif hasattr(pra, "wpe") and hasattr(pra.wpe, "wpe"):
            Y = pra.wpe.wpe(X, taps=taps, delay=delay, iterations=iterations)
        else:
            warnings.warn("⚠️ No compatible WPE in pyroomacoustics. Skipping dereverberation.")
            return x
    except Exception as e:
        warnings.warn(f"⚠️ WPE dereverberation failed: {e}")
        return x

    y = librosa.istft(Y, hop_length=256, win_length=512, window="hann", length=len(x))
    return y.astype(np.float32)

def spectral_subtraction_mmse_lsa(x: np.ndarray, sr: int, noise_floor_db=-65.0) -> np.ndarray:
    """
    MMSE-LSA inspired denoiser (Ephraim & Malah, 1984) with DD update and exp integral.
    """
    n_fft = 1024
    hop = 256
    win = 512
    S = librosa.stft(x, n_fft=n_fft, hop_length=hop, win_length=win, window="hann")
    mag = np.abs(S)
    ph = np.angle(S)
    frame_e = (mag**2).sum(axis=0)
    k = max(1, int(0.1 * frame_e.size))
    noise_psd = np.median(mag[:, np.argsort(frame_e)[:k]]**2, axis=1, keepdims=True) + 1e-12

    alpha = 0.98
    xi_prev = 1.0
    min_gain = 10 ** (noise_floor_db / 20.0)

    for t in range(mag.shape[1]):
        y2 = mag[:, [t]]**2
        gamma = y2 / (noise_psd + 1e-12)
        xi = alpha * xi_prev + (1 - alpha) * np.maximum(gamma - 1.0, 0.0)
        nu = xi * gamma / (1 + xi + 1e-12)
        # LSA gain approximation via E1 exponential integral
        G = (xi / (1 + xi)) * np.exp(0.5 * special.expn(1, nu + 1e-12))
        mag[:, [t]] = np.maximum(G, min_gain) * mag[:, [t]]
        xi_prev = xi

    Y = mag * np.exp(1j * ph)
    y = librosa.istft(Y, hop_length=hop, win_length=win, window="hann", length=len(x))
    return y.astype(np.float32)

def loudness_normalize_lufs(x: np.ndarray, sr: int, target_lufs=-23.0, truepeak_dbfs=-1.0) -> np.ndarray:
    """
    EBU R128 loudness normalization with true-peak limiting.
    """
    if pyln is None:
        warnings.warn("pyloudnorm not installed. Skipping loudness normalization.")
        return x

    meter = pyln.Meter(sr)
    loud = meter.integrated_loudness(x)
    y = pyln.normalize.loudness(x, loud, target_lufs)
    peak = np.max(np.abs(y)) + 1e-12
    lim = 10 ** (truepeak_dbfs / 20.0)
    if peak > lim:
        y = y * (lim / peak)
    return y.astype(np.float32)


# ======================================================================
# VAD and Segmentation
# ======================================================================

def vad_segments_webrtc(x: np.ndarray, sr: int, mode=2, frame_ms=20,
                        min_sec=2.0, max_sec=20.0, pad_sec=0.05) -> List[Tuple[int, int]]:
    if webrtcvad is None:
        warnings.warn("webrtcvad not installed. Returning full-file as one segment.")
        return [(0, len(x)-1)]
    vad = webrtcvad.Vad(mode)
    frame_len = int(sr * frame_ms / 1000)
    hop = frame_len
    voiced = np.zeros(len(x), dtype=bool)
    for i in range(0, len(x) - frame_len, hop):
        frame = (x[i:i+frame_len] * 32768.0).astype(np.int16).tobytes()
        if vad.is_speech(frame, sr):
            voiced[i:i+frame_len] = True
    idx = np.where(np.diff(voiced.astype(int)) != 0)[0] + 1
    if voiced[0]:
        idx = np.r_[0, idx]
    if voiced[-1]:
        idx = np.r_[idx, len(voiced)]
    out = []
    pad = int(pad_sec * sr)
    for s, e in idx.reshape(-1, 2):
        if e - s < int(min_sec * sr):
            continue
        cursor = s
        while cursor < e:
            ee = min(cursor + int(max_sec * sr), e)
            out.append((max(0, cursor - pad), min(len(x)-1, ee + pad)))
            cursor = ee
    return out


# ======================================================================
# Feature Extraction
# ======================================================================

def compute_mels_mfcc(x: np.ndarray, sr: int, n_fft: int, win_ms: int, hop_ms: int,
                      n_mels: int, n_mfcc: int, fmin: int, fmax: int) -> Dict[str, np.ndarray]:
    win = int(sr * win_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    mel = librosa.feature.melspectrogram(y=x, sr=sr, n_fft=n_fft, win_length=win, hop_length=hop,
                                         n_mels=n_mels, fmin=fmin, fmax=fmax, power=2.0)
    logmel = np.log(mel + 1e-6)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=n_mfcc)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)
    return {"logmel": logmel.astype(np.float32), "mfcc": mfcc.astype(np.float32),
            "d1": d1.astype(np.float32), "d2": d2.astype(np.float32)}

def lpc_coeffs(x: np.ndarray, order: int) -> np.ndarray:
    a = librosa.lpc(x + 1e-9, order=order)
    return a.astype(np.float32)

def _yin_frame_length(sr: int, f0_min: int) -> int:
    """
    Choose a frame length that fits at least two periods of the minimum F0 to avoid warnings.
    """
    two_period = int(max(1, 2 * sr / max(1, f0_min)))
    default = int(sr * 0.025)  # 25 ms
    return max(two_period, default)

def prosody_f0_energy(x: np.ndarray, sr: int, f0_min: int, f0_max: int,
                      hop_ms: int) -> Dict[str, Any]:
    hop = int(sr * hop_ms / 1000)
    win = _yin_frame_length(sr, f0_min)
    f0 = librosa.yin(x, fmin=f0_min, fmax=f0_max, sr=sr, frame_length=win, hop_length=hop)
    f0 = f0[np.isfinite(f0)]
    if len(f0) == 0:
        f0_mean = f0_med = f0_rng = 0.0
    else:
        f0_mean = float(np.mean(f0))
        f0_med = float(np.median(f0))
        f0_rng = float(np.max(f0) - np.min(f0))
    frames = librosa.util.frame(x, frame_length=int(sr * 0.025), hop_length=hop)
    env = (frames**2).mean(axis=0)
    return {"f0_mean": f0_mean, "f0_median": f0_med, "f0_range": f0_rng,
            "energy_mean": float(np.mean(env)), "energy_std": float(np.std(env))}

def phonetic_measures(path_wav: str) -> Dict[str, float]:
    """
    Extracts phonetic and prosodic measures using Praat via Parselmouth.
    Robust to changing 'To Pitch (ac)' signatures; falls back to CC pitch
    and finally to librosa/YIN-based proxies if Praat calls fail.
    """
    # Default (safe) return in case everything fails
    safe = {
        "f0_mean_hz": 0.0,
        "hnr_db": 0.0,
        "jitter_local": 0.0,
        "shimmer_local": 0.0,
        "F1_mean": 0.0,
        "F2_mean": 0.0,
        "F3_mean": 0.0,
    }
    if parselmouth is None or call is None:
        warnings.warn("parselmouth not installed; returning zeros for phonetic measures.")
        return safe

    snd = parselmouth.Sound(path_wav)

    # Try compatible signatures of "To Pitch (ac)" progressively
    pitch = None
    signatures = [
        (snd, "To Pitch (ac)", 0.0, 75, 600),                    # legacy
        (snd, "To Pitch (ac)", 0.0, 75, 600, "yes"),             # some mid versions
        (snd, "To Pitch (ac)", 0.0, 75, 600, "yes", 0.03, 0.45)  # newer ≥0.6.x (silence, voicing)
    ]
    for args in signatures:
        try:
            pitch = call(*args)
            break
        except Exception:
            continue

    # If all (ac) signatures fail, try "To Pitch (cc)" (cross-correlation)
    if pitch is None:
        try:
            pitch = call(snd, "To Pitch (cc)", 0.0, 75, 600)
        except Exception:
            pitch = None

    # If pitch is still None, fallback entirely to librosa for f0; return zeros for other Praat-dependent metrics
    if pitch is None:
        warnings.warn("Praat Pitch estimation failed; using librosa YIN for f0 and zeros for other phonetic metrics.")
        x, sr = snd.values[0], snd.sampling_frequency
        f0 = librosa.yin(x, fmin=75, fmax=600, sr=int(sr), frame_length=_yin_frame_length(int(sr), 75), hop_length=int(sr * 0.01))
        f0 = f0[np.isfinite(f0)]
        f0m = float(np.mean(f0)) if len(f0) else 0.0
        safe["f0_mean_hz"] = f0m
        return safe

    # Normal path with Parselmouth
    try:
        f0m = call(pitch, "Get mean", 0, 0, "Hertz")
        harm = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = call(harm, "Get mean", 0, 0)
        point_proc = call(snd, "To PointProcess (periodic, cc)", 75, 600)
        jitter = call(point_proc, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer = call([snd, point_proc], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        form = call(snd, "To Formant (burg)", 0.0, 5, 5000, 0.025, 50)
        dur = snd.get_total_duration()
        times = np.linspace(0.05, max(0.05, dur - 0.05), num=10)
        F1, F2, F3 = [], [], []
        for t in times:
            F1.append(call(form, "Get value at time", 1, t, "Hertz", "Linear"))
            F2.append(call(form, "Get value at time", 2, t, "Hertz", "Linear"))
            F3.append(call(form, "Get value at time", 3, t, "Hertz", "Linear"))
        return {
            "f0_mean_hz": float(f0m),
            "hnr_db": float(hnr),
            "jitter_local": float(jitter),
            "shimmer_local": float(shimmer),
            "F1_mean": float(np.nanmean(F1)),
            "F2_mean": float(np.nanmean(F2)),
            "F3_mean": float(np.nanmean(F3)),
        }
    except Exception as e:
        warnings.warn(f"Parselmouth feature extraction failed ({e}); returning safe zeros for phonetic metrics.")
        return safe


# ======================================================================
# Annotation Scaffolds
# ======================================================================

def write_textgrid_scaffold(duration: float, tiers: List[str], out_path: str):
    template = f'''File type = "ooTextFile"
Object class = "TextGrid"
xmin = 0
xmax = {duration:.3f}
tiers? <exists>
size = {len(tiers)}
item []:
'''
    idx = 1
    for t in tiers:
        if t in ("words", "phones", "breaks"):
            template += f'''    item [{idx}]:
        class = "IntervalTier"
        name = "{t}"
        xmin = 0
        xmax = {duration:.3f}
        intervals: size = 1
        intervals [1]: xmin = 0; xmax = {duration:.3f}; text = ""
'''
        else:
            template += f'''    item [{idx}]:
        class = "TextTier"
        name = "{t}"
        xmin = 0
        xmax = {duration:.3f}
        points: size = 0
'''
        idx += 1
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(template)


# ======================================================================
# Verification Metrics
# ======================================================================

def snr_proxy(raw: np.ndarray, proc: np.ndarray) -> float:
    return float(10 * np.log10((np.sum(proc**2) + 1e-12) / (np.sum((raw - proc)**2) + 1e-12)))

def si_sdr(ref: np.ndarray, est: np.ndarray) -> float:
    ref = ref.astype(np.float32)
    est = est.astype(np.float32)
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    e_true = alpha * ref
    e_res = est - e_true
    return float(10 * np.log10((np.sum(e_true**2) + 1e-12) / (np.sum(e_res**2) + 1e-12)))

def stoi_score(x: np.ndarray, y: np.ndarray, sr: int) -> float:
    if stoi_fn is None:
        return float("nan")
    return float(stoi_fn(x, y, sr))

def pesq_score(x: np.ndarray, y: np.ndarray, sr: int) -> float:
    if pesq_fn is None:
        return float("nan")
    mode = "wb" if sr >= 16000 else "nb"
    try:
        return float(pesq_fn(sr, x, y, mode))
    except Exception:
        return float("nan")

def lsd_db(x: np.ndarray, y: np.ndarray, n_fft=1024) -> float:
    X = librosa.stft(x, n_fft=n_fft)
    Y = librosa.stft(y, n_fft=n_fft)
    Xd = 10 * np.log10(np.maximum(np.abs(X)**2, 1e-12))
    Yd = 10 * np.log10(np.maximum(np.abs(Y)**2, 1e-12))
    m = min(Xd.shape[1], Yd.shape[1])
    D = (Xd[:, :m] - Yd[:, :m])**2
    return float(np.sqrt(np.mean(D)))

def f0_corr(x: np.ndarray, y: np.ndarray, sr: int, fmin=50, fmax=500) -> float:
    f0x = librosa.yin(x, fmin=fmin, fmax=fmax, sr=sr, frame_length=_yin_frame_length(sr, fmin), hop_length=int(sr*0.01))
    f0y = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr, frame_length=_yin_frame_length(sr, fmin), hop_length=int(sr*0.01))
    m = min(len(f0x), len(f0y))
    if m < 4:
        return float("nan")
    c = np.corrcoef(f0x[:m], f0y[:m])[0, 1]
    return float(c)

def plot_spectrogram_diff(x: np.ndarray, y: np.ndarray, sr: int, out_png: str, n_fft=1024):
    D1 = librosa.amplitude_to_db(np.abs(librosa.stft(x, n_fft=n_fft)), ref=np.max)
    D2 = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=n_fft)), ref=np.max)
    diff = np.abs(D1 - D2)
    ensure_dir(os.path.dirname(out_png))
    plt.figure(figsize=(10, 4))
    plt.imshow(diff, origin="lower", aspect="auto")
    plt.title("Spectrogram difference |raw - processed| (dB)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
