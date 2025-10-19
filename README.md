# AI-Voice-Cloning---Dataset
Rasool Bux Palijo Digital Speech Processing Pipeline

> **Portable | Modular | Reproducible | Scientific**
>
> End-to-end system for *speech extraction, enhancement, phonetic analysis,*  
> and *AI-ready dataset generation* from public speeches, oral history, and field recordings.

---

## 📘 Overview

**RBP-DSP** is a professional-grade digital speech processing pipeline built for  
scientific preservation and linguistic analysis of *Rasool Bux Palijo’s* spoken legacy.  
It integrates modern **Digital Signal Processing (DSP)**, **Phonetics**, and **Machine Learning (ML)**  
principles into a unified, reproducible framework.

The system automatically performs:

1. 🎧 **Acquisition & Pre-processing**  
2. 🎚️ **Signal Separation & Enhancement Chain**  
3. 🔬 **Phonetic–Linguistic Feature Extraction**  
4. 🗣️ **Annotation & Alignment (TextGrid scaffolds)**  
5. 📊 **Validation & Verification**

All stages produce reproducible metadata, structured logs, and  
scientifically meaningful features for research, training, and archiving.

---

## 🧩 System Architecture

        ┌───────────────────────────────────────────────────────┐
        │                 INPUT DOMAIN (raw_data)                │
        │  Public speeches, TV talks, field recordings           │
        └───────────────────────────────────────────────────────┘
                                   │
                                   ▼
        ┌───────────────────────────────────────────────────────┐
        │           1. ACQUISITION & PRE-PROCESSING              │
        │  - Resampling / Mono downmix / Amplitude calibration   │
        │  - Metadata registration                               │
        └───────────────────────────────────────────────────────┘
                                   │
                                   ▼
        ┌───────────────────────────────────────────────────────┐
        │            2. SIGNAL SEPARATION CHAIN                  │
        │  (Speech Isolation)                                   │
        │  - VAD + Diarization (WebRTC, pyannote)               │
        │  - Source Separation (Conv-TasNet / Demucs)           │
        │  - Dereverberation (WPE)                              │
        │  - Denoising (MMSE-LSA, RNNoise)                      │
        │  - De-hum / De-click / Dynamic Control                │
        └───────────────────────────────────────────────────────┘
                                   │
                                   ▼
        ┌───────────────────────────────────────────────────────┐
        │     3. PHONETIC-LINGUISTIC FEATURE EXTRACTION          │
        │  - Acoustic: MFCC, PNCC, LPC, LogMel                   │
        │  - Prosodic: F0, Energy, Duration, ToBI-LR             │
        │  - Phonetic: Formants, VOT, Jitter, Shimmer, CPP       │
        │  - Linguistic: Syllable timing, Accent, Rhythm         │
        └───────────────────────────────────────────────────────┘
                                   │
                                   ▼
        ┌───────────────────────────────────────────────────────┐
        │        4. ANNOTATION & ALIGNMENT LAYERS                │
        │  - Word / Phone alignment (MFA / Kaldi)                │
        │  - Prosody & tone tiers (Praat / TextGrid)             │
        │  - Discourse / Dialect flags                           │
        └───────────────────────────────────────────────────────┘
                                   │
                                   ▼
        ┌───────────────────────────────────────────────────────┐
        │          5. VALIDATION & VERIFICATION MODULE           │
        │  - SNR / PESQ / STOI / SDR / LSD                       │
        │  - F0-contour correlation                              │
        │  - Spectrogram & envelope similarity                   │
        │  - Automated report generation                         │
        └───────────────────────────────────────────────────────┘
                                   │
                                   ▼
        ┌───────────────────────────────────────────────────────┐
        │               OUTPUT: AI-READY DATASET                │
        │  Clean audio + aligned transcripts + full features     │
        └───────────────────────────────────────────────────────┘


---

## 🧮 Core Mathematical & Algorithmic Foundations

### 1. Pre-processing and Resampling

Discrete-time resampling from \( f_{s0} \to f_{s1} \):

\[
y[n] = \sum_k x[k]\, h_r[n - k/R], \quad R = \frac{f_{s1}}{f_{s0}}
\]
where \( h_r[n] \) is a Kaiser-windowed sinc interpolation kernel.

### 2. Voice Activity Detection (VAD)

Energy-based frame decision:

\[
D(i) =
\begin{cases}
1, & 10\log_{10}\!\left(\frac{E_s(i)}{E_n(i)}\right) > \theta \\
0, & \text{otherwise}
\end{cases}
\]
implemented via WebRTC-VAD with adaptive thresholding.

### 3. Dereverberation (WPE)

Weighted Prediction Error (Yoshioka et al., 2012):

\[
x[n] = \sum_{m=0}^{M-1} g[m]\,s[n-m] + e[n]
\]
Minimize  
\(\displaystyle
E\!\left[\frac{|x[n] - \sum_{m=1}^M g[m]x[n-m]|^2}{\lambda[n]}\right]
\)
iteratively updating \( g[m] \) and \( \lambda[n] \).

### 4. MMSE-LSA Denoising (Ephraim & Malah 1984)

Posterior and prior SNR:

\[
\xi_k = \alpha\xi_{k-1} + (1-\alpha)\max(\gamma_k - 1, 0)
\]
Gain function:

\[
G(\xi,\gamma) = \frac{\xi}{1+\xi}\exp\!\left(\tfrac{1}{2}E_1(\nu)\right)
\]
applied in the STFT magnitude domain.

### 5. Multi-Notch De-hum Filtering

Each harmonic \( kf_h \):

\[
H_k(z)=\frac{1 - 2r\cos(2\pi f_h/f_s)z^{-1}+r^2z^{-2}}
{1 - 2\cos(2\pi f_h/f_s)z^{-1}+z^{-2}}
\]
with \( 0.95<r<0.99 \).

### 6. Loudness Normalization (EBU R128)

\[
g = 10^{(L_T - L_I)/20}, \quad y_n = g\,x_n,
\]
keeping \( \text{TruePeak} < -1\text{ dBFS} \).

### 7. Feature Extraction

- **MFCC / LogMel** via mel filterbanks & DCT  
- **LPC coefficients** from Burg predictor  
- **Prosody:** YIN for \(F_0\), short-term energy  
- **Phonetics:** Formants via LPC poles; Jitter, Shimmer, HNR via Praat  
- **Correlation metrics:** \(r = \text{corr}(F_{0,raw}, F_{0,proc})\)

---

## 🧰 Key Features & Highlights

| Module | Technique | Library |
|--------|------------|----------|
| Resampling | Kaiser-best & polyphase | `librosa`, `scipy` |
| De-hum | Multi-notch adaptive filter | custom |
| Dereverb | WPE (Pyroomacoustics) | `pyroomacoustics` |
| Denoise | MMSE-LSA | custom |
| Loudness | EBU R128 normalization | `pyloudnorm` |
| VAD | Frame-based WebRTC | `webrtcvad` |
| Acoustic features | MFCC, Mel, LPC | `librosa` |
| Phonetic features | Jitter, Shimmer, Formants | `parselmouth` |
| Evaluation | SNR, SI-SDR, STOI, PESQ | `pystoi`, `pesq` |

---

## 📂 Directory Structure

RBP-DSP/
│
├── cfg/
│ ├── core_config.yml
│ └── kpi_thresholds.yml
│
├── raw_data/ # Input speeches, videos
│
├── processed_data/
│ ├── 1_acquisition_preprocessing/
│ ├── 2_signal_separation_chain/
│ ├── 3_phonetic_linguistic_features/
│ ├── 4_annotation_alignment/
│ └── 5_validation_verification/
│
├── metadata/
│ ├── sources.csv
│ ├── segments.csv
│ ├── features_index.csv
│ └── processing_log.jsonl
│
├── scripts/
│ ├── dsp_helpers.py
│ └── video_to_audio_converter.py
│
└── rbp_dsp_master_pipeline.py


---

## ⚙️ Configuration

Edit **`cfg/core_config.yml`**:

```yaml
project:
  samplerate: 16000

paths:
  raw: raw_data
  processed: processed_data
  metadata: metadata

dsp:
  dehum:
    enable: true
    mains_hz: 50
    harmonics: 5
    pole_r: 0.985
  dereverb:
    enable: true
    taps: 10
    delay: 3
    iterations: 3
  denoise:
    enable: true
    noise_floor_db: -65
  loudness:
    enable: true
    target_lufs: -23
    truepeak_dbfs: -1
  vad:
    enable: true
    mode: 2
    frame_ms: 20
    min_sec: 2.0
    max_sec: 20.0
    pad_sec: 0.05

---

## ▶️ Running the Pipeline

# Activate your virtual environment
source .venv/Scripts/activate      # Windows PowerShell

# Execute master pipeline
python rbp_dsp_master_pipeline.py


---

## Output Example

[INFO] Using configuration files:
 - cfg/core_config.yml
 - cfg/kpi_thresholds.yml

[1/5] Acquisition & Pre-processing
100%|██████████| 1/1 [01:12<00:00, 72.10s/it]

[2/5] Signal Separation Chain
100%|██████████| 1/1 [00:58<00:00, 58.02s/it]

[3/5] Feature Extraction
100%|██████████| 532/532 [00:45<00:00, 11.8it/s]

[4/5] Annotation (TextGrid scaffolds)
[5/5] Validation & Verification
✅ Pipeline completed successfully.

All results (audio, CSVs, figures) appear in processed_data/ and metadata/.

---
🧠 Logging and Progress

Real-time progress via TQDM bars per stage.

Structured JSONL logs at:
metadata/processing_log.jsonl

Errors and warnings automatically appended to:
RBP-DSP/pipeline_run.log

Stage summaries saved as CSV.

⚗️ Validation Metrics
Metric	Symbol	Description	Library
SNR	
10
log
⁡
10
∥
𝑦
∥
2
∥
𝑥
−
𝑦
∥
2
10log
10
	​

∥x−y∥
2
∥y∥
2
	​

	Signal clarity	internal
SI-SDR		Scale-invariant quality	internal
STOI		Intelligibility	pystoi
PESQ		Perceived quality (ITU-T P.862)	pesq
LSD		Log-Spectral Distortion	internal
F₀ Corr		Pitch-contour correlation	internal
🧑‍🔬 For Scientists

Provides reproducible DSP/phonetics workflow from raw audio to formant statistics.

Formant extraction via Burg LPC + Praat Formant (Burg).

Statistical robustness: F₀ mean/median/range; energy variance; formant averages.

CSV outputs are compatible with R, MATLAB, or Jupyter notebooks.

Suitable for phonetic correlation studies, dialectology, speech prosody research.

👩‍💻 For Engineers

Fully modular helper functions in scripts/dsp_helpers.py.

Easily extend pipeline with machine learning or speech recognition models.

Supports both headless CLI and IDE (PyCharm/VSCode) execution.

Automatic folder creation ensures portability.

Logging and error-tolerant design for production use.

🎓 For Students

Learn real-world DSP by reading each function:

Resampling: anti-aliasing, polyphase filters

Dereverb: Weighted Prediction Error (WPE)

Noise reduction: MMSE-LSA algorithm

Feature extraction: MFCC, LPC, prosody

Evaluation: SNR, STOI, PESQ, LSD

Visualization: Spectrogram difference maps

A perfect foundation for courses in Speech Processing, Digital Signal Processing, or Machine Learning for Audio.

🧭 Scientific Impact

Bridges linguistics and signal processing for cultural preservation.

Enables cross-disciplinary datasets — combining speech, rhetoric, and phonetics.

Forms a basis for automatic speech scoring, style transfer, and expressive synthesis.

🧩 Troubleshooting
Issue	Cause	Solution
No module named resampy	Missing dependency	pip install resampy
No compatible WPE in pyroomacoustics	Version mismatch	Upgrade: pip install -U pyroomacoustics
Praat 'To Pitch (ac)' failed	Parselmouth version conflict	pip install -U praat-parselmouth
Slow processing	Large files or low CPU	Use smaller sampling rate or disable heavy modules
📚 References

Ephraim, Y. & Malah, D. (1984). Speech enhancement using MMSE short-time spectral amplitude estimator. IEEE Trans. Acoustics, Speech, Signal Processing.

Yoshioka, T. et al. (2012). Dereverberation using Weighted Prediction Error. IEEE TASLP.

Boersma, P. & Weenink, D. (2024). Praat: Doing phonetics by computer.

EBU-R128 Loudness Recommendation.

ITU-T P.862 PESQ Standard.

🧭 Future Roadmap
Phase	Description
2.0	Integration with PyTorch Conv-TasNet / Demucs for true source separation
2.1	Kaldi or MFA-based forced alignment
2.2	MLflow & DVC integration for experiment tracking
3.0	Web dashboard + REST API for dataset search
3.1	Expansion to Sindhi & Urdu phonetic corpora
4.0	Neural vocoder resynthesis (HiFi-GAN)
🏁 Citation

If you use this system in academic work, please cite:

Hussain, M. M. (2025).
RBP-DSP: A Modular Speech Processing Pipeline for Phonetic and Linguistic Analysis.
Internal Technical Report, Germany.

🧾 License

MIT License © 2025 Muhammad Manzar Hussain
This software is open for scientific and educational purposes.

