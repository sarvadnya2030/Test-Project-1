# Mandarin Speech Intelligence Pipeline

End-to-end pipeline: Chinese military speech → English intelligence report with intent classification.

```
WAV audio → paraformer-zh ASR → qwen3:1.7b translate+classify → LaBSE scoring → .txt report
```

**Benchmark (10 files):** 79.8% meaning preservation | 8.0s per file | paraformer-zh + qwen3:1.7b

---

## Model Stack

| Step | Model | Size | Runs On |
|------|-------|------|---------|
| ASR (primary) | `paraformer-zh` + `fsmn-vad` + `ct-punc` | ~1.1 GB | CPU |
| ASR (fallback) | `openai/whisper-large-v3-turbo` | ~1.6 GB | GPU |
| Translate + Classify | `qwen3:1.7b` via Ollama | 1.4 GB | GPU (pinned) |
| Meaning Scoring | `sentence-transformers/LaBSE` | ~1.8 GB | GPU |

---

## System Requirements

| Component | Requirement |
|-----------|-------------|
| GPU | NVIDIA 8GB+ VRAM (RTX 2070 or better) |
| RAM | 16GB+ |
| Python | 3.10+ |
| CUDA | 12.1+ |
| Ollama | Latest |

---

## Setup

### 1. Clone
```bash
git clone https://github.com/sarvadnya2030/Test-Project-1.git
cd Test-Project-1
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3. Install Ollama + pull model
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:1.7b
ollama serve &
```

### 4. HuggingFace models
Downloaded automatically on first run:
- `paraformer-zh` + VAD + punctuation models (~1.1 GB) — one-time download
- `openai/whisper-large-v3-turbo` (~1.6 GB)
- `sentence-transformers/LaBSE` (~1.8 GB)

---

## Usage

### One-shot demo
```bash
bash run.sh
```
Checks all dependencies, starts Ollama if needed, runs pipeline on 10 audio files, outputs report.

### Manual
```bash
# All 10 default samples
python3 intel_pipeline_fast.py

# Specific files
python3 intel_pipeline_fast.py audio1.wav audio2.wav

# Directory of WAVs
python3 intel_pipeline_fast.py --dir /path/to/wavs

# Custom output path
python3 intel_pipeline_fast.py --out my_report.txt
```

---

## Output

Each run produces:
- `intel_fast_report_<timestamp>.txt` — human-readable intelligence report
- `intel_fast_report_<timestamp>.json` — structured data

### Sample output
```
SAMPLE 01  [espionage_06.wav]
------------------------------------------------------------------------
TRANSCRIPT (ZH) : 专家银狐准备就绪，请求狐狸空中接应指挥官狐狸三号正在接近，预计三分钟到达。
TRANSLATION (EN): Expert Fox is ready, requesting aerial support. Fox Three approaching, ETA 3 minutes.
MEANING SCORE   : 82.4%  (loss: 17.6%)

PRIMARY INTENT  : ASSET_MOVEMENT
CONFIDENCE      : 95%
THREAT LEVEL    : LOW  🟢
REASONING       : Unit repositioning for extraction operation

CODE WORDS      : Fox, SilverFox
```

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  INPUT: WAV audio files (Chinese military speech)           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  PRE-FLIGHT CHECKS                                          │
│  • Ollama running + qwen3:1.7b available                    │
│  • Warmup → pin model to GPU (keep_alive=-1)                │
│  • GPU memory check                                         │
│  • WAV file validation (soundfile)                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  ASR — paraformer-zh  (primary)                             │
│  • Alibaba DAMO, ~4.5% CER on Mandarin                      │
│  • Real-time factor 0.1x (10s audio → 1s inference)         │
│  • Native punctuation via ct-punc                           │
│  • VAD via fsmn-vad                                         │
│  ↓ fallback if failed                                       │
│  whisper-large-v3-turbo  (fallback)                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  TRANSLATE + CLASSIFY — qwen3:1.7b (Ollama)                 │
│  • Single combined prompt (one API call per file)           │
│  • 3 parallel workers                                       │
│  • 3x retry with exponential backoff                        │
│  • JSON validation + fallback prompt on parse error         │
│  • think:false — no wasted tokens                           │
│  • keep_alive:-1 — model stays hot on GPU                   │
│  Output: English translation + intent + threat level        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  LaBSE MEANING SCORE                                        │
│  • Cross-lingual cosine similarity (ZH source vs EN output) │
│  • 0.0 = completely different  |  1.0 = identical meaning   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  CODE-WORD DETECTION                                        │
│  • 15 military callsigns matched in ZH transcript + EN      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  OUTPUT: .txt report + .json                                │
└─────────────────────────────────────────────────────────────┘
```

---

## Intent Classes

| Intent | Meaning |
|--------|---------|
| `STRIKE_AUTHORIZATION` | Order to fire, launch, or engage a target |
| `RECON_REPORT` | Sensor, radar, or satellite data relay |
| `INFILTRATION_CONFIRM` | Agent or unit breached enemy position |
| `CYBER_OPERATION` | Network intrusion or system disruption |
| `ASSET_MOVEMENT` | Repositioning units or extraction |
| `DEFENSE_ACTIVATION` | Defensive system triggered or intercept |
| `STATUS_REPORT` | Readiness or mission completion update |

---

## Benchmark Results

### Current (paraformer-zh + qwen3:1.7b)
| Metric | Value |
|--------|-------|
| Avg meaning preserved | **79.8%** |
| Total time (10 files) | 79.8s |
| Per-file latency | **8.0s** |
| Code-word recall | ~40% |

### Model comparison (research phase)

| ASR | MT Model | Avg Similarity |
|-----|----------|---------------|
| BELLE Whisper | NLLB-600M (baseline) | 64.7% |
| BELLE Whisper | opus-mt-zh-en | 63.2% |
| BELLE Whisper | NLLB-1.3B | 72.5% |
| whisper-turbo | qwen3:1.7b | 70.9% |
| **paraformer-zh** | **qwen3:1.7b** | **79.8%** |
| NLLB-1.3B on gold text | — (ceiling) | 97.2% |

> Key finding: **89% of meaning loss comes from ASR errors, not translation model choice.**
> Switching ASR from whisper → paraformer gave +11.6 points — bigger gain than any MT model swap.

---

## Known Limitations

- ASR still garbles some callsigns (e.g. 苍鹰→苍蝇, 黑剑→黑键)
- Code-word recall ~40% due to phonetic confusions
- Intent classifier defaults to ASSET_MOVEMENT on ambiguous inputs
- Tested on synthetic TTS audio — real field recordings may vary

---

## Project Files

| File | Purpose |
|------|---------|
| `intel_pipeline_fast.py` | Main pipeline — fast, robust, production-ready |
| `intel_pipeline.py` | Full 10-file pipeline (original) |
| `run.sh` | One-shot demo entry point |
| `translation_comparison.py` | 4-model MT comparison experiment |
| `espionage_asr_pipeline.py` | Original baseline pipeline v1 |
| `espionage_asr_pipeline_v2.py` | Upgraded pipeline v2 |
| `meaning_loss_analysis.py` | LaBSE meaning loss analysis script |
| `espionage_samples/` | 10 synthetic Chinese audio samples + results |
| `sample_reports/` | Example output reports |

---

## Hardware Used

| Component | Spec |
|-----------|------|
| GPU | NVIDIA RTX 2070 8GB |
| ASR (paraformer) | CPU |
| Ollama (qwen3:1.7b) | GPU — pinned permanently |
| LaBSE scoring | GPU |
