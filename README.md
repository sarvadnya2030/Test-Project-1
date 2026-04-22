# Mandarin Speech Intelligence Pipeline

End-to-end pipeline: Chinese speech → English intelligence report with intent classification.

```
WAV audio → BELLE/Whisper ASR → NLLB-1.3B translation → qwen3:1.7b intent classification → .txt report
```

**Benchmark:** 70.9% meaning preservation | 4.9s per file | qwen3:1.7b pinned to GPU

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

## Setup (Reproduce from Scratch)

### 1. Clone the repo
```bash
git clone https://github.com/sarvadnya2030/Test-Project-1.git
cd Test-Project-1
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3. Install and start Ollama
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull qwen3:1.7b (the intent+translation model)
ollama pull qwen3:1.7b

# Start Ollama server (keep this running in background)
ollama serve &
```

### 4. Download HuggingFace models (auto-downloaded on first run)
These download automatically when you first run the pipeline:
- `BELLE-2/Belle-whisper-large-v3-zh` — Chinese ASR (~3GB)
- `openai/whisper-large-v3-turbo` — Fast ASR (~1.6GB)
- `facebook/nllb-200-1.3B` — Translation model (~2.7GB)
- `sentence-transformers/LaBSE` — Meaning scorer (~1.8GB)

---

## Usage

### Fast Pipeline (recommended — top-5 best files, ~25s)
```bash
# Run on default top-5 samples
python3 intel_pipeline_fast.py

# Run on specific WAV files
python3 intel_pipeline_fast.py audio1.wav audio2.wav

# Run on a directory of WAVs
python3 intel_pipeline_fast.py --dir /path/to/wavs

# Custom output file
python3 intel_pipeline_fast.py --out my_report.txt
```

### Full Pipeline (all 10 files, ~15 min)
```bash
python3 intel_pipeline.py
```

### Translation Model Comparison (research)
```bash
python3 translation_comparison.py
```

---

## Output

Each run produces two files:
- `intel_fast_report_<timestamp>.txt` — human-readable intelligence report
- `intel_fast_report_<timestamp>.json` — structured data for downstream processing

### Sample report structure
```
SAMPLE 01  [espionage_06.wav]
TRANSCRIPT (ZH) : 专家银湖准备继续请求狐狸空中接引...
TRANSLATION (EN): Expert Silver Fox is preparing to request air pickup...
MEANING SCORE   : 70.6%  (loss: 29.4%)

PRIMARY INTENT  : ASSET_MOVEMENT
CONFIDENCE      : 95%
THREAT LEVEL    : LOW  🟢
REASONING       : Unit repositioning for extraction

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
│  STEP 1: PRE-FLIGHT CHECKS                                  │
│  • Ollama running + qwen3:1.7b available                    │
│  • Warmup model → pin to GPU (keep_alive=-1)                │
│  • GPU memory check                                         │
│  • WAV file validation                                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: ASR — openai/whisper-large-v3-turbo                │
│  • Chinese-language transcription                           │
│  • GPU if free > 2GB, else CPU                              │
│  Output: Mandarin text transcript                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 3+4: TRANSLATE + CLASSIFY — qwen3:1.7b (Ollama)       │
│  • Combined prompt: translation + intent in one call        │
│  • Parallel execution (3 workers)                           │
│  • 3x retry with backoff on failure                         │
│  • JSON validation + fallback prompt on parse error         │
│  Output: English translation + intent + threat level        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 5: LaBSE MEANING SCORING                              │
│  • Cross-lingual semantic similarity (ZH source vs EN out)  │
│  Output: 0.0–1.0 meaning preservation score                │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 6: CODE-WORD DETECTION                                │
│  • 15 military callsigns matched in ZH + EN                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  OUTPUT: intel_fast_report_<timestamp>.txt + .json          │
└─────────────────────────────────────────────────────────────┘
```

---

## Benchmark Results

| Metric | Value |
|--------|-------|
| Meaning preservation (avg) | 70.9% |
| Pipeline latency (5 files) | 24.6s |
| Per-file latency | **4.9s** |
| Code-word recall | 40% (limited by ASR errors) |
| ASR model | whisper-large-v3-turbo |
| Translation model | qwen3:1.7b (Ollama) |

### Model comparison (from research phase)

| Model | Input | Avg Similarity |
|-------|-------|---------------|
| NLLB-600M (baseline) | ASR | 64.7% |
| opus-mt-zh-en | ASR | 63.2% |
| **NLLB-1.3B (best)** | **ASR** | **72.5%** |
| NLLB-1.3B (ceiling) | Gold text | 97.2% |

> Key finding: **89% of meaning loss comes from ASR errors, not translation.**

---

## Project Files

| File | Purpose |
|------|---------|
| `intel_pipeline_fast.py` | **Main pipeline** — fast, robust, production-ready |
| `intel_pipeline.py` | Full 10-file pipeline with all checks |
| `translation_comparison.py` | 4-model MT comparison experiment |
| `espionage_asr_pipeline.py` | Original baseline pipeline v1 |
| `espionage_asr_pipeline_v2.py` | Upgraded pipeline v2 |
| `meaning_loss_analysis.py` | LaBSE meaning loss analysis |
| `espionage_samples/` | 10 synthetic Chinese audio samples + results |
| `sample_reports/` | Example output reports |

---

## Known Limitations

- ASR garbles military callsigns (e.g. 苍鹰→苍蝇, 黑剑→黑箭) — phonetic confusions
- Intent classification occasionally wrong on electronic warfare samples
- Tested on synthetic TTS audio — real field recordings may perform differently
- Requires Ollama running in background at all times

---

## Hardware Used

| Component | Spec |
|-----------|------|
| GPU | NVIDIA RTX 2070 8GB |
| ASR inference | CUDA (GPU) |
| Translation | CPU (Ollama holds GPU) |
| Ollama models | qwen3:1.7b pinned to GPU |
