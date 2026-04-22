#!/bin/bash
# ============================================================
#  Mandarin Speech Intelligence Pipeline — One-Shot Demo
#  Usage: bash run.sh
# ============================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

echo -e "${BOLD}${CYAN}"
echo "════════════════════════════════════════════════════════"
echo "   MANDARIN SPEECH INTELLIGENCE PIPELINE"
echo "════════════════════════════════════════════════════════"
echo -e "${NC}"

# ── Check Python ──────────────────────────────────────────
echo -e "${YELLOW}[1/4] Checking Python...${NC}"
if ! command -v python3 &>/dev/null; then
    echo -e "${RED}[FAIL] python3 not found. Install Python 3.10+${NC}"; exit 1
fi
PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo -e "${GREEN}[OK]   Python $PY_VER${NC}"

# ── Check/install Python dependencies ─────────────────────
echo -e "\n${YELLOW}[2/4] Checking Python dependencies...${NC}"
python3 -c "import torch, transformers, sentence_transformers, soundfile, pandas, requests, tqdm" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "  Installing dependencies..."
    pip install -q -r requirements.txt
    echo -e "${GREEN}[OK]   Dependencies installed${NC}"
else
    echo -e "${GREEN}[OK]   All dependencies present${NC}"
fi

# ── Check/start Ollama ────────────────────────────────────
echo -e "\n${YELLOW}[3/4] Checking Ollama + qwen3:1.7b...${NC}"
if ! command -v ollama &>/dev/null; then
    echo -e "${RED}[FAIL] Ollama not installed.${NC}"
    echo "       Install: curl -fsSL https://ollama.com/install.sh | sh"
    exit 1
fi

# Start ollama serve if not already running
if ! curl -s http://localhost:11434/api/tags &>/dev/null; then
    echo "  Starting Ollama server..."
    ollama serve &>/dev/null &
    OLLAMA_PID=$!
    sleep 3
    echo -e "${GREEN}[OK]   Ollama started (PID $OLLAMA_PID)${NC}"
else
    echo -e "${GREEN}[OK]   Ollama already running${NC}"
fi

# Pull qwen3:1.7b if not present
MODELS=$(curl -s http://localhost:11434/api/tags | python3 -c "import sys,json; print(' '.join(m['name'] for m in json.load(sys.stdin).get('models',[])))" 2>/dev/null)
if echo "$MODELS" | grep -q "qwen3:1.7b"; then
    echo -e "${GREEN}[OK]   qwen3:1.7b available${NC}"
else
    echo "  Pulling qwen3:1.7b (1.4GB — one-time download)..."
    ollama pull qwen3:1.7b
    echo -e "${GREEN}[OK]   qwen3:1.7b ready${NC}"
fi

# ── Check audio files ─────────────────────────────────────
echo -e "\n${YELLOW}[4/4] Checking audio files...${NC}"
WAVS=(
    "espionage_samples/espionage_06.wav"
    "espionage_samples/espionage_07.wav"
    "espionage_samples/espionage_08.wav"
    "espionage_samples/espionage_09.wav"
    "espionage_samples/espionage_10.wav"
)
MISSING=0
for WAV in "${WAVS[@]}"; do
    if [ -f "$WAV" ]; then
        echo -e "${GREEN}  [OK]   $WAV${NC}"
    else
        echo -e "${RED}  [MISSING] $WAV${NC}"
        MISSING=$((MISSING+1))
    fi
done
if [ $MISSING -gt 0 ]; then
    echo -e "${RED}[FAIL] $MISSING audio file(s) missing. Check espionage_samples/${NC}"
    exit 1
fi

# ── Run pipeline ──────────────────────────────────────────
echo -e "\n${BOLD}${CYAN}"
echo "════════════════════════════════════════════════════════"
echo "   RUNNING PIPELINE ON 5 AUDIO FILES..."
echo "════════════════════════════════════════════════════════"
echo -e "${NC}"

python3 intel_pipeline_fast.py "${WAVS[@]}"

# ── Show report location ───────────────────────────────────
LATEST_REPORT=$(ls -t intel_fast_report_*.txt 2>/dev/null | head -1)
if [ -n "$LATEST_REPORT" ]; then
    echo -e "\n${BOLD}${GREEN}"
    echo "════════════════════════════════════════════════════════"
    echo "   DONE — Report: $LATEST_REPORT"
    echo "════════════════════════════════════════════════════════"
    echo -e "${NC}"
fi
