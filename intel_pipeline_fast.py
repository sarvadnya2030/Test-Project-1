#!/usr/bin/env python3
"""
Mandarin Speech Intelligence Pipeline — FAST + ROBUST
- qwen3:1.7b kept warm on GPU permanently (keep_alive=-1)
- Whisper-large-v3-turbo for ASR (4x faster than large-v3)
- Translate + classify in ONE combined Ollama call
- Parallel Ollama calls across files
- Full pre-flight checks, per-sample fault isolation, retry logic

Usage:
    python3 intel_pipeline_fast.py                        # top-5 wavs
    python3 intel_pipeline_fast.py file1.wav file2.wav    # specific files
    python3 intel_pipeline_fast.py --dir /path/to/wavs    # directory
"""

import os, sys, gc, re, json, time, warnings, argparse, unicodedata
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings("ignore")

import torch
import requests
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434"
OLLAMA_MODEL = "qwen3:1.7b"
OLLAMA_RETRIES = 3
OLLAMA_TIMEOUT = 60

TOP5_DEFAULT = [
    "espionage_samples/espionage_06.wav",
    "espionage_samples/espionage_07.wav",
    "espionage_samples/espionage_08.wav",
    "espionage_samples/espionage_09.wav",
    "espionage_samples/espionage_10.wav",
]

CODE_WORD_MAP = {
    "狐狸": "Fox",         "鲨鱼": "Shark",
    "苍鹰": "BlueEagle",   "黑剑": "BlackSword",
    "幽灵": "Ghost",       "铁拳": "IronFist",
    "夜莺": "Nightingale", "毒蛇": "Viper",
    "银狐": "SilverFox",   "红旗": "RedFlag",
    "钢龙": "SteelDragon", "暗流": "DarkCurrent",
    "铁网": "IronNet",     "雷霆": "Thunder",
    "迷雾": "Fog",
}

INTENT_CLASSES = [
    "STRIKE_AUTHORIZATION", "RECON_REPORT", "INFILTRATION_CONFIRM",
    "CYBER_OPERATION", "ASSET_MOVEMENT", "DEFENSE_ACTIVATION", "STATUS_REPORT",
]
THREAT_LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
THREAT_EMOJI  = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴", "CRITICAL": "⚠️"}

FALLBACK_INTENT = {
    "translation": "",
    "primary_intent": "UNKNOWN",
    "secondary_intent": None,
    "confidence": 0.0,
    "threat_level": "LOW",
    "reasoning": "Analysis unavailable",
}


# ══════════════════════════════════════════════════════════════════════════════
# PRE-FLIGHT CHECKS
# ══════════════════════════════════════════════════════════════════════════════

def check_ollama():
    """Verify Ollama is running and qwen3:1.7b is available."""
    print("[preflight] Checking Ollama...")
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
    except Exception as e:
        print(f"  [FAIL] Ollama not reachable at {OLLAMA_URL}: {e}")
        print("         Start it with: ollama serve")
        sys.exit(1)

    matches = [m for m in models if OLLAMA_MODEL.split(":")[0] in m]
    if not matches:
        print(f"  [FAIL] Model '{OLLAMA_MODEL}' not found. Available: {models}")
        print(f"         Pull it with: ollama pull {OLLAMA_MODEL}")
        sys.exit(1)

    print(f"  [OK]   Ollama running  |  {OLLAMA_MODEL} available")


def warmup_model():
    """Load qwen3:1.7b into GPU and keep it there permanently."""
    print(f"[preflight] Warming up {OLLAMA_MODEL} on GPU (keep_alive=-1)...")
    try:
        r = requests.post(f"{OLLAMA_URL}/api/generate", json={
            "model": OLLAMA_MODEL,
            "prompt": "Ready.",
            "stream": False,
            "think": False,
            "keep_alive": -1,
            "options": {"num_predict": 5, "temperature": 0}
        }, timeout=30)
        r.raise_for_status()
        print(f"  [OK]   {OLLAMA_MODEL} loaded and pinned to GPU")
    except Exception as e:
        print(f"  [WARN] Warmup failed: {e} — will still attempt inference")


def check_gpu():
    """Report GPU state."""
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        used  = (torch.cuda.memory_allocated(0) + torch.cuda.memory_reserved(0)) / 1024**3
        free  = total - used
        print(f"[preflight] GPU: {total:.1f} GiB total  |  {free:.1f} GiB free")
        return free
    print("[preflight] GPU: not available — using CPU")
    return 0.0


def validate_wavs(paths):
    """Check all WAV files exist and are readable."""
    print(f"[preflight] Validating {len(paths)} WAV file(s)...")
    import soundfile as sf
    good = []
    for p in paths:
        if not os.path.exists(p):
            print(f"  [SKIP] Not found: {p}")
            continue
        try:
            info = sf.info(p)
            print(f"  [OK]   {os.path.basename(p)}  ({info.duration:.1f}s, {info.samplerate}Hz)")
            good.append(p)
        except Exception as e:
            print(f"  [SKIP] Unreadable: {p} — {e}")
    if not good:
        print("[FAIL] No valid WAV files.")
        sys.exit(1)
    return good


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def normalize_zh(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[：，。！？、；「」『』【】《》〈〉\s]+", "", text)
    return text.strip()

def is_chinese(text, threshold=0.35):
    if not text: return True
    cjk = sum(1 for c in text if '一' <= c <= '鿿')
    return (cjk / len(text)) > threshold

def cer(ref, hyp):
    r, h = list(ref.replace(" ","")), list(hyp.replace(" ",""))
    dp = list(range(len(h)+1))
    for i in range(1, len(r)+1):
        prev, dp[0] = dp[0], i
        for j in range(1, len(h)+1):
            tmp = dp[j]
            dp[j] = prev if r[i-1]==h[j-1] else 1+min(prev,dp[j],dp[j-1])
            prev = tmp
    return round(dp[len(h)] / max(len(r), 1), 4)

def detect_code_words(zh_text, en_text):
    zh_hits = [cw for cw in CODE_WORD_MAP if cw in zh_text]
    en_hits = [cw for cw, en in CODE_WORD_MAP.items() if en.lower() in en_text.lower()]
    return list(dict.fromkeys(zh_hits + en_hits))


# ══════════════════════════════════════════════════════════════════════════════
# OLLAMA — retry-safe call
# ══════════════════════════════════════════════════════════════════════════════

def ollama_call(prompt, num_predict=400):
    """Call Ollama with retries. Returns empty string on all failures."""
    for attempt in range(1, OLLAMA_RETRIES + 1):
        try:
            resp = requests.post(f"{OLLAMA_URL}/api/generate", json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "think": False,
                "keep_alive": -1,
                "options": {"temperature": 0.1, "num_predict": num_predict}
            }, timeout=OLLAMA_TIMEOUT)
            resp.raise_for_status()
            raw = resp.json().get("response", "").strip()
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            if raw:
                return raw
            if attempt < OLLAMA_RETRIES:
                time.sleep(1)
        except Exception as e:
            if attempt < OLLAMA_RETRIES:
                time.sleep(2 ** attempt)
            else:
                print(f"    [warn] Ollama call failed after {OLLAMA_RETRIES} attempts: {e}")
    return ""


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: COLLECT WAVs
# ══════════════════════════════════════════════════════════════════════════════

def collect_wavs(args):
    if args.files:
        paths = args.files
    elif args.dir:
        paths = sorted([
            os.path.join(args.dir, f)
            for f in os.listdir(args.dir) if f.endswith(".wav")
        ])
    else:
        paths = TOP5_DEFAULT
    return paths


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: ASR — whisper-large-v3-turbo
# ══════════════════════════════════════════════════════════════════════════════

def transcribe(wav_paths, gpu_free_gb):
    # Use GPU only if >2 GiB free after Ollama model is loaded
    use_gpu = torch.cuda.is_available() and gpu_free_gb > 2.0
    device  = 0 if use_gpu else -1
    dtype   = torch.float16 if use_gpu else torch.float32
    dev_str = f"cuda (free={gpu_free_gb:.1f}GB)" if use_gpu else "cpu"

    print(f"\n[Step 2] ASR  →  whisper-large-v3-turbo  ({dev_str})")
    from transformers import pipeline as hf_pipeline

    asr = hf_pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        generate_kwargs={"language": "chinese", "task": "transcribe"},
        device=device,
        torch_dtype=dtype,
    )

    transcripts = []
    t0 = time.time()
    for i, wav in enumerate(tqdm(wav_paths, desc="  transcribing"), 1):
        try:
            result = asr(wav)
            text   = result["text"].strip()
            transcripts.append(text)
        except Exception as e:
            print(f"  [warn] ASR failed on {wav}: {e}")
            transcripts.append("")

    elapsed = time.time() - t0
    print(f"  Done: {elapsed:.1f}s total  |  {elapsed/len(wav_paths):.1f}s per file")
    del asr; gc.collect()
    if use_gpu: torch.cuda.empty_cache()
    return transcripts


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3+4: TRANSLATE + CLASSIFY (one Ollama call per file, run in parallel)
# ══════════════════════════════════════════════════════════════════════════════

COMBINED_PROMPT = """You are a military signals intelligence analyst.

Given a Mandarin Chinese intercepted communication (may be partially garbled by speech recognition), perform THREE tasks:

TASK 1 — TRANSLATE to fluent English. If characters are garbled, infer the likely military meaning from context.
TASK 2 — CLASSIFY primary intent (choose exactly one):
  STRIKE_AUTHORIZATION  = order to fire, launch, or engage
  RECON_REPORT          = sensor / radar / satellite data relay
  INFILTRATION_CONFIRM  = agent or unit breached enemy position
  CYBER_OPERATION       = network intrusion or system disruption
  ASSET_MOVEMENT        = repositioning units or extraction
  DEFENSE_ACTIVATION    = defensive system triggered or intercept
  STATUS_REPORT         = readiness or mission completion update
TASK 3 — THREAT level: LOW / MEDIUM / HIGH / CRITICAL

MANDARIN INPUT:
{text}

OUTPUT valid JSON only:
{{
  "translation": "<fluent English>",
  "primary_intent": "<intent>",
  "secondary_intent": "<intent or null>",
  "confidence": <0.0-1.0>,
  "threat_level": "<LOW|MEDIUM|HIGH|CRITICAL>",
  "reasoning": "<one concise sentence>"
}}"""

def parse_intent_json(raw):
    """Extract and validate JSON from Ollama response."""
    if not raw:
        return None
    try:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        data = json.loads(m.group() if m else raw)

        # Validate and sanitize fields
        intent = data.get("primary_intent", "UNKNOWN").upper()
        if intent not in INTENT_CLASSES:
            intent = "UNKNOWN"

        threat = data.get("threat_level", "LOW").upper()
        if threat not in THREAT_LEVELS:
            threat = "LOW"

        conf = float(data.get("confidence", 0.0))
        conf = max(0.0, min(1.0, conf))

        sec = data.get("secondary_intent")
        if sec and sec.upper() not in INTENT_CLASSES:
            sec = None

        return {
            "translation":     str(data.get("translation", "")).strip(),
            "primary_intent":  intent,
            "secondary_intent":sec,
            "confidence":      round(conf, 2),
            "threat_level":    threat,
            "reasoning":       str(data.get("reasoning", "")).strip()[:200],
        }
    except Exception:
        return None

def analyze_one(idx, text):
    """Translate + classify one sample. Returns (idx, result)."""
    if not text.strip():
        r = FALLBACK_INTENT.copy()
        r["reasoning"] = "Empty transcript"
        return idx, r

    raw    = ollama_call(COMBINED_PROMPT.format(text=text))
    result = parse_intent_json(raw)

    if result is None or is_chinese(result.get("translation", "")):
        # JSON parse failed or translation is still Chinese — retry once with simpler prompt
        fallback_prompt = (
            f"Translate this Mandarin military text to English, then on a new line "
            f"write the intent as one of: {', '.join(INTENT_CLASSES)}.\n\n{text}"
        )
        raw2 = ollama_call(fallback_prompt, num_predict=200)
        lines = [l.strip() for l in raw2.split("\n") if l.strip()]
        translation = lines[0] if lines else ""
        intent_line = next((l for l in lines[1:] if any(c in l.upper() for c in INTENT_CLASSES)), "")
        detected_intent = next((c for c in INTENT_CLASSES if c in intent_line.upper()), "UNKNOWN")
        result = {
            "translation":     translation,
            "primary_intent":  detected_intent,
            "secondary_intent":None,
            "confidence":      0.5,
            "threat_level":    "MEDIUM" if detected_intent != "UNKNOWN" else "LOW",
            "reasoning":       "Recovered via fallback prompt",
        }

    return idx, result

def translate_and_classify(norm_texts):
    print(f"\n[Step 3+4] Translate + Classify  →  {OLLAMA_MODEL}  (parallel, 3 workers)")
    results = [None] * len(norm_texts)
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(analyze_one, i, t): i for i, t in enumerate(norm_texts)}
        for future in tqdm(as_completed(futures), total=len(futures), desc="  analyzing"):
            idx, result = future.result()
            results[idx] = result
            print(f"  [{idx+1:02d}] {result['primary_intent']:<26} "
                  f"conf={result['confidence']*100:.0f}%  "
                  f"threat={result['threat_level']:<8}  "
                  f"→ {result['translation'][:65]}")

    elapsed = time.time() - t0
    print(f"  Done: {elapsed:.1f}s total  |  {elapsed/len(norm_texts):.1f}s per file")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: LaBSE SCORING
# ══════════════════════════════════════════════════════════════════════════════

def score_meaning(zh_texts, en_texts):
    valid = [(i, z, e) for i, (z, e) in enumerate(zip(zh_texts, en_texts))
             if z.strip() and e.strip() and not is_chinese(e)]
    if not valid:
        print("\n[Step 5] LaBSE: no valid EN translations to score")
        return [None] * len(zh_texts)

    print(f"\n[Step 5] LaBSE scoring  ({len(valid)}/{len(zh_texts)} valid pairs)...")
    from sentence_transformers import SentenceTransformer, util
    labse  = SentenceTransformer("sentence-transformers/LaBSE")
    idxs   = [v[0] for v in valid]
    zh_emb = labse.encode([v[1] for v in valid], convert_to_tensor=True, normalize_embeddings=True)
    en_emb = labse.encode([v[2] for v in valid], convert_to_tensor=True, normalize_embeddings=True)
    sims   = util.cos_sim(zh_emb, en_emb).diagonal().cpu().tolist()
    del labse; gc.collect()

    out = [None] * len(zh_texts)
    for i, sim in zip(idxs, sims):
        out[i] = round(sim, 4)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: WRITE REPORT
# ══════════════════════════════════════════════════════════════════════════════

def write_report(records, out_path, elapsed_total):
    sep  = "=" * 72
    dash = "-" * 72
    lines = []

    lines.append(sep)
    lines.append("  MANDARIN SPEECH INTELLIGENCE REPORT")
    lines.append(f"  Generated   : {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    lines.append(f"  ASR         : openai/whisper-large-v3-turbo")
    lines.append(f"  Analysis    : {OLLAMA_MODEL} via Ollama (translate + classify)")
    lines.append(f"  Files       : {len(records)}")
    lines.append(f"  Total time  : {elapsed_total:.1f}s  ({elapsed_total/max(len(records),1):.1f}s/file)")
    lines.append(sep)

    for r in records:
        lines.append(f"\nSAMPLE {r['sample_id']:02d}  [{os.path.basename(r['wav'])}]")
        lines.append(dash)
        lines.append(f"TRANSCRIPT (ZH) : {r['asr_raw'] or '[transcription failed]'}")
        lines.append(f"TRANSLATION (EN): {r['translation_en'] or '[translation failed]'}")
        if r['similarity'] is not None:
            lines.append(f"MEANING SCORE   : {r['similarity']*100:.1f}%  (loss: {(1-r['similarity'])*100:.1f}%)")
        else:
            lines.append(f"MEANING SCORE   : N/A")
        lines.append("")

        intent = r['intent']
        tl     = intent.get('threat_level', 'LOW')
        lines.append(f"PRIMARY INTENT  : {intent.get('primary_intent', 'UNKNOWN')}")
        sec = intent.get('secondary_intent')
        if sec:
            lines.append(f"SECONDARY INTENT: {sec}")
        lines.append(f"CONFIDENCE      : {intent.get('confidence', 0)*100:.0f}%")
        lines.append(f"THREAT LEVEL    : {tl}  {THREAT_EMOJI.get(tl, '')}")
        lines.append(f"REASONING       : {intent.get('reasoning', '')}")
        lines.append("")

        cw    = r['code_words']
        en_cw = [CODE_WORD_MAP[c] for c in cw if c in CODE_WORD_MAP]
        lines.append(f"CODE WORDS      : {', '.join(en_cw) if en_cw else 'none detected'}")

    # ── Summary table ──────────────────────────────────────────────────────────
    lines.append(f"\n\n{sep}")
    lines.append("  SUMMARY")
    lines.append(sep)
    lines.append(f"{'#':<4} {'File':<26} {'Meaning':>8} {'Intent':<26} {'Threat':<10}")
    lines.append("-" * 72)
    for r in records:
        sim_str = f"{r['similarity']*100:.1f}%" if r['similarity'] is not None else "  N/A"
        lines.append(
            f"{r['sample_id']:<4} {os.path.basename(r['wav']):<26} "
            f"{sim_str:>8} {r['intent'].get('primary_intent','UNKNOWN'):<26} "
            f"{r['intent'].get('threat_level','LOW'):<10}"
        )

    valid_sim = [r['similarity'] for r in records if r['similarity'] is not None]
    if valid_sim:
        lines.append(f"\n  Avg meaning preserved : {sum(valid_sim)/len(valid_sim)*100:.1f}%")

    intent_counts = {}
    for r in records:
        k = r['intent'].get('primary_intent', 'UNKNOWN')
        intent_counts[k] = intent_counts.get(k, 0) + 1
    lines.append(f"\n  Intent breakdown:")
    for k, v in sorted(intent_counts.items(), key=lambda x: -x[1]):
        lines.append(f"    {k:<30} {v} sample(s)")

    threat_counts = {}
    for r in records:
        k = r['intent'].get('threat_level', 'LOW')
        threat_counts[k] = threat_counts.get(k, 0) + 1
    lines.append(f"\n  Threat breakdown:")
    for k in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        if k in threat_counts:
            lines.append(f"    {k:<12} {threat_counts[k]} sample(s)  {THREAT_EMOJI.get(k,'')}")

    lines.append(f"\n  Pipeline latency: {elapsed_total:.1f}s total  |  {elapsed_total/max(len(records),1):.1f}s per file")
    lines.append(f"\n{sep}")
    lines.append(f"  Report saved → {out_path}")
    lines.append(sep)

    text = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Mandarin Speech Intelligence Pipeline")
    parser.add_argument("files", nargs="*", help="WAV file(s) to process")
    parser.add_argument("--dir", default=None, help="Directory of WAV files")
    parser.add_argument("--out", default=None, help="Output .txt path")
    args = parser.parse_args()

    print("=" * 72)
    print("  INTEL PIPELINE  —  PRE-FLIGHT")
    print("=" * 72)

    # Pre-flight
    check_ollama()
    warmup_model()
    gpu_free = check_gpu()

    raw_paths = collect_wavs(args)
    wav_paths = validate_wavs(raw_paths)

    out_path = args.out or f"intel_fast_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    print("\n" + "=" * 72)
    print("  PIPELINE START")
    print("=" * 72)
    t_start = time.time()

    # ASR
    transcripts = transcribe(wav_paths, gpu_free)
    normalized  = [normalize_zh(t) for t in transcripts]

    # Translate + Classify
    analyses     = translate_and_classify(normalized)
    translations = [a.get("translation", "") for a in analyses]

    # LaBSE
    sims = score_meaning(transcripts, translations)

    elapsed = time.time() - t_start

    # Assemble records
    records = []
    for i, wav in enumerate(wav_paths):
        records.append({
            "sample_id":      i + 1,
            "wav":            wav,
            "asr_raw":        transcripts[i],
            "translation_en": translations[i],
            "similarity":     sims[i],
            "intent":         analyses[i],
            "code_words":     detect_code_words(normalized[i], translations[i]),
        })

    # Save JSON
    json_path = out_path.replace(".txt", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    # Write and print report
    report = write_report(records, out_path, elapsed)
    print("\n" + report)


if __name__ == "__main__":
    main()
