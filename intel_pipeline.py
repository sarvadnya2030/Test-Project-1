#!/usr/bin/env python3
"""
Mandarin Speech Intelligence Pipeline — v3
Usage:
    python3 intel_pipeline.py                        # processes all wavs in espionage_samples/
    python3 intel_pipeline.py file1.wav file2.wav    # specific files
    python3 intel_pipeline.py --dir /path/to/wavs    # custom directory

Output: intel_report_<timestamp>.txt
"""

import os, sys, gc, re, json, warnings, argparse, unicodedata
from datetime import datetime
warnings.filterwarnings("ignore")

import torch
import requests
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ── Device config ──────────────────────────────────────────────────────────────
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
MT_DEVICE = "cpu"
MT_DTYPE  = torch.float32

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:1.7b"

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
    "STRIKE_AUTHORIZATION",   # order to fire, launch, or engage a target
    "RECON_REPORT",           # sensor, radar, or satellite data relay
    "INFILTRATION_CONFIRM",   # agent or unit has breached enemy position
    "CYBER_OPERATION",        # network intrusion or system disruption
    "ASSET_MOVEMENT",         # repositioning units, extraction, approach
    "DEFENSE_ACTIVATION",     # defensive system triggered, intercept success
    "STATUS_REPORT",          # readiness or mission completion update
]

# ── Helpers ────────────────────────────────────────────────────────────────────
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
    return dp[len(h)] / max(len(r), 1)


# ── STEP 1: Collect WAV files ──────────────────────────────────────────────────
def collect_wavs(args):
    if args.files:
        paths = [f for f in args.files if os.path.exists(f)]
    elif args.dir:
        paths = sorted([
            os.path.join(args.dir, f)
            for f in os.listdir(args.dir) if f.endswith(".wav")
        ])
    else:
        default = "espionage_samples"
        paths = sorted([
            os.path.join(default, f)
            for f in os.listdir(default) if f.endswith(".wav")
        ]) if os.path.isdir(default) else []

    if not paths:
        print("[ERROR] No WAV files found.")
        sys.exit(1)
    print(f"[Step 1] Found {len(paths)} WAV file(s)")
    for p in paths: print(f"         {p}")
    return paths


# ── STEP 2: ASR — BELLE Whisper large-v3-zh ───────────────────────────────────
def transcribe(wav_paths):
    print(f"\n[Step 2] ASR  →  BELLE-2/Belle-whisper-large-v3-zh  (device={DEVICE})")
    from transformers import pipeline as hf_pipeline
    # Force CPU — Ollama occupies entire GPU (two models loaded = ~7.3 GiB / 7.6 GiB)
    asr = hf_pipeline(
        "automatic-speech-recognition",
        model="BELLE-2/Belle-whisper-large-v3-zh",
        generate_kwargs={"language": "chinese", "task": "transcribe"},
        device=-1,
        torch_dtype=torch.float32,
    )
    out = []
    for wav in tqdm(wav_paths, desc="  transcribing"):
        try:
            result = asr(wav)
            out.append(result["text"].strip())
        except Exception as e:
            print(f"  [warn] ASR failed on {wav}: {e}")
            out.append("")
    del asr; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return out


# ── STEP 3: Translation ────────────────────────────────────────────────────────
# Primary: Ollama qwen3:1.7b  (handles garbled ASR well, no torch.load issues)
# Fallback: NLLB-1.3B via safetensors  (if Ollama unavailable)

TRANSLATE_PROMPT = (
    "Translate the following Mandarin Chinese military communication to English. "
    "Output ONLY the English translation, nothing else. "
    "If characters are garbled or unclear, infer the most likely military meaning.\n\n"
    "Mandarin: {text}\n\nEnglish:\n/no_think"
)

def ollama_call(prompt, num_predict=300):
    resp = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {"temperature": 0.1, "num_predict": num_predict}
    }, timeout=45)
    raw = resp.json()["response"].strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    return raw

def translate_ollama(text):
    if not text.strip():
        return ""
    try:
        return ollama_call(TRANSLATE_PROMPT.format(text=text))
    except Exception:
        return ""

def translate_nllb(texts):
    model_id  = "facebook/nllb-200-1.3B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model     = AutoModelForSeq2SeqLM.from_pretrained(
        model_id, dtype=MT_DTYPE, use_safetensors=True
    ).to(MT_DEVICE)
    model.eval()
    tokenizer.src_lang = "zho_Hans"
    forced_bos = tokenizer.convert_tokens_to_ids("eng_Latn")
    out = []
    for text in texts:
        if not text.strip(): out.append(""); continue
        inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inp = {k: v.to(MT_DEVICE) for k, v in inp.items()}
        with torch.no_grad():
            ids = model.generate(**inp, forced_bos_token_id=forced_bos, max_new_tokens=256)
        out.append(tokenizer.decode(ids[0], skip_special_tokens=True))
    del model, tokenizer; gc.collect()
    return out

def translate(texts):
    print(f"\n[Step 3] Translation  →  {OLLAMA_MODEL} (Ollama)")
    translations = []
    for i, text in enumerate(tqdm(texts, desc="  translating"), 1):
        en = translate_ollama(text)
        translations.append(en)
        print(f"  [{i:02d}] {en[:80]}")

    # fallback to NLLB for any that came back empty
    failed_idx = [i for i, t in enumerate(translations) if not t.strip()]
    if failed_idx:
        print(f"  [fallback] {len(failed_idx)} empty — retrying with NLLB-1.3B")
        fb = translate_nllb([texts[i] for i in failed_idx])
        for idx, i in enumerate(failed_idx):
            translations[i] = fb[idx]

    return translations


# ── STEP 4: Code-word detection ────────────────────────────────────────────────
def detect_code_words(zh_text, en_text):
    zh_hits = [cw for cw in CODE_WORD_MAP if cw in zh_text]
    en_hits = [en for en in CODE_WORD_MAP.values() if en.lower() in en_text.lower()]
    combined = list(dict.fromkeys(zh_hits + [k for k,v in CODE_WORD_MAP.items() if v in en_hits]))
    return combined


# ── STEP 5: Intent classification via Ollama qwen3:1.7b ───────────────────────
INTENT_PROMPT = """You are a military signals intelligence analyst.
Classify the primary intent of this intercepted and translated Mandarin military communication.

INTENT OPTIONS:
  STRIKE_AUTHORIZATION  - order to fire, launch missile, or engage a target
  RECON_REPORT          - relaying sensor, radar, or satellite intelligence
  INFILTRATION_CONFIRM  - agent or unit has breached enemy territory/network
  CYBER_OPERATION       - network intrusion, system disruption, or cyberattack
  ASSET_MOVEMENT        - repositioning units, extraction, or approach orders
  DEFENSE_ACTIVATION    - defensive system triggered, intercept, or jamming
  STATUS_REPORT         - readiness confirmation or mission completion update

COMMUNICATION:
\"{text}\"

Respond ONLY with valid JSON. No thinking tags. No text outside JSON:
{{
  "primary_intent": "<ONE OF THE INTENT OPTIONS>",
  "secondary_intent": "<ONE OF THE INTENT OPTIONS or null>",
  "confidence": <0.0 to 1.0>,
  "threat_level": "<LOW|MEDIUM|HIGH|CRITICAL>",
  "reasoning": "<one sentence>"
}}
/no_think"""

def classify_intent(translation):
    if not translation.strip():
        return {"primary_intent": "UNKNOWN", "secondary_intent": None,
                "confidence": 0.0, "threat_level": "LOW", "reasoning": "Empty translation"}
    try:
        raw = ollama_call(INTENT_PROMPT.format(text=translation), num_predict=250)
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            return json.loads(m.group())
        return json.loads(raw)
    except Exception as e:
        return {"primary_intent": "UNKNOWN", "secondary_intent": None,
                "confidence": 0.0, "threat_level": "LOW", "reasoning": f"Classification error: {e}"}


# ── STEP 6: LaBSE meaning-preservation (optional — skipped if no gold source) ─
def score_meaning(zh_texts, en_texts):
    print("\n[Step 6] LaBSE meaning-preservation scoring...")
    from sentence_transformers import SentenceTransformer, util
    labse  = SentenceTransformer("sentence-transformers/LaBSE")
    zh_emb = labse.encode(zh_texts, convert_to_tensor=True, normalize_embeddings=True)
    en_emb = labse.encode(en_texts, convert_to_tensor=True, normalize_embeddings=True)
    sims   = util.cos_sim(zh_emb, en_emb).diagonal().cpu().tolist()
    del labse; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return [round(s, 4) for s in sims]


# ── STEP 7: Write report ───────────────────────────────────────────────────────
THREAT_EMOJI = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴", "CRITICAL": "⚠️"}

def write_report(records, out_path):
    sep  = "=" * 72
    dash = "-" * 72

    lines = []
    lines.append(sep)
    lines.append("  MANDARIN SPEECH INTELLIGENCE REPORT")
    lines.append(f"  Generated : {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    lines.append(f"  ASR Model : BELLE-2/Belle-whisper-large-v3-zh")
    lines.append(f"  MT  Model : facebook/nllb-200-1.3B  (opus-mt fallback)")
    lines.append(f"  Intent    : {OLLAMA_MODEL} via Ollama")
    lines.append(f"  Samples   : {len(records)}")
    lines.append(sep)

    for r in records:
        lines.append(f"\nSAMPLE {r['sample_id']:02d}  [{os.path.basename(r['wav'])}]")
        lines.append(dash)
        lines.append(f"TRANSCRIPT (ZH) : {r['asr_raw']}")
        lines.append(f"TRANSLATION (EN): {r['translation_en']}")
        lines.append(f"CER             : {r['cer']*100:.1f}%")
        if r['similarity'] is not None:
            lines.append(f"MEANING SCORE   : {r['similarity']*100:.1f}%  (loss: {(1-r['similarity'])*100:.1f}%)")
        lines.append("")

        intent = r['intent']
        tl     = intent.get('threat_level', 'LOW')
        emoji  = THREAT_EMOJI.get(tl, '')
        lines.append(f"PRIMARY INTENT  : {intent.get('primary_intent', 'UNKNOWN')}")
        if intent.get('secondary_intent'):
            lines.append(f"SECONDARY INTENT: {intent['secondary_intent']}")
        lines.append(f"CONFIDENCE      : {intent.get('confidence', 0)*100:.0f}%")
        lines.append(f"THREAT LEVEL    : {tl}  {emoji}")
        lines.append(f"REASONING       : {intent.get('reasoning', '')}")
        lines.append("")

        detected = r['code_words']
        if detected:
            cw_en = [CODE_WORD_MAP[c] for c in detected if c in CODE_WORD_MAP]
            lines.append(f"CODE WORDS      : {', '.join(cw_en)}  ({len(detected)} detected)")
        else:
            lines.append(f"CODE WORDS      : none detected")

    # ── Summary table ──────────────────────────────────────────────────────────
    lines.append(f"\n\n{sep}")
    lines.append("  SUMMARY")
    lines.append(sep)
    lines.append(f"{'#':<4} {'File':<24} {'CER':>6} {'Meaning':>8} {'Intent':<26} {'Threat':<10}")
    lines.append("-" * 72)
    for r in records:
        sim_str = f"{r['similarity']*100:.1f}%" if r['similarity'] else "  N/A"
        lines.append(
            f"{r['sample_id']:<4} "
            f"{os.path.basename(r['wav']):<24} "
            f"{r['cer']*100:>5.1f}% "
            f"{sim_str:>8} "
            f"{r['intent'].get('primary_intent','UNKNOWN'):<26} "
            f"{r['intent'].get('threat_level','LOW'):<10}"
        )

    valid_sim = [r['similarity'] for r in records if r['similarity'] is not None]
    avg_cer   = sum(r['cer'] for r in records) / len(records)
    avg_sim   = sum(valid_sim) / len(valid_sim) if valid_sim else 0

    lines.append("-" * 72)
    lines.append(f"{'AVG':<4} {'':<24} {avg_cer*100:>5.1f}% {avg_sim*100:>7.1f}%")

    intent_counts = {}
    for r in records:
        k = r['intent'].get('primary_intent', 'UNKNOWN')
        intent_counts[k] = intent_counts.get(k, 0) + 1
    lines.append(f"\n  Intent breakdown:")
    for k, v in sorted(intent_counts.items(), key=lambda x: -x[1]):
        lines.append(f"    {k:<28} {v} sample(s)")

    threat_counts = {}
    for r in records:
        k = r['intent'].get('threat_level', 'LOW')
        threat_counts[k] = threat_counts.get(k, 0) + 1
    lines.append(f"\n  Threat level breakdown:")
    for k in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        if k in threat_counts:
            lines.append(f"    {k:<12} {threat_counts[k]} sample(s)  {THREAT_EMOJI.get(k,'')}")

    lines.append(f"\n{sep}")
    lines.append(f"  Report saved → {out_path}")
    lines.append(sep)

    text = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Mandarin Speech Intelligence Pipeline")
    parser.add_argument("files",  nargs="*",       help="WAV file(s) to process")
    parser.add_argument("--dir",  default=None,    help="Directory of WAV files")
    parser.add_argument("--out",  default=None,    help="Output txt path (auto-named if omitted)")
    args = parser.parse_args()

    wav_paths = collect_wavs(args)
    out_path  = args.out or f"intel_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    # ASR
    transcripts = transcribe(wav_paths)
    normalized  = [normalize_zh(t) for t in transcripts]

    # Translation
    translations = translate(normalized)

    # LaBSE scoring
    sims = score_meaning(transcripts, translations)

    # Intent classification
    print(f"\n[Step 5] Intent classification  →  {OLLAMA_MODEL}")
    intents = []
    for i, en in enumerate(tqdm(translations, desc="  classifying"), 1):
        intent = classify_intent(en)
        intents.append(intent)
        print(f"  [{i:02d}] {intent.get('primary_intent','?'):<28} "
              f"conf={intent.get('confidence',0)*100:.0f}%  "
              f"threat={intent.get('threat_level','?')}")

    # Assemble records
    records = []
    for i, wav in enumerate(wav_paths):
        records.append({
            "sample_id":     i + 1,
            "wav":           wav,
            "asr_raw":       transcripts[i],
            "asr_normalized":normalized[i],
            "translation_en":translations[i],
            "cer":           round(cer(transcripts[i], normalized[i]), 4),
            "similarity":    sims[i],
            "intent":        intents[i],
            "code_words":    detect_code_words(normalized[i], translations[i]),
        })

    # Save JSON alongside txt
    json_path = out_path.replace(".txt", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    # Write and print report
    report_text = write_report(records, out_path)
    print("\n" + report_text)


if __name__ == "__main__":
    main()
