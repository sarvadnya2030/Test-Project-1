#!/usr/bin/env python3
"""
Mandarin Military/Espionage ASR Pipeline — v2
Improvements over v1:
  - ASR  : BELLE Whisper large-v3-zh  (lower CER than standard Whisper)
  - MT   : NLLB-200-1.3B on CPU       (+7.8% meaning preservation vs 600M)
  - Score: LaBSE inline               (meaning preservation per sample)
  - Detection: code-words checked in both ZH transcript and EN translation
"""

import os, gc, json, warnings
warnings.filterwarnings("ignore")

import torch
import pandas as pd
import soundfile as sf
from tqdm import tqdm

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MT_DEVICE   = "cpu"          # Ollama holds ~5.5 GiB of RTX 2070 — MT stays on CPU
MT_DTYPE    = torch.float32

OUT_DIR  = "espionage_samples"
CSV_PATH = f"{OUT_DIR}/espionage_pipeline_v2_results.csv"
JSON_PATH= f"{OUT_DIR}/espionage_pipeline_v2_report.json"
os.makedirs(OUT_DIR, exist_ok=True)

print(f"[config] ASR device={DEVICE}  MT device={MT_DEVICE}\n")

# ── Dialogues ──────────────────────────────────────────────────────────────────
DIALOGUES = [
    "指挥官：狐狸编队准备就绪，确认目标坐标三五点六北纬。专家：收到，狐狸一号至十号全员待命，夜莺卫星图像已传输。",
    "专家：鲨鱼信号确认，方位一二零度，距离四十五海里。指挥官：启动黑剑拦截程序，苍鹰中队立即升空。",
    "指挥官：幽灵小队已通过铁拳火力掩护，成功渗透敌后。专家：银狐线人确认敌指挥部位置，坐标已发送。",
    "专家：毒蛇已成功渗透目标网络，获取指挥系统权限。指挥官：启动暗流程序，瘫痪敌方系统。",
    "指挥官：红旗雷达侦测到黑剑来袭，铁网系统启动。专家：铁网拦截成功，黑剑残骸落入预定区域。",
    "专家：银狐准备就绪，请求狐狸空中接应。指挥官：狐狸三号正在接近，预计三分钟到达。",
    "指挥官：铁拳锁定目标，坐标二三到四五，反复三次确认。专家：铁拳就绪，火力覆盖范围已设定。",
    "专家：苍鹰中队报告，敌机已被击落两架。指挥官：维持苍鹰编队高度，防止雷霆反击。",
    "指挥官：迷雾电子战系统启动，压制红旗雷达信号。专家：红旗信号已消失，敌方防空盲区形成。",
    "专家：指挥部遭遇暗流攻击，网络已完全瘫痪。指挥官：启动备用通信，银狐立即转移至B点。",
]

CODE_WORD_MAP = {
    "狐狸": "Fox",          "鲨鱼": "Shark",
    "苍鹰": "BlueEagle",    "黑剑": "BlackSword",
    "幽灵": "Ghost",        "铁拳": "IronFist",
    "夜莺": "Nightingale",  "毒蛇": "Viper",
    "银狐": "SilverFox",    "红旗": "RedFlag",
    "钢龙": "SteelDragon",  "暗流": "DarkCurrent",
    "铁网": "IronNet",      "雷霆": "Thunder",
    "迷雾": "Fog",
}
EN_CODE_WORDS = set(CODE_WORD_MAP.values())


# ── STEP 1: TTS — reuse existing wavs if present ──────────────────────────────
def get_or_generate_audio():
    existing = [f"{OUT_DIR}/espionage_{i:02d}.wav" for i in range(1, 11)]
    if all(os.path.exists(p) for p in existing):
        print("[Step 1] WAV files already exist — skipping TTS generation.")
        return existing

    print("[Step 1] Generating audio with MeloTTS...")
    from melo.api import TTS
    speeds = [0.85, 0.90, 0.95, 1.00, 1.05, 0.88, 0.93, 0.98, 1.02, 0.87]
    tts = TTS(language="ZH", device="auto")
    speaker_ids = tts.hps.data.spk2id
    spk = speaker_ids.get("ZH", list(speaker_ids.values())[0])
    paths = []
    for i, (text, spd) in enumerate(zip(DIALOGUES, speeds), 1):
        out = f"{OUT_DIR}/espionage_{i:02d}.wav"
        tts.tts_to_file(text, spk, out, speed=spd)
        paths.append(out)
        print(f"  [{i:02d}] {out}")
    del tts; gc.collect()
    return paths


# ── STEP 2: ASR — BELLE Whisper large-v3-zh (better Chinese CER than vanilla) ─
def transcribe(wav_paths):
    print("\n[Step 2] ASR: BELLE-2/Belle-whisper-large-v3-zh")
    from transformers import pipeline as hf_pipeline

    asr = hf_pipeline(
        "automatic-speech-recognition",
        model="BELLE-2/Belle-whisper-large-v3-zh",
        generate_kwargs={"language": "chinese", "task": "transcribe"},
        device=0 if DEVICE == "cuda" else -1,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    )

    transcripts = []
    for i, wav in enumerate(tqdm(wav_paths, desc="  transcribing"), 1):
        result = asr(wav)
        text   = result["text"].strip()
        transcripts.append(text)
        print(f"  [{i:02d}] {text[:80]}")

    del asr; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return transcripts


# ── STEP 3: Normalize transcript ──────────────────────────────────────────────
def normalize(text: str) -> str:
    import unicodedata, re
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[：，。！？、；「」『』【】《》〈〉\s]+", "", text)
    return text


# ── STEP 4: Translation — NLLB-1.3B on CPU (proven best in comparison study) ──
def translate_batch(texts):
    print("\n[Step 4] Translation: facebook/nllb-200-1.3B  (CPU)")
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    model_id  = "facebook/nllb-200-1.3B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model     = AutoModelForSeq2SeqLM.from_pretrained(model_id, dtype=MT_DTYPE, use_safetensors=True).to(MT_DEVICE)
    model.eval()

    tokenizer.src_lang = "zho_Hans"
    forced_bos = tokenizer.convert_tokens_to_ids("eng_Latn")

    translations = []
    for i, text in enumerate(tqdm(texts, desc="  translating"), 1):
        if not text.strip():
            translations.append("")
            continue
        inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inp = {k: v.to(MT_DEVICE) for k, v in inp.items()}
        with torch.no_grad():
            ids = model.generate(**inp, forced_bos_token_id=forced_bos, max_new_tokens=256)
        en = tokenizer.decode(ids[0], skip_special_tokens=True)
        translations.append(en)
        print(f"  [{i:02d}] {en[:80]}")

    del model, tokenizer; gc.collect()
    return translations


# ── STEP 5: Code-word detection (ZH transcript + EN translation) ──────────────
def detect_code_words(asr_text, en_text):
    zh_hits = [cw for cw in CODE_WORD_MAP if cw in asr_text]
    en_hits = [en for en in EN_CODE_WORDS if en.lower() in en_text.lower()]
    return zh_hits, en_hits


# ── STEP 6: CER ───────────────────────────────────────────────────────────────
def cer(ref: str, hyp: str) -> float:
    ref_c, hyp_c = list(ref.replace(" ", "")), list(hyp.replace(" ", ""))
    r, h = len(ref_c), len(hyp_c)
    dp = list(range(h + 1))
    for i in range(1, r + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, h + 1):
            tmp = dp[j]
            dp[j] = prev if ref_c[i-1] == hyp_c[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = tmp
    return dp[h] / max(r, 1)


# ── STEP 7: LaBSE meaning-preservation score ──────────────────────────────────
def score_meaning(sources_zh, translations_en):
    print("\n[Step 7] LaBSE meaning-preservation scoring...")
    from sentence_transformers import SentenceTransformer, util

    labse   = SentenceTransformer("sentence-transformers/LaBSE")
    zh_emb  = labse.encode(sources_zh,       convert_to_tensor=True, normalize_embeddings=True)
    en_emb  = labse.encode(translations_en,  convert_to_tensor=True, normalize_embeddings=True)
    sims    = util.cos_sim(zh_emb, en_emb).diagonal().cpu().numpy()
    del labse; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return sims


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    wav_paths    = get_or_generate_audio()

    # ASR
    raw_transcripts  = transcribe(wav_paths)
    norm_transcripts = [normalize(t) for t in raw_transcripts]

    # Translation
    translations = translate_batch(norm_transcripts)

    # CER per sample
    cer_scores = [
        round(cer(normalize(DIALOGUES[i]), norm_transcripts[i]), 4)
        for i in range(len(DIALOGUES))
    ]

    # Code-word detection
    cw_results = [
        detect_code_words(norm_transcripts[i], translations[i])
        for i in range(len(DIALOGUES))
    ]

    # LaBSE
    sims = score_meaning(DIALOGUES, translations)

    # ── Assemble records ───────────────────────────────────────────────────────
    records = []
    for i in range(len(DIALOGUES)):
        zh_hits, en_hits = cw_results[i]
        expected = [cw for cw in CODE_WORD_MAP if cw in DIALOGUES[i]]
        recall   = round(len(zh_hits) / max(len(expected), 1), 4)
        records.append({
            "sample_id":            i + 1,
            "source_zh":            DIALOGUES[i],
            "asr_raw":              raw_transcripts[i],
            "asr_normalized":       norm_transcripts[i],
            "translation_en":       translations[i],
            "cer":                  cer_scores[i],
            "meaning_similarity":   round(float(sims[i]), 4),
            "meaning_loss":         round(float(1 - sims[i]), 4),
            "expected_code_words":  expected,
            "detected_zh":          zh_hits,
            "detected_en":          en_hits,
            "code_word_recall":     recall,
        })

    df = pd.DataFrame(records)
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

    # ── Summary ────────────────────────────────────────────────────────────────
    sep = "=" * 72
    print(f"\n{sep}")
    print("PIPELINE v2 — RESULTS SUMMARY")
    print(f"  Model ASR : BELLE-2/Belle-whisper-large-v3-zh")
    print(f"  Model MT  : facebook/nllb-200-1.3B")
    print(sep)
    print(f"{'ID':<4} {'CER':>6} {'Similarity':>11} {'Loss':>7} {'CW Recall':>10}")
    print("-" * 42)
    for r in records:
        print(f"{r['sample_id']:<4} {r['cer']:>6.4f} {r['meaning_similarity']:>11.4f} "
              f"{r['meaning_loss']:>7.4f} {r['code_word_recall']:>10.4f}")

    avg_cer  = df["cer"].mean()
    avg_sim  = df["meaning_similarity"].mean()
    avg_loss = df["meaning_loss"].mean()
    avg_cwr  = df["code_word_recall"].mean()

    print(f"\n{'AVG':<4} {avg_cer:>6.4f} {avg_sim:>11.4f} {avg_loss:>7.4f} {avg_cwr:>10.4f}")
    print(f"\n  Meaning preserved : {avg_sim*100:.1f}%")
    print(f"  Meaning lost      : {avg_loss*100:.1f}%")
    print(f"  Avg CER           : {avg_cer*100:.1f}%")
    print(f"  Code-word recall  : {avg_cwr*100:.1f}%")

    # v1 baseline for comparison
    print(f"\n  ── vs v1 baseline (nllb-600M + standard Whisper) ──")
    print(f"  v1 meaning preserved : ~64.7%")
    print(f"  v2 meaning preserved : {avg_sim*100:.1f}%")
    print(f"  Improvement          : +{(avg_sim - 0.6473)*100:.1f} points")

    # Save JSON report
    report = {
        "pipeline": {
            "asr_model":         "BELLE-2/Belle-whisper-large-v3-zh",
            "translation_model": "facebook/nllb-200-1.3B",
            "scoring_model":     "sentence-transformers/LaBSE",
        },
        "averages": {
            "cer":               round(avg_cer,  4),
            "meaning_similarity":round(avg_sim,  4),
            "meaning_loss":      round(avg_loss, 4),
            "code_word_recall":  round(avg_cwr,  4),
        },
        "samples": records,
    }
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved CSV  → {CSV_PATH}")
    print(f"  Saved JSON → {JSON_PATH}")


if __name__ == "__main__":
    main()
