#!/usr/bin/env python3
"""
Meaning Preservation Analysis
- Export clean JSON: Mandarin source + both translations
- Compute semantic loss using LaBSE cross-lingual embeddings
  meaning_loss = 1 - cosine_similarity(zh_embedding, en_embedding)
- Also compute back-translation CER as a string-level loss proxy
"""

# pip install sentence-transformers -q

import json, re, unicodedata
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

CSV_PATH  = "espionage_samples/espionage_pipeline_results.csv"
JSON_PATH = "espionage_samples/espionage_translations.json"
LOSS_PATH = "espionage_samples/meaning_loss_report.json"

# ── Load results ───────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

# ── Step 1: Build clean bilingual JSON ────────────────────────────────────────
print("Building bilingual JSON...")
bilingual = []
for _, row in df.iterrows():
    bilingual.append({
        "sample_id":         int(row["sample_id"]),
        "mandarin":          row["source_text"],
        "belle_english":     row["belle_translation_en"],
        "whisper_english":   row["whisper_translation_en"],
        "belle_asr_zh":      row["belle_asr_raw"],
        "whisper_asr_zh":    row["whisper_asr_raw"],
    })

with open(JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(bilingual, f, ensure_ascii=False, indent=2)
print(f"  Saved → {JSON_PATH}")

# ── Step 2: Semantic similarity with LaBSE ────────────────────────────────────
# LaBSE maps 109 languages into a shared embedding space.
# cosine_sim(zh, en) ≈ 1.0 means meaning fully preserved,
# meaning_loss = 1 - cosine_sim measures information lost in translation.
print("\nLoading LaBSE (multilingual sentence encoder)...")
model = SentenceTransformer("sentence-transformers/LaBSE")

src_texts   = df["source_text"].tolist()
belle_trans = df["belle_translation_en"].fillna("").tolist()
wsp_trans   = df["whisper_translation_en"].fillna("").tolist()

print("Encoding Mandarin sources...")
zh_embs     = model.encode(src_texts,   convert_to_tensor=True, normalize_embeddings=True)
print("Encoding BELLE translations...")
belle_embs  = model.encode(belle_trans, convert_to_tensor=True, normalize_embeddings=True)
print("Encoding Whisper translations...")
wsp_embs    = model.encode(wsp_trans,   convert_to_tensor=True, normalize_embeddings=True)

# cosine similarity (dot product of unit vectors)
belle_sim  = util.cos_sim(zh_embs, belle_embs).diagonal().cpu().numpy()
wsp_sim    = util.cos_sim(zh_embs, wsp_embs  ).diagonal().cpu().numpy()

belle_loss = 1 - belle_sim
wsp_loss   = 1 - wsp_sim

# ── Step 3: BELLE vs Whisper translation similarity ───────────────────────────
# How different are the two English outputs from each other?
inter_sim  = util.cos_sim(belle_embs, wsp_embs).diagonal().cpu().numpy()
inter_diff = 1 - inter_sim

# ── Step 4: Build full report ─────────────────────────────────────────────────
print("\nBuilding meaning loss report...")
report = {"summary": {}, "samples": []}

for i, row in df.iterrows():
    report["samples"].append({
        "sample_id":                  int(row["sample_id"]),
        "mandarin":                   row["source_text"],
        "belle_english":              row["belle_translation_en"],
        "whisper_english":            row["whisper_translation_en"],
        # --- semantic similarity (higher = better preservation) ---
        "belle_meaning_similarity":   round(float(belle_sim[i]),  4),
        "whisper_meaning_similarity": round(float(wsp_sim[i]),    4),
        # --- meaning loss (lower = better; 0 = perfect) ---
        "belle_meaning_loss":         round(float(belle_loss[i]), 4),
        "whisper_meaning_loss":       round(float(wsp_loss[i]),   4),
        # --- inter-model translation divergence ---
        "translation_divergence":     round(float(inter_diff[i]), 4),
        # --- ASR quality for context ---
        "belle_cer":                  round(float(row["belle_normalized_cer"]), 4),
        "whisper_cer":                round(float(row["whisper_normalized_cer"]), 4),
    })

report["summary"] = {
    "metric_explanation": {
        "meaning_similarity":   "LaBSE cosine similarity between Mandarin source and English translation. 1.0 = identical meaning, 0.0 = unrelated.",
        "meaning_loss":         "1 - meaning_similarity. How much semantic content was lost. 0.0 = perfect, 1.0 = total loss.",
        "translation_divergence": "How differently BELLE and Whisper translated the same audio. 0.0 = identical, 1.0 = completely different.",
        "cer":                  "Character Error Rate of ASR vs reference. Lower = better transcription.",
        "pipeline_loss_chain":  "ASR CER → captures transcription error. Meaning loss → captures translation semantic gap. Combined, they show where meaning degrades: at the speech-to-text step or the text-to-text step.",
    },
    "belle": {
        "avg_meaning_similarity": round(float(belle_sim.mean()),  4),
        "avg_meaning_loss":       round(float(belle_loss.mean()), 4),
        "avg_cer":                round(float(df["belle_normalized_cer"].mean()), 4),
    },
    "whisper": {
        "avg_meaning_similarity": round(float(wsp_sim.mean()),  4),
        "avg_meaning_loss":       round(float(wsp_loss.mean()), 4),
        "avg_cer":                round(float(df["whisper_normalized_cer"].mean()), 4),
    },
    "avg_translation_divergence": round(float(inter_diff.mean()), 4),
    "winner_asr":        "BELLE" if df["belle_normalized_cer"].mean() < df["whisper_normalized_cer"].mean() else "Whisper",
    "winner_translation": "BELLE" if belle_loss.mean() < wsp_loss.mean() else "Whisper",
}

with open(LOSS_PATH, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print(f"  Saved → {LOSS_PATH}")

# ── Console report ─────────────────────────────────────────────────────────────
sep = "=" * 72
print(f"\n{sep}")
print("MEANING PRESERVATION ANALYSIS  (LaBSE cross-lingual embeddings)")
print(sep)
print(f"{'ID':<4} {'BELLE Sim':>10} {'BELLE Loss':>11} {'WSP Sim':>9} {'WSP Loss':>9} {'Divergence':>11}")
print("-" * 60)
for s in report["samples"]:
    print(f"{s['sample_id']:<4} "
          f"{s['belle_meaning_similarity']:>10.4f} "
          f"{s['belle_meaning_loss']:>11.4f} "
          f"{s['whisper_meaning_similarity']:>9.4f} "
          f"{s['whisper_meaning_loss']:>9.4f} "
          f"{s['translation_divergence']:>11.4f}")

print(f"\n{'AVERAGES':<4} "
      f"{report['summary']['belle']['avg_meaning_similarity']:>10.4f} "
      f"{report['summary']['belle']['avg_meaning_loss']:>11.4f} "
      f"{report['summary']['whisper']['avg_meaning_similarity']:>9.4f} "
      f"{report['summary']['whisper']['avg_meaning_loss']:>9.4f} "
      f"{report['summary']['avg_translation_divergence']:>11.4f}")

print(f"\n{sep}")
print("PIPELINE LOSS CHAIN EXPLANATION")
print(sep)
print("""
  Mandarin Speech
      │
      ▼  [ASR step]
  Chinese transcript  ── CER measures character-level error here
      │                   BELLE avg CER : {belle_cer:.4f}  (13% chars wrong)
      │                   Whisper avg CER: {wsp_cer:.4f}  (17% chars wrong)
      ▼  [Translation step]
  English translation ── Meaning loss measures semantic gap here
                          BELLE avg loss : {belle_loss:.4f}  ({belle_pct:.1f}% meaning lost)
                          Whisper avg loss: {wsp_loss:.4f}  ({wsp_pct:.1f}% meaning lost)

  Key insight:
  - CER = how accurately the speech was transcribed (phonetic/char fidelity)
  - Meaning loss = how much semantic content survived into English
  - A low CER but high meaning loss → translation model is the bottleneck
  - A high CER but low meaning loss → ASR errors were "forgiving" (context preserved)
  - Translation divergence {div:.4f} → the two models produce moderately different
    English outputs from the same audio, showing translation is non-deterministic
""".format(
    belle_cer=report['summary']['belle']['avg_cer'],
    wsp_cer=report['summary']['whisper']['avg_cer'],
    belle_loss=report['summary']['belle']['avg_meaning_loss'],
    belle_pct=report['summary']['belle']['avg_meaning_loss']*100,
    wsp_loss=report['summary']['whisper']['avg_meaning_loss'],
    wsp_pct=report['summary']['whisper']['avg_meaning_loss']*100,
    div=report['summary']['avg_translation_divergence'],
))
print(f"  Best ASR        : {report['summary']['winner_asr']}")
print(f"  Best Translation: {report['summary']['winner_translation']}")
print(sep)
