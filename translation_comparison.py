#!/usr/bin/env python3
"""
Translation model comparison:
  A. nllb-200-distilled-600M  (current baseline)
  B. nllb-200-1.3B            (larger NLLB)
  C. opus-mt-tc-big-zh-en     (specialized zh→en)
  D. nllb-200-1.3B from GOLD  (translate reference text, not ASR — shows translation ceiling)

Metric: LaBSE cosine similarity (zh_source vs en_translation) = meaning preservation
"""

import json, torch, warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sentence_transformers import SentenceTransformer, util

CSV_PATH = "espionage_samples/espionage_pipeline_results.csv"
OUT_JSON = "espionage_samples/translation_comparison.json"

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
PIPE_DEV    = 0 if DEVICE == "cuda" else -1
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Translation models run on CPU — Ollama occupies ~5.5 GiB of the 8 GiB GPU
MT_DEVICE = "cpu"
MT_DTYPE  = torch.float32

df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
sources_zh   = df["source_text"].tolist()
belle_asr_zh = df["belle_asr_normalized"].fillna("").tolist()

print(f"[config] device={DEVICE}\n")

# ── helpers ───────────────────────────────────────────────────────────────────
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import gc

def translate_nllb(texts, model_id, src_lang="zho_Hans", tgt_lang="eng_Latn"):
    print(f"  Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, dtype=MT_DTYPE).to(MT_DEVICE)
    model.eval()
    tokenizer.src_lang = src_lang
    forced_bos = tokenizer.convert_tokens_to_ids(tgt_lang)
    out_texts = []
    for t in tqdm(texts, desc=f"  {model_id.split('/')[-1]}"):
        if not t or not t.strip(): out_texts.append(""); continue
        try:
            inp = tokenizer(t, return_tensors="pt", truncation=True, max_length=512)
            inp = {k: v.to(MT_DEVICE) for k, v in inp.items()}
            with torch.no_grad():
                ids = model.generate(**inp, forced_bos_token_id=forced_bos, max_new_tokens=256)
            out_texts.append(tokenizer.decode(ids[0], skip_special_tokens=True))
        except Exception as e:
            print(f"  [warn] {e}"); out_texts.append("")
    del model, tokenizer; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return out_texts

def translate_opus(texts, model_id="Helsinki-NLP/opus-mt-zh-en"):
    print(f"  Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, dtype=MT_DTYPE).to(MT_DEVICE)
    model.eval()
    out_texts = []
    for t in tqdm(texts, desc=f"  {model_id.split('/')[-1]}"):
        if not t or not t.strip(): out_texts.append(""); continue
        try:
            inp = tokenizer(t, return_tensors="pt", truncation=True, max_length=512)
            inp = {k: v.to(MT_DEVICE) for k, v in inp.items()}
            with torch.no_grad():
                ids = model.generate(**inp, max_new_tokens=256)
            out_texts.append(tokenizer.decode(ids[0], skip_special_tokens=True))
        except Exception as e:
            print(f"  [warn] {e}"); out_texts.append("")
    del model, tokenizer; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return out_texts

# ── Step 1: Run ALL translations first (no LaBSE in memory) ──────────────────
print("[A] Baseline NLLB-600M (already in CSV)")
trans_A = df["belle_translation_en"].fillna("").tolist()

print("\n[B] NLLB-1.3B from BELLE ASR")
trans_B = translate_nllb(belle_asr_zh, "facebook/nllb-200-1.3B")

print("\n[C] opus-mt-zh-en from BELLE ASR")
trans_C = translate_opus(belle_asr_zh)

print("\n[D] NLLB-1.3B from GOLD Mandarin (translation ceiling)")
trans_D = translate_nllb(sources_zh, "facebook/nllb-200-1.3B")

# ── Step 2: Score everything with LaBSE (load once, after translations done) ──
print("\nLoading LaBSE scorer (translations complete, GPU clear)...")
labse = SentenceTransformer("sentence-transformers/LaBSE")

def score(src_texts, en_texts, label):
    zh_emb = labse.encode(src_texts, convert_to_tensor=True, normalize_embeddings=True)
    en_emb = labse.encode(en_texts,  convert_to_tensor=True, normalize_embeddings=True)
    sims   = util.cos_sim(zh_emb, en_emb).diagonal().cpu().numpy()
    print(f"  {label:<45} sim={sims.mean():.4f}  loss={1-sims.mean():.4f}")
    return sims

sims_A = score(sources_zh, trans_A, "A  nllb-600M  (ASR input)")
sims_B = score(sources_zh, trans_B, "B  nllb-1.3B  (ASR input)")
sims_C = score(sources_zh, trans_C, "C  opus-big   (ASR input)")
sims_D = score(sources_zh, trans_D, "D  nllb-1.3B  (GOLD input — ceiling)")

# ── Assemble results dict ──────────────────────────────────────────────────────
results = {}
for idx, sid in enumerate(df["sample_id"]):
    results[int(sid)] = {
        "mandarin":            sources_zh[idx],
        "A_nllb600m_from_asr": trans_A[idx], "A_similarity": round(float(sims_A[idx]),4), "A_meaning_loss": round(float(1-sims_A[idx]),4),
        "B_nllb1_3b_from_asr": trans_B[idx], "B_similarity": round(float(sims_B[idx]),4), "B_meaning_loss": round(float(1-sims_B[idx]),4),
        "C_opus_big_from_asr": trans_C[idx], "C_similarity": round(float(sims_C[idx]),4), "C_meaning_loss": round(float(1-sims_C[idx]),4),
        "D_nllb1_3b_from_gold":trans_D[idx], "D_similarity": round(float(sims_D[idx]),4), "D_meaning_loss": round(float(1-sims_D[idx]),4),
    }

avgs = {k: float(v.mean()) for k, v in zip(["A","B","C","D"],[sims_A,sims_B,sims_C,sims_D])}

# ── Summary table ──────────────────────────────────────────────────────────────
sep = "=" * 76
print(f"\n{sep}")
print("PER-SAMPLE COMPARISON  (LaBSE similarity — higher is better)")
print(sep)
print(f"{'ID':<4} {'A:nllb-600M':>12} {'B:nllb-1.3B':>12} {'C:opus-big':>11} {'D:gold-ceil':>12}")
print("-" * 54)
for sid, row in results.items():
    print(f"{sid:<4} "
          f"{row['A_similarity']:>12.4f} "
          f"{row['B_similarity']:>12.4f} "
          f"{row['C_similarity']:>11.4f} "
          f"{row['D_similarity']:>12.4f}")

avgs = {
    "A": sum(r["A_similarity"] for r in results.values()) / len(results),
    "B": sum(r["B_similarity"] for r in results.values()) / len(results),
    "C": sum(r["C_similarity"] for r in results.values()) / len(results),
    "D": sum(r["D_similarity"] for r in results.values()) / len(results),
}
print(f"\n{'AVG':<4} {avgs['A']:>12.4f} {avgs['B']:>12.4f} {avgs['C']:>11.4f} {avgs['D']:>12.4f}")

winner = max(["A","B","C"], key=lambda k: avgs[k])
names  = {"A":"nllb-600M","B":"nllb-1.3B","C":"opus-big"}
print(f"\n  Best model (from ASR input): {names[winner]}")
print(f"  ASR-induced loss  = ceiling(D) - best(B/C) = {avgs['D']-avgs[winner]:.4f}")
print(f"  Model-size gain   = best - baseline(A)     = {avgs[winner]-avgs['A']:.4f}")
print(f"  Total meaning loss= 1 - best               = {1-avgs[winner]:.4f}  ({(1-avgs[winner])*100:.1f}%)")

# ── Sample translations side-by-side ──────────────────────────────────────────
print(f"\n{sep}")
print("SAMPLE 1 — side by side")
print(sep)
r1 = results[1]
print(f"ZH source : {r1['mandarin']}")
print(f"A nllb-600M: {r1['A_nllb600m_from_asr']}")
print(f"B nllb-1.3B: {r1['B_nllb1_3b_from_asr']}")
print(f"C opus-big : {r1['C_opus_big_from_asr']}")
print(f"D gold-ceil: {r1['D_nllb1_3b_from_gold']}")

print(f"\n{sep}")
print("SAMPLE 6 — side by side")
print(sep)
r6 = results[6]
print(f"ZH source : {r6['mandarin']}")
print(f"A nllb-600M: {r6['A_nllb600m_from_asr']}")
print(f"B nllb-1.3B: {r6['B_nllb1_3b_from_asr']}")
print(f"C opus-big : {r6['C_opus_big_from_asr']}")
print(f"D gold-ceil: {r6['D_nllb1_3b_from_gold']}")

# ── Save ──────────────────────────────────────────────────────────────────────
final = {
    "summary": {
        "models": {
            "A": {"id": "facebook/nllb-200-distilled-600M", "input": "belle_asr",  "avg_similarity": round(avgs["A"],4), "avg_loss": round(1-avgs["A"],4)},
            "B": {"id": "facebook/nllb-200-1.3B",            "input": "belle_asr",  "avg_similarity": round(avgs["B"],4), "avg_loss": round(1-avgs["B"],4)},
            "C": {"id": "Helsinki-NLP/opus-mt-zh-en",         "input": "belle_asr",  "avg_similarity": round(avgs["C"],4), "avg_loss": round(1-avgs["C"],4)},
            "D": {"id": "facebook/nllb-200-1.3B",            "input": "gold_mandarin", "avg_similarity": round(avgs["D"],4), "avg_loss": round(1-avgs["D"],4), "note": "translation ceiling — no ASR error"},
        },
        "best_model_from_asr": names[winner],
        "gain_over_baseline":  round(avgs[winner]-avgs["A"], 4),
        "asr_induced_loss":    round(avgs["D"]-avgs[winner], 4),
    },
    "samples": list(results.values()),
}
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(final, f, ensure_ascii=False, indent=2)
print(f"\n  Saved → {OUT_JSON}")
