#!/usr/bin/env python3
"""
Mandarin Military/Espionage ASR Pipeline
TTS (MeloTTS) → Whisper ASR → NLLB Translation → Code Word Detection
"""

import os
import sys
import pandas as pd
import soundfile as sf

# ── Install dependencies ───────────────────────────────────────────────────────
def install_deps():
    import subprocess
    pkgs = [
        "git+https://github.com/myshell-ai/MeloTTS.git",
        "transformers", "torch", "torchaudio",
        "datasets", "soundfile", "accelerate",
        "unidic-lite", "mecab-python3",
    ]
    for pkg in pkgs:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
    # MeloTTS needs unidic data
    subprocess.check_call([sys.executable, "-m", "unidic", "download"])

# ── Dialogues ──────────────────────────────────────────────────────────────────
DIALOGUES = [
    "指挥官: 狐狸编队准备就绪，确认目标坐标35.6北纬。专家: 收到，狐狸一号至十号全员待命，夜莺卫星图像已传输。",
    "专家: 鲨鱼信号确认，方位120度，距离45海里，正向台湾方向。指挥官: 启动黑剑拦截程序，苍鹰中队立即升空。",
    "指挥官: 幽灵小队已通过铁拳火力掩护，成功渗透敌后。专家: 银狐线人确认敌指挥部位置，坐标已发送。",
    "专家: 毒蛇已成功渗透目标网络，获取指挥系统权限。指挥官: 启动暗流病毒，瘫痪敌方C4ISR系统。",
    "指挥官: 红旗雷达侦测到黑剑来袭，铁网系统启动。专家: 铁网拦截成功，黑剑残骸落入预定区域。",
    "专家: 银狐准备就绪，请求狐狸空中接应。指挥官: 狐狸三号正在接近，预计3分钟到达LZ Bravo。",
    "指挥官: 铁拳锁定目标，坐标23-45，反复三次确认。专家: 铁拳就绪，火力覆盖范围已设定。",
    "专家: 苍鹰中队报告，敌机已被击落两架。指挥官: 维持苍鹰编队高度，防止雷霆反击。",
    "指挥官: 迷雾电子战系统启动，压制红旗雷达信号。专家: 红旗信号已消失，敌方防空盲区形成。",
    "专家: 指挥部遭遇暗流攻击，网络已完全瘫痪。指挥官: 启动备用通信，银狐立即转移至B点。",
]

CODE_WORD_MAP = {
    "狐狸": "Fox(drone)",       "鲨鱼": "Shark(sub)",
    "苍鹰": "BlueEagle(fighter)","黑剑": "BlackSword(missile)",
    "幽灵": "Ghost(specops)",    "铁拳": "IronFist(artillery)",
    "夜莺": "Nightingale(sat)",  "毒蛇": "Viper(cyber)",
    "银狐": "SilverFox(defector)","红旗": "RedFlag(radar)",
    "钢龙": "SteelDragon(armor)","暗流": "DarkCurrent(cyber)",
    "铁网": "IronNet(defense)",  "雷霆": "Thunder(airstrike)",
    "迷雾": "Fog(EW)",
}

OUT_DIR = "espionage_samples"
os.makedirs(OUT_DIR, exist_ok=True)


# ── TASK 1 & 2: TTS with MeloTTS ──────────────────────────────────────────────
def generate_audio():
    print("\n=== TASK 1+2: MeloTTS Generation ===")
    from melo.api import TTS

    # MeloTTS ZH has one built-in Mandarin speaker; cycle speed for variety
    speeds = [0.85, 0.90, 0.95, 1.00, 1.05, 0.88, 0.93, 0.98, 1.02, 0.87]
    tts = TTS(language="ZH", device="auto")
    speaker_ids = tts.hps.data.spk2id
    spk = speaker_ids.get("ZH", list(speaker_ids.values())[0])

    wav_paths = []
    for i, (text, spd) in enumerate(zip(DIALOGUES, speeds), 1):
        out_path = os.path.join(OUT_DIR, f"espionage_{i:02d}.wav")
        tts.tts_to_file(text, spk, out_path, speed=spd)
        wav_paths.append(out_path)
        print(f"  [{i:02d}] saved → {out_path}")
    return wav_paths


# ── TASK 3: Whisper ASR ────────────────────────────────────────────────────────
def transcribe(wav_paths):
    print("\n=== TASK 3: Whisper large-v3 ASR ===")
    from transformers import pipeline as hf_pipeline

    asr = hf_pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3",
        generate_kwargs={"language": "chinese", "task": "transcribe"},
        device_map="auto",
        torch_dtype="auto",
    )

    results = []
    for i, wav in enumerate(wav_paths, 1):
        out = asr(wav)
        text = out["text"]
        print(f"  [{i:02d}] {text[:90]}...")
        results.append({
            "file": wav,
            "original": DIALOGUES[i - 1],
            "mandarin_asr": text,
        })
    return results


# ── TASK 4: NLLB Translation ───────────────────────────────────────────────────
def translate(results):
    print("\n=== TASK 4: NLLB-200 Translation (ZH → EN) ===")
    from transformers import pipeline as hf_pipeline

    translator = hf_pipeline(
        "translation",
        model="facebook/nllb-200-distilled-600M",
        src_lang="zho_Hans",
        tgt_lang="eng_Latn",
        device_map="auto",
    )

    for r in results:
        eng = translator(r["mandarin_asr"], max_length=512)[0]["translation_text"]
        r["english"] = eng
        print(f"  {eng[:90]}...")
    return results


# ── TASK 5: Code Word Detection ───────────────────────────────────────────────
def detect_code_words(results):
    print("\n=== TASK 5: Code Word Detection ===")
    for r in results:
        detected = {cw: CODE_WORD_MAP[cw]
                    for cw in CODE_WORD_MAP if cw in r["mandarin_asr"]}
        r["code_words_detected"] = list(detected.keys())
        r["code_words_english"] = list(detected.values())
        r["n_code_words"] = len(detected)

    df = pd.DataFrame(results)
    csv_path = "espionage_pipeline_results.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\nResults saved → {csv_path}")
    return df


# ── WER Calculation (optional) ────────────────────────────────────────────────
def compute_wer(ref: str, hyp: str) -> float:
    ref_tokens = list(ref.replace(" ", ""))
    hyp_tokens = list(hyp.replace(" ", ""))
    r, h = len(ref_tokens), len(hyp_tokens)
    dp = list(range(h + 1))
    for i in range(1, r + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, h + 1):
            tmp = dp[j]
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = tmp
    return dp[h] / max(r, 1)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    # install_deps()   # uncomment on first run

    wav_paths = generate_audio()
    results   = transcribe(wav_paths)
    results   = translate(results)

    for r in results:
        r["cer"] = round(compute_wer(r["original"], r["mandarin_asr"]), 4)

    df = detect_code_words(results)

    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)
    display_cols = ["file", "english", "code_words_english", "cer", "n_code_words"]
    print(df[display_cols].to_string(index=False))

    print(f"\nAvg CER       : {df['cer'].mean():.4f}")
    print(f"Avg code words: {df['n_code_words'].mean():.1f} / {len(CODE_WORD_MAP)}")
    total = sum(len(r["code_words_detected"]) for r in results)
    print(f"Total detected: {total} code-word hits across 10 files")


if __name__ == "__main__":
    main()
