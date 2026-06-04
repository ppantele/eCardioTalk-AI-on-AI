#!/usr/bin/env python3
"""
Generate TTS narration (Kokoro) for all 12 model/question pairs.
==================================================================
Per-model voice from config_video.BRAND (Claude bf_emma, ChatGPT bm_lewis,
Gemini am_onyx) + text preprocessing that strips markdown and naturalises
sentence structure for spoken delivery.

Outputs: video_pipeline/audio/q{1-4}_{claude,openai,gemini}.wav

Usage:
    python video_pipeline/generate_tts.py
    python video_pipeline/generate_tts.py --only claude q1   # single file
    python video_pipeline/generate_tts.py --preview-text     # print preprocessed text only
"""

import argparse
import json
import pathlib
import re
import sys

import numpy as np
import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from video_pipeline.config_video import (
    BRAND, INTERVIEW_JSON, AUDIO_DIR, TTS_SAMPLE_RATE, TTS_SPEED,
)


# ---------------------------------------------------------------------------
# Text preprocessing — make interview answers speech-ready
# ---------------------------------------------------------------------------

def preprocess_for_speech(text: str) -> str:
    """
    Transform written interview text into clean, natural spoken prose.

    Handles:
      1. Markdown symbol stripping (** * _ # ` | > [...])
      2. Prosody normalisation (em-dashes, semicolons, colons)
      3. Abbreviation expansion (e.g., i.e.)
      4. Long-sentence splitting
      5. Parenthetical removal
    """

    # ── 1. Markdown / symbol stripping ─────────────────────────────────────
    # Double asterisk bold: **text** → text
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text, flags=re.DOTALL)
    # Single asterisk italic: *text* → text  (guard against lone * in lists)
    text = re.sub(r'(?<!\*)\*(.+?)\*(?!\*)', r'\1', text, flags=re.DOTALL)
    # Underscored bold/italic
    text = re.sub(r'__(.+?)__', r'\1', text)
    text = re.sub(r'_(.+?)_', r'\1', text)
    # Heading hashes (# at line start)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Backtick code spans
    text = re.sub(r'`+(.+?)`+', r'\1', text, flags=re.DOTALL)
    # Markdown links [label](url) → label
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Pipe characters (table separators or list markers)
    text = re.sub(r'\|', '', text)
    # Blockquote markers
    text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)
    # Bullet list markers (-, •, *, at line start)
    text = re.sub(r'^[\s]*[-•]\s+', '', text, flags=re.MULTILINE)
    # Numbered list markers (1. 2. at line start)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)

    # ── 2. Remaining lone asterisks / underscores (safety net) ─────────────
    text = re.sub(r'\*', '', text)   # kill any stray * left over

    # ── 3. Prosody normalisation ────────────────────────────────────────────
    # Em-dash (—) → comma-space  (mid-sentence interruption feels more natural)
    text = re.sub(r'\s*—\s*', ', ', text)
    # Semicolons → period + capital (full stop creates natural pause)
    text = re.sub(r';\s*([a-z])', lambda m: '. ' + m.group(1).upper(), text)
    text = re.sub(r';', '.', text)   # remaining semicolons → period
    # Colon before a new line or start of list → period
    text = re.sub(r':\s*\n', '.\n', text)
    # Multiple spaces / tabs → single space
    text = re.sub(r'[ \t]{2,}', ' ', text)

    # ── 4. Abbreviation expansion ───────────────────────────────────────────
    text = re.sub(r'\be\.g\.\s*', 'for example, ', text)
    text = re.sub(r'\bi\.e\.\s*', 'that is, ', text)
    text = re.sub(r'\betc\.\s*', 'and so on. ', text)
    text = re.sub(r'\bvs\.\s*', 'versus ', text)
    text = re.sub(r'\bDr\.\s+', 'Doctor ', text)
    text = re.sub(r'\bProf\.\s+', 'Professor ', text)
    text = re.sub(r'\bFig\.\s+', 'Figure ', text)

    # ── 5. Remove parenthetical asides >7 words (disrupt spoken flow) ───────
    def _drop_long_parens(m: re.Match) -> str:
        inner = m.group(1).strip()
        return '' if len(inner.split()) > 7 else f'({inner})'
    text = re.sub(r'\(([^)]{40,})\)', _drop_long_parens, text)  # 40 chars ≈ 7-8 words
    # Clean up any double spaces left after removal
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r' ,', ',', text)
    text = re.sub(r' \.', '.', text)

    # ── 6. Split sentences that are too long (>42 words) ────────────────────
    SPLIT_WORDS = r'\b(but|and yet|and|however|where|which|because|although|whereas)\b'
    lines = []
    for sentence in re.split(r'(?<=[.!?])\s+', text):
        words = sentence.split()
        if len(words) <= 42:
            lines.append(sentence)
            continue
        # Try to split at a conjunction after at least 15 words
        partial = ' '.join(words[:15])
        remainder = ' '.join(words[15:])
        m = re.search(SPLIT_WORDS, remainder, re.IGNORECASE)
        if m:
            split_pt = 15 + len(remainder[:m.start()].split())
            part1 = ' '.join(words[:split_pt]).rstrip(',')
            part2 = words[split_pt].capitalize() + ' ' + ' '.join(words[split_pt + 1:])
            lines.append(part1 + '.')
            lines.append(part2)
        else:
            lines.append(sentence)

    text = ' '.join(lines)

    # ── 7. Final whitespace cleanup ─────────────────────────────────────────
    text = re.sub(r'\n{3,}', '\n\n', text)    # max 2 consecutive blank lines
    text = re.sub(r' {2,}', ' ', text)
    text = text.strip()

    return text


# ---------------------------------------------------------------------------
# Main TTS synthesis
# ---------------------------------------------------------------------------

def synthesise(model_key: str, q_id: str, raw_text: str,
               dest: pathlib.Path) -> pathlib.Path:
    """Preprocess text and synthesise to WAV using Kokoro."""
    from kokoro import KPipeline

    clean_text = preprocess_for_speech(raw_text)
    cfg       = BRAND[model_key]
    voice     = cfg["voice"]
    lang_code = cfg["lang_code"]
    speed     = cfg.get("speed", TTS_SPEED)

    out_path = dest / f"{q_id}_{model_key}.wav"

    pipe = KPipeline(lang_code=lang_code)
    chunks = []
    for _gs, _ps, audio in pipe(clean_text, voice=voice, speed=speed,
                                 split_pattern=r'\n+'):
        if audio is not None and len(audio) > 0:
            chunks.append(audio)

    if not chunks:
        raise RuntimeError(f"[{model_key}/{q_id}] Kokoro returned no audio.")

    combined = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    dest.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), combined, TTS_SAMPLE_RATE)

    duration = len(combined) / TTS_SAMPLE_RATE
    word_count = len(raw_text.split())
    tqdm.write(f"    [{model_key}/{q_id}]  {out_path.name}  "
               f"({duration:.1f}s  ·  {word_count} words  ·  voice={voice}  speed={speed})")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate TTS audio (V2: improved voices + text preprocessing).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--only", nargs=2, metavar=("MODEL", "QID"),
                   help="Generate one file: e.g. --only claude q1")
    p.add_argument("--preview-text", action="store_true",
                   help="Print preprocessed text for each answer without synthesising.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    print(f"\neCardio Vodcast — Stage 2 (V2): TTS Generation")
    print(f"  Voices : claude=bm_lewis(0.87)  openai=am_adam(0.91)  gemini=af_heart(0.90)")
    print(f"  Output : {AUDIO_DIR}/\n")

    with open(INTERVIEW_JSON, encoding="utf-8") as f:
        interview = json.load(f)
    segments = interview["segments"]

    if args.only:
        model_key, q_id = args.only
        tasks = [(model_key, q_id)]
    else:
        tasks = [(mk, seg["id"]) for seg in segments for mk in BRAND]

    if args.preview_text:
        print("=== Preprocessed text preview ===\n")
        for model_key, q_id in tasks:
            seg  = next(s for s in segments if s["id"] == q_id)
            raw  = seg["answers"][model_key]["text"]
            clean = preprocess_for_speech(raw)
            print(f"--- {q_id}/{model_key} ---")
            print(clean[:600])
            print("…\n")
        return 0

    print(f"  Tasks  : {len(tasks)} file(s)\n")

    # Group by lang_code to instantiate one Kokoro pipeline per language
    from kokoro import KPipeline

    generated = []

    for lang_code in ("b", "a"):
        models_this = [mk for mk in BRAND if BRAND[mk]["lang_code"] == lang_code]
        relevant    = [(mk, qid) for (mk, qid) in tasks if mk in models_this]
        if not relevant:
            continue

        label = "British" if lang_code == "b" else "American"
        print(f"  — {label} English  ({', '.join(models_this)}) —")

        pipe = KPipeline(lang_code=lang_code)

        for model_key, q_id in tqdm(relevant, desc=f"  TTS/{label}", unit="file"):
            seg      = next(s for s in segments if s["id"] == q_id)
            raw_text = seg["answers"][model_key]["text"]
            clean    = preprocess_for_speech(raw_text)
            voice    = BRAND[model_key]["voice"]
            speed    = BRAND[model_key].get("speed", TTS_SPEED)

            out_path = AUDIO_DIR / f"{q_id}_{model_key}.wav"
            AUDIO_DIR.mkdir(parents=True, exist_ok=True)

            chunks = []
            for _gs, _ps, audio in pipe(clean, voice=voice, speed=speed,
                                         split_pattern=r'\n+'):
                if audio is not None and len(audio) > 0:
                    chunks.append(audio)

            if not chunks:
                tqdm.write(f"\n  ⚠  [{model_key}/{q_id}] no audio — skipping.")
                continue

            combined = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
            sf.write(str(out_path), combined, TTS_SAMPLE_RATE)
            dur = len(combined) / TTS_SAMPLE_RATE
            tqdm.write(f"    [{model_key}/{q_id}]  {out_path.name}  "
                       f"({dur:.1f}s  voice={voice}  speed={speed})")
            generated.append(out_path)

        del pipe

    print(f"\n✓  {len(generated)} / {len(tasks)} files  →  {AUDIO_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
