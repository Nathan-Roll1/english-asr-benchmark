"""
Shared scoring/audio harness for the ASR benchmark.

Every model notebook imports from here instead of redefining these functions.
This is what keeps scoring identical across models added at different times —
if WER is computed differently between asr_evaluation.ipynb and a later
notebook, the results are not comparable and the whole benchmark is void.

Usage in a Colab cell:
    import sys; sys.path.append("/content/drive/MyDrive/asr_benchmark/english-asr-benchmark")
    from common import read_audio, chunk, score, normalizer, SR
"""

import numpy as np
import soundfile as sf
import librosa
import jiwer
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

SR = 16000
normalizer = EnglishTextNormalizer({})


def read_audio(path):
    """Load any audio file as mono float32 at SR. Resamples if needed."""
    a, sr = sf.read(str(path), dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != SR:
        a = librosa.resample(a, orig_sr=sr, target_sr=SR)
    return a


def chunk(audio, win_s=480, hop_s=None, overlap_s=10):
    """Yield overlapping windows of a long audio array for models with a
    context limit. win_s/hop_s in seconds. Default 480s/470s (8 min windows,
    10s overlap) suits Parakeet; pass tighter values for models with a
    shorter cap (e.g. Canary: chunk(audio, 30, 28))."""
    hop_s = hop_s or (win_s - overlap_s)
    w, s, pos = win_s * SR, hop_s * SR, 0
    while pos < len(audio):
        c = audio[pos:pos + w]
        if len(c) < SR:
            break
        yield c
        pos += s


def score(ref_text, hyp_text):
    """Corpus-comparable error counts for one file. Both texts are run
    through the Whisper English normalizer before scoring — do this for
    every model so results stay comparable. Returns (counts_dict, normalized_hyp)."""
    ref = normalizer(ref_text)
    hyp = normalizer(hyp_text)
    m = jiwer.process_words(ref, hyp or "\u2205")
    return {
        "S": m.substitutions, "D": m.deletions, "I": m.insertions,
        "N": m.hits + m.substitutions + m.deletions,
        "hyp_words": len(hyp.split()),
    }, hyp


def rollup(sub):
    """Corpus-level (micro-averaged) WER over a group of per-file rows.
    Sums S/D/I/N across files and divides once — NOT the mean of per-file
    WERs, which would weight a 12-word clip the same as a 500-word one.
    `sub` needs columns: S, D, I, N, audio_s, infer_s, vram_gb, wer."""
    import pandas as pd
    return pd.Series({
        "wer":      (sub.S.sum() + sub.D.sum() + sub.I.sum()) / sub.N.sum() * 100,
        "sub_rate": sub.S.sum() / sub.N.sum() * 100,
        "del_rate": sub.D.sum() / sub.N.sum() * 100,
        "ins_rate": sub.I.sum() / sub.N.sum() * 100,
        "rtfx":     sub.audio_s.sum() / sub.infer_s.sum(),
        "vram_gb":  sub.vram_gb.max(),
        "ref_words": sub.N.sum(),
    })
