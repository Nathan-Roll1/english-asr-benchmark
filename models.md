# Models

Status of every model in scope for this benchmark. Update this table as models are added.

| Model | Params | Status | Notebook | Notes |
|---|---|---|---|---|
| [Parakeet-TDT-0.6B-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) | 0.6B | ✅ Done | `asr_evaluation.ipynb` | Recommended — see [`report/`](report/) |
| [Canary-1B-Flash](https://huggingface.co/nvidia/canary-1b-flash) | 1B | ✅ Done | `asr_evaluation.ipynb` | ~40s input cap forces chunking on long audio |
| [faster-whisper-large-v3](https://github.com/SYSTRAN/faster-whisper) | 1.5B | ✅ Done | `asr_evaluation.ipynb` | CTranslate2 runtime; VRAM measured via `nvidia-smi`, not PyTorch |
| [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | 1.7B (2B incl. audio encoder) | 🔜 Planned | `qwen3_asr_eval.ipynb` | Strong published English WER; also does streaming + LID for 30 languages. Published Open ASR Leaderboard AMI WER: 10.56 (note: likely scored against a cleaner AMI reference than ours — not directly comparable to our Tier C numbers without checking) |
| [FireRedASR2-AED](https://huggingface.co/FireRedTeam/FireRedASR2-AED) | — | 🔜 Planned | `fireredasr2_eval.ipynb` | Published benchmarks are Mandarin/Chinese-dialect CER only — **English support unverified**. Confirm before investing benchmark time; may not belong in an English-only comparison |
| Vosk | — | ⏸️ Deferred | — | CPU-only, ~2x WER of GPU models. Being evaluated separately by [[mahad-rehman]] |
| Qwen2-Audio | — | ❌ Dropped | — | Audio-language model, not production ASR. Much slower with no accuracy gain |

## Adding a new model

1. Add a row above (Planned → Done as you go).
2. New models get their **own notebook** (`<model>_eval.ipynb`), not edits to `asr_evaluation.ipynb`. Reuse the harness: import `chunk()`, `score()`, `read_audio()` from `common.py` so scoring stays identical across every model.
3. Run against the same three tiers in `asr_benchmark/` on Drive. Don't introduce a new dataset without adding it to `results/README.md`.
4. Append — never overwrite — `results/per_item.csv`. Same schema: `model, tier, item, band, audio_s, S, D, I, N, infer_s, vram_gb`.
5. Re-run `aggregate.py` (Cells 8–9 of the harness) to regenerate `summary_*.csv` and figures from the combined file.
6. Update the headline table in `README.md` and this file.
