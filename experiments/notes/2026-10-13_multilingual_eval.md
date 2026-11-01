# 2026-10-13 multilingual evaluation and hardening

Dataset: data/evaluation/questions_v1.json (20 q x 6 languages; AI-written, translations unreviewed).
Runs: eval_v1_run1 (baseline) ... run5 (final). Run 2 stopped early; run 3 = intermediate config.

| run | config change | answerable answered | off-topic refused | live ok (own language) |
|---|---|---|---|---|
| 1 | baseline | 63/72 | 24/24 | 17/24 |
| 3 | greedy decoding, exact dense score, LLM evidence check, BM25 stopwords | 71/72 | 24/24 | 24/24 |
| 4 | + neutral translation prompt, hazard-gated lenient evidence check | 71/72 | 24/24 | 24/24 |
| 5 | + hazard glossary, wider live-info patterns | 72/72 | 24/24 | 24/24 |

Run 3 note: its single miss was the English flagship flood question (strict evidence check said NO).
Run 4 misses: Telugu/Kannada heat-wave questions mistranslated (fixed by glossary in run 5).

Voice (synthetic TTS speech, stress_run1.jsonl, final code before the VAD fix): 11/12 answerable spoken questions answered
(Malayalam flood question: degenerate ASR), 6/6 spoken off-topic refused, 6/6 spoken live-info refused.
Latency per spoken query 12-40 s (large-v3 ASR + translation + evidence check + generation + TTS).

Evidence-check ablation: strict prompt 11/12 kept, 20/20 off-topic refused; lenient prompt alone 12/12 kept,
18/20 refused (leaks: snake bite, driving test); strict+hazard-gated lenient 12/12 and 20/20.
Fusion comparison (answerable, MRR): minmax .844, rrf .844, dense .734, bm25 .819.
Whisper: small/medium wrote Telugu/Kannada/Malayalam as Devanagari (medium: correct script 4/10); large-v3 8/10, Malayalam fixed by in-language prompt retry.
