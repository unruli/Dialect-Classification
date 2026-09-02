# Inference results: G3-A (Sortformer) and G4-A (MOSS)

Frozen evaluation set: 95 recordings (17 AfriSpeech-Dialog, 16 AMI, 28 Bangor Miami, 34 Playlogue).
Runner: `inference/run_model.py`. Outputs are normalized 10-field anonymized RTTMs (no transcript text).
Run on HiPerGator (NVIDIA L4, driver 580.178, torch cu118 for G3-A / cu128 for G4-A).

## G3-A -- nvidia/diar_streaming_sortformer_4spk-v2.1
- **95/95 success**, 0 failures. Checkpoint rev `fafaab5faa1617a0ca52d38dd3dc4bd636800d3d`.
- Per-dataset: afrispeech 17/17, ami 16/16, bangor 28/28, playlogue 34/34.
- ~30 min total. Speaker counts exact on AMI/AfriSpeech; over-predicts on 2-3 spk files (4-slot model).

## G4-A -- OpenMOSS-Team/MOSS-Transcribe-Diarize (0.9B, generative)
- **95/95 have RTTMs** (87 direct single-pass at 65536 tokens + 8 recovered via chunking).
- 8 token-dense long recordings (EN2002c + 7 Bangor) truncated single-pass even at 131072 tokens
  (bilingual code-switch = ~40-50 turns/min); recovered via chunked inference with overlap-based
  speaker linking. Those RTTMs have full temporal coverage but cross-chunk speaker labels may over-fragment.
- The 8 recovered files are a **secondary recovery condition**, not primary
  single-pass successes under the frozen protocol; primary-condition reporting
  must preserve the 87/8 split.

## Provenance / caveats
- AMI is in Sortformer's training data (G3-A AMI numbers are not an independent test).
- No DER scoring in this handoff (execution/runtime study only).
- G4-A `run_manifest.json` was rebuilt from on-disk RTTMs + logs after 4 parallel jobs raced on the
  shared manifest file (original saved as `run_manifest.raced.json`).
