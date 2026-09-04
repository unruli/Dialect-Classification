# Inference results: G1-A, G1-B, G2-A, G2-B

Frozen evaluation set: 95 recordings (17 AfriSpeech-Dialog, 16 AMI, 28 Bangor
Miami, 34 Playlogue), same set as `RESULTS.md`'s G3-A/G4-A. Outputs are
normalized 10-field anonymized RTTMs (no transcript text). Run on
levi-compute (RTX 4090, driver 535.309.01).

Every recording's RTTM was independently re-parsed and duration-bound
validated via `scripts/model_completion_status.py` (not just the runner's
own exit code) -- see `MODEL_COMPLETION_STATUS.csv` /
`MODEL_COMPLETION_SUMMARY.json` for the full per-recording record.
`scripts/` here holds that validator plus `compute_der_full95.py` (the DER/JER
scorer below) and `rttm_tools.py` -- as run on levi-compute, with a
hardcoded local `BASE` path; adjust that constant before reusing elsewhere.
Both scripts are scoped to the 4 systems below -- G3-A and G4-A aren't in
their `MODELS` lists (see "Note on G3-A" and "Note on G4-A").

## Naming

Directories here are named `<model>_full95` (e.g. `g1a_nemo_full95`) so the
underlying model is recoverable from the path alone. `G3-A/` and `G4-A/`
elsewhere in `runs/architecture_audit/` predate this and were deliberately
left as bare system IDs rather than renamed, to avoid rewriting a
colleague's already-merged contribution without checking with them first --
new work goes in as `<model>_full95` going forward.

## G1-A -- NeMo MarbleNet VAD + TitaNet embeddings + spectral clustering
- **95/95 success**, 0 failures. Checkpoints: `vad_marblenet@10477085`,
  `titanet_large@11ba0924`.
- Re-run once after an initial cross-recording RTTM-contamination bug in
  the batch driver (a shared, non-uri-keyed working directory) was found
  and fixed; all 95 independently re-validated post-fix.

## G1-B -- BUTSpeechFIT VBx (ONNX x-vector extraction, CPU) + AHC/VB-HMM
- **95/95 success**, 0 failures. Reuses G1-A's MarbleNet VAD output (no VAD
  of its own).

## G2-A -- pyannote community-1
- **95/95 success**, 0 failures. Checkpoint:
  `pyannote/speaker-diarization-community-1@3533c8cf`. Runs on CPU (its
  `torch>=2.8.0` floor has no cu-index wheel at or below cu124, and this
  host's driver caps at CUDA 12.2).

## G2-B -- NeMo `diar_msdd_telephonic`
- **95/95 success**, 0 failures. Wall time ~100,607s (~27.9h) -- driven by
  RTF, which varies 0.44-1.24x by content (AMI's dense multi-party turn-taking
  is the costliest).

## DER / JER

`DER_JER_recording_level_full95.csv` (475 rows = 95 recordings x 5 models,
includes G3-A alongside these 4 for a single combined table) /
`DER_JER_model_level_full95.csv` -- pyannote.metrics, overlap retained, DER
at collar=0.0 (primary/strict) and collar=0.25 (conventional), plus JER,
against each recording's human reference RTTM + UEM. Model order in the
model-level CSV is the fixed pipeline order (G1-A/G1-B/G2-A/G2-B/G3-A) --
not a ranking, per this project's own instructions.

## Note on G3-A

This export does **not** include a `G3-A/` directory -- one already exists
here from a colleague's HiPerGator run (see `RESULTS.md`). A separate,
independently-run G3-A 95/95 result also exists on levi-compute (same
checkpoint, same deterministic non-generative algorithm) but is deliberately
**not pushed** to avoid silently overwriting or duplicating the merged
copy. If cross-validating the two is useful, ask and it can be added
alongside (e.g. `G3-A-levicompute/`) rather than replacing what's here.

## Note on G4-A

Not run by this project's own pipeline -- 95/95 RTTMs from a colleague's
HiPerGator run, merged into `main` via PR #2 (see `RESULTS.md`). Pulled to
levi-compute and independently re-validated 2026-09-03 (re-parsed, duration-
bound checked, same gate as the 4 systems above): **95/95 valid, no
inconsistencies**. Folded into this project's local `MODEL_COMPLETION_STATUS`
and the DER/JER snapshot below, but `G4-A/` itself is not touched or
duplicated here -- same reasoning as G3-A above.

## Not included here

See `RESULTS.md`'s G3-B/G4-B notes and the `g3b-diaper-export` branch --
unchanged by this export.
