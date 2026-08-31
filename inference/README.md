# Speaker-diarization inference runbook

This directory is the handoff point for the four-family inference study. The
goal is that a collaborator can clone the repository, select a model, point it
at an existing local audio manifest, and run the same validated command without
copying any corpus into Git.

## Code availability

**Updated 2026-08-31.** The G1-A/G1-B/G2-A runner has now been exported from
`levi-compute` (branch `codex/export-inference-runners`) as
`g1a_nemo/`, `g1b_vbx/`, `g2a_pyannote/`, and a shared `common/` module,
behind the single CLI described below. G3-A (`g3a_sortformer/`) has also been
implemented and **passed a live 90-second GPU smoke test** on `levi-compute`
during this export (device cuda, 7.97s wall, 918.5 MiB peak GPU memory,
2 speakers predicted on a 2-speaker pilot recording -- see that run's
`run_manifest.json` for the full record). G4-A (`g4a_moss/`) has been written
from its published model card but **has NOT been run** -- no environment was
built and no live test was performed for it in this export; treat it as
unvalidated until someone actually runs it.

| System | Implementation status | Should collaborator run it? |
| --- | --- | --- |
| G1-A | Exported, proven (95-recording run completed) | No, already run |
| G1-B | Exported, proven (95-recording run completed) | No, already run |
| G2-A | Exported, proven (95-recording run completed) | No, already run |
| G2-B | Not implemented (placeholder directory only) | Third priority |
| G3-A | Exported; 90-second GPU smoke test **passed** | **First priority** -- full pilot/batch still pending |
| G3-B | Not implemented | Do not batch yet |
| G4-A | Exported, **code-only, unvalidated** -- no environment built, no test run | **Second priority**, but validate the environment/smoke test first |
| G4-B | Not implemented | Do not batch yet |

The authoritative system definitions, eligibility checklist, pilot IDs, and
output contract are in
[`../MODEL_SELECTION_AND_INFERENCE.md`](../MODEL_SELECTION_AND_INFERENCE.md).

## Repository interface

```text
inference/
  README.md
  run_model.py
  common/              # manifest join/validation, RTTM normalize+validate, provenance
  g1a_nemo/            # MarbleNet VAD + TitaNet-Large + NME-SC (GPU)
  g1b_vbx/             # MarbleNet VAD (reused from G1-A) + BUT VBx (CPU/ONNX)
  g2a_pyannote/         # pyannote/speaker-diarization-community-1 (CPU-forced)
  g2b_msdd/            # placeholder -- not implemented
  g3a_sortformer/       # nvidia/diar_streaming_sortformer_4spk-v2.1 (GPU) -- smoke-tested
  g4a_moss/            # OpenMOSS-Team/MOSS-Transcribe-Diarize (unvalidated)
  legacy/              # historical scripts that produced the live G1-A/G1-B/G2-A run; superseded by run_model.py
```

The common command is:

```bash
python inference/run_model.py \
  --system G3-A \
  --path-manifest /local/path/to/inference_ready/manifest.csv \
  --selection-manifest dataset_metadata/final_evaluation_manifest.csv \
  --output-dir /local/path/to/runs/architecture_audit/G3-A \
  --pilot
```

Run each system under its own environment -- see that system's
`ENVIRONMENT.md`. `run_model.py` itself has no heavy dependencies and only
imports a system's adapter (and that adapter's own deps) once `--system` is
selected, so `--help` and `--validate-only` work under plain `python3`.

Verified behavior (this export):

1. Manifest join rejects duplicate IDs, missing IDs, and (for a `--full` run
   with no `--recording-id`/`--limit`) a matched count other than 95 --
   confirmed with three deliberate failure cases (missing IDs, a duplicate
   ID, and a valid run) during this export.
2. `--validate-only`, `--pilot`, `--full`, `--recording-id` (repeatable),
   `--limit`, and `--trim-seconds` (a single-recording smoke test without the
   second full-recording pass `--pilot` adds) are all implemented.
3. Reference speaker count is recorded (`n_speakers_reference` in
   `run_manifest.json`) for provenance only -- never passed to a model.
4. Raw output, normalized 10-field anonymous RTTM, per-file runtime, peak GPU
   memory (adapter-reported where available, else nvidia-smi-polled), and
   full command/environment/GPU provenance are written per recording.
5. Resume-safe: a recording is skipped only if its normalized RTTM both
   parses and validates against the source duration, AND the last recorded
   status for it is `success` -- a partial or failed prior attempt is retried,
   not silently treated as done.
6. No machine-specific absolute paths are hardcoded; `--path-manifest`,
   `--selection-manifest`, `--output-dir`, and `--cache-dir` are all required
   or explicit CLI arguments.

## Collaborator quick start

### 1. Clone and inspect

```bash
git clone https://github.com/unruli/Dialect-Classification.git
cd Dialect-Classification
git checkout codex/export-inference-runners   # or the merged commit, once merged to main
python inference/run_model.py --help
```

### 2. Check the GPU before installing or running

```bash
hostname
nvidia-smi
```

If `nvidia-smi` shows another process using material GPU memory or compute,
stop and tell the project owner. `run_model.py` itself also checks this
before any GPU run and refuses to proceed if the GPU is occupied.

### 3. Keep audio where it already lives

Do not copy a complete corpus just to match another machine's directory. Build
or reuse a local path-bearing CSV whose `audio_path` values point to the
collaborator's existing lawful copy of the prepared 16-kHz mono audio. The
repository's path-free
[`../dataset_metadata/final_evaluation_manifest.csv`](../dataset_metadata/final_evaluation_manifest.csv)
selects the frozen 95 recordings.

Validate the manifest without loading a model:

```bash
python inference/run_model.py \
  --system G3-A \
  --path-manifest /local/path/to/inference_ready/manifest.csv \
  --selection-manifest dataset_metadata/final_evaluation_manifest.csv \
  --output-dir /local/path/to/runs/architecture_audit/G3-A \
  --validate-only
```

### 4. Run the fixed pilot before the batch

Use one isolated environment per model (`ENVIRONMENT.md` in each system's
directory). Do not upgrade a working shared environment. Then run:

```bash
python inference/run_model.py \
  --system G3-A \
  --path-manifest /local/path/to/inference_ready/manifest.csv \
  --selection-manifest dataset_metadata/final_evaluation_manifest.csv \
  --output-dir /local/path/to/runs/architecture_audit/G3-A \
  --pilot
```

The fixed pilot is AfriSpeech-Dialog
`5129fd8c-7b8c-4d05-a03a-196bcae4deff`, Playlogue `ew_42pc_22148`, AMI
`EN2002a`, and Bangor Miami `sastre03`. `--pilot` runs a 90-second excerpt of
each first, then the complete recording -- this is the same pattern the G3-A
smoke test in this export used, just for one recording instead of all four.

### 5. Continue all 95 only after validation

```bash
python inference/run_model.py \
  --system G3-A \
  --path-manifest /local/path/to/inference_ready/manifest.csv \
  --selection-manifest dataset_metadata/final_evaluation_manifest.csv \
  --output-dir /local/path/to/runs/architecture_audit/G3-A \
  --full
```

Repeat with `--system G4-A` only after building and validating its own
environment -- see `g4a_moss/ENVIRONMENT.md` for the documented, unverified
risk with that system's GPU requirements. Run `G2-B` third if GPU time
permits, once it is implemented. Do not start G3-B or G4-B as a
95-recording batch until their longest-file gates pass.

Expected outputs (produced by `run_model.py`):

```text
<output-dir>/
  config/                     # one JSON per invocation: full CLI args
  logs/
  raw/<dataset>/<recording_id>.*      # native raw output, never overwritten with normalized data
  rttm/<dataset>/<recording_id>.rttm  # normalized, anonymous SPEAKER_XX, 10-field
  run_manifest.json           # system/checkpoint/env/GPU provenance + per-recording records
  failures.jsonl              # one line per non-success recording
```

## Compute-side export checklist

- [x] Every source/config file used by the successful 95-file run is included
      (`g1a_nemo/`, `g1b_vbx/`, `g2a_pyannote/`, `common/parse_rttm_standalone.py`).
- [x] The full-batch orchestrator and VBx invocation are included
      (`legacy/run_full_batch.sh`, `legacy/run_pilot_file.sh`, historical;
      `g1b_vbx/run_g1b_vbx.sh`, the generalized/parameterized current version).
- [x] NeMo writes a dedicated `--result-json`; log output is not parsed as JSON
      (`g1a_nemo/run_g1a_nemo.py`, `g3a_sortformer/run_g3a_sortformer.py`).
- [x] VBx output discovery handles its generated filename rather than assuming
      a hard-coded RTTM name (`g1b_vbx/run_g1b_vbx.sh`, `find ... -name "*.rttm"`).
- [x] Absolute `/home/kelechi` and `/dev/shm` paths are command-line options
      (verified: `grep` for both found no hits in `run_model.py` or any adapter).
- [x] No Hugging Face token, SSH key, audio, model cache, raw result, RTTM, or
      large log is committed (verified via `git diff --stat` before commit).
- [x] `python inference/run_model.py --help` succeeds from the repository root
      (tested under plain `python3`, no environment activated).
- [x] Manifest-only validation passes and selects exactly 95 unique IDs
      (tested against the real frozen selection manifest and a real local
      path manifest).
- [x] A 90-second G3-A smoke test succeeds on GPU before handing off the branch
      (7.97s wall, 918.5 MiB peak GPU memory, 2/2 speakers, `status: success`).
- [x] The branch/commit SHA and exact smoke-test command are reported (see the
      export report delivered alongside this branch).

## Prompt for the Claude agent on `levi-compute`

This section's original contents (asking for exactly the export completed in
this commit) are preserved in git history. See `legacy/README.md` and this
file's "Code availability" section above for what was actually done and
where it diverges from that original ask.
