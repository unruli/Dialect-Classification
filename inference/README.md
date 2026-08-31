# Speaker-diarization inference runbook

This directory is the handoff point for the four-family inference study. The
goal is that a collaborator can clone the repository, select a model, point it
at an existing local audio manifest, and run the same validated command without
copying any corpus into Git.

## Code availability

The successful three-system runner is **not yet present in this Git checkout**.
It currently exists on `levi-compute` under
`~/Dialect-Classification/diar_smoke/scripts/`. The pilot report identifies at
least these tested files:

- `run_g1_nemo.py` for G1-A (MarbleNet + TitaNet + NME-SC);
- `run_g2_pyannote.py` for G2-A (pyannote community-1);
- `parse_rttm.py` for output normalization; and
- `run_pilot_file.sh` for the G1-A/G1-B/G2-A per-file workflow.

The full 95-recording orchestration file, its configuration files, and the VBx
wrapper must also be exported from `levi-compute`; their exact filenames are
not recoverable from the report alone. Until those source files are imported
and checked, this document is a handoff contract rather than a claim that the
current checkout can reproduce the run.

| System | Implementation status | Should collaborator run it? |
| --- | --- | --- |
| G1-A | Proven on `levi-compute`; source export pending | No, already running/completed elsewhere |
| G1-B | Proven on `levi-compute`; source export pending | No, already running/completed elsewhere |
| G2-A | Proven on `levi-compute`; source export pending | No, already running/completed elsewhere |
| G2-B | Runner and balanced GPU pilot pending | Third priority |
| G3-A | Runner and balanced GPU pilot pending | **First priority** |
| G3-B | Longest-file feasibility gate pending | Do not batch yet |
| G4-A | Runner, parser, and balanced GPU pilot pending | **Second priority** |
| G4-B | 24-GB memory/parser gate pending | Do not batch yet |

The authoritative system definitions, eligibility checklist, pilot IDs, and
output contract are in
[`../MODEL_SELECTION_AND_INFERENCE.md`](../MODEL_SELECTION_AND_INFERENCE.md).

## Required repository interface

The compute-side export should provide this stable interface. Internal file
names may differ, but the collaborator should not have to edit Python source or
hard-coded machine paths.

```text
inference/
  README.md
  run_model.py
  common/
  g1a_nemo/
  g1b_vbx/
  g2a_pyannote/
  g2b_msdd/
  g3a_sortformer/
  g4a_moss/
```

The common command must be:

```bash
python inference/run_model.py \
  --system G3-A \
  --path-manifest /local/path/to/inference_ready/manifest.csv \
  --selection-manifest dataset_metadata/final_evaluation_manifest.csv \
  --output-dir /local/path/to/runs/architecture_audit/G3-A \
  --pilot
```

Required behavior:

1. Join the path-bearing manifest to the frozen selection manifest on
   `dataset` and `recording_id` and reject duplicates, missing IDs, or a count
   other than 95 for a full run.
2. Accept `--pilot`, `--full`, `--recording-id`, and `--limit` modes.
3. Never use the reference speaker count in the primary automatic condition.
4. Preserve native raw output, normalized 10-field RTTM, logs, failure records,
   per-file runtime, peak GPU memory, and exact model/environment provenance.
5. Resume safely: skip only recordings whose raw output, RTTM, and success
   metadata all validate. Never silently overwrite a partial or failed output.
6. Use command-line paths and cache variables; do not contain machine-specific
   absolute paths, access tokens, audio, checkpoints, or generated results.

## Collaborator quick start (after the runner is exported)

### 1. Clone and inspect

```bash
git clone https://github.com/unruli/Dialect-Classification.git
cd Dialect-Classification
git pull --ff-only
python inference/run_model.py --help
```

The person exporting the runner must provide the branch or commit SHA. If the
code has not been merged into `main`, check out that exact branch or SHA before
continuing.

### 2. Check the GPU before installing or running

```bash
hostname
nvidia-smi
```

If `nvidia-smi` shows another process using material GPU memory or compute,
stop and tell the project owner. Record the GPU name, driver version, and CUDA
version shown by `nvidia-smi`.

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

Use one isolated environment per model. Follow the environment file or setup
script shipped with that model's runner; do not upgrade a working shared
environment. Then run:

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
`EN2002a`, and Bangor Miami `sastre03`. Each must first pass a 90-second smoke
test and then a complete-recording test.

### 5. Continue all 95 only after validation

```bash
python inference/run_model.py \
  --system G3-A \
  --path-manifest /local/path/to/inference_ready/manifest.csv \
  --selection-manifest dataset_metadata/final_evaluation_manifest.csv \
  --output-dir /local/path/to/runs/architecture_audit/G3-A \
  --full
```

Repeat with `--system G4-A` only after its own environment and parser pilot
pass. Run `G2-B` third if GPU time permits. Do not start G3-B or G4-B as a
95-recording batch until their longest-file gates pass.

Expected outputs are:

```text
runs/architecture_audit/<SYSTEM_ID>/
  config/
  logs/
  raw/<dataset>/<recording_id>.*
  rttm/<dataset>/<recording_id>.rttm
  run_manifest.json
  failures.jsonl
```

## Compute-side export checklist

Before telling a collaborator to use the code, the exporter must confirm:

- [ ] Every source/config file used by the successful 95-file run is included.
- [ ] The full-batch orchestrator and VBx invocation are included, not only the
      four pilot-report filenames.
- [ ] NeMo writes a dedicated `--result-json`; log output is not parsed as JSON.
- [ ] VBx output discovery handles its generated filename rather than assuming
      a hard-coded RTTM name.
- [ ] Absolute `/home/kelechi` and `/dev/shm` paths are command-line options.
- [ ] No Hugging Face token, SSH key, audio, model cache, raw result, RTTM, or
      large log is committed.
- [ ] `python inference/run_model.py --help` succeeds from the repository root.
- [ ] Manifest-only validation passes and selects exactly 95 unique IDs.
- [ ] A 90-second G3-A smoke test succeeds on GPU before handing off the branch.
- [ ] The branch/commit SHA and exact smoke-test command are reported.

## Prompt for the Claude agent on `levi-compute`

Copy the following prompt verbatim. It asks the machine that holds the proven
runner to export it safely and then add the missing collaborator entry points.

```text
Work inside ~/Dialect-Classification on levi-compute. Do not launch the full
95-recording inference run. First read MODEL_SELECTION_AND_INFERENCE.md and
inference/README.md completely.

Goal: export the exact source and configuration used by the successful
G1-A/G1-B/G2-A pilot/full run, and provide one documented, path-parameterized
CLI for the collaborator's pending G3-A and G4-A pilots.

1. Inventory the actual source files imported or executed by the running/full
pipeline. The report names diar_smoke/scripts/run_g1_nemo.py,
run_g2_pyannote.py, parse_rttm.py, and run_pilot_file.sh, but you must also find
and include the full 95-file orchestrator, VBx wrapper/invocation, YAML/JSON
configs, parser/validator code, and environment lock/spec files.
2. Preserve the two known fixes: NeMo must write a dedicated --result-json
instead of parsing logger stdout, and VBx result discovery must handle the
filename it actually generates.
3. Refactor copies into inference/ without changing or deleting the currently
running artifacts. Implement inference/run_model.py with the CLI contract in
inference/README.md: --system, --path-manifest, --selection-manifest,
--output-dir, and exactly one of --validate-only/--pilot/--full, plus optional
--recording-id and --limit. Remove hard-coded /home/kelechi and /dev/shm paths.
4. Add isolated environment/setup files and pinned model/code revisions. Do
not include environments themselves, downloaded checkpoints, caches, tokens,
audio, raw outputs, RTTMs, or large logs.
5. Add G3-A nvidia/diar_streaming_sortformer_4spk-v2.1 first. Add G4-A
OpenMOSS-Team/MOSS-Transcribe-Diarize 0.9B second. Follow the frozen settings
and output contract in MODEL_SELECTION_AND_INFERENCE.md. Do not implement a
different model under those IDs.
6. Run only read-only/preflight checks, --help, manifest validation, syntax/unit
tests, and one 90-second G3-A smoke test if the GPU is free. Check nvidia-smi
first; if occupied, skip the GPU smoke test and report the process. Do not run
DER and do not start a full dataset batch.
7. Update inference/README.md only where the tested commands differ from its
interface. Mark G4-A unvalidated if its GPU smoke test was not actually run.
8. Inspect git diff for secrets and large/generated files. Commit only the
inference code/docs/configs on a new branch named
codex/export-inference-runners and push that branch to origin. Do not add the
audio, model caches, run outputs, unrelated dirty files, or credentials.

Report: pushed branch, commit SHA, every exported file, exact successful test
commands, GPU/driver/PyTorch versions, model/checkpoint revisions, and any
remaining blocker. If you cannot push, create a tar.gz containing only those
source/docs/config files and report its absolute path and SHA-256 checksum.
```
