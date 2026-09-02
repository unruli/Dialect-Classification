# Model selection and collaborator inference brief

**Decision date:** 31 August 2026
**Scope:** four architectural evaluation strata, two core systems per stratum

The labels G1--G4 are study-specific architectural strata, not a claim that
the literature universally recognizes four chronological model generations.
The core comparison contains eight individual systems. A model is included
only after it passes the common eligibility and inference checks below.

## Core model matrix

| ID | Architecture | Frozen candidate | Current status |
| --- | --- | --- | --- |
| **G1-A** | Embedding--clustering cascade | NeMo MarbleNet VAD + TitaNet-Large + NME-SC | Full 95-recording run complete |
| **G1-B** | Embedding--clustering cascade | MarbleNet VAD + [BUT SpeechFIT VBx](https://github.com/BUTSpeechFIT/VBx) | Full 95-recording run complete |
| **G2-A** | Neuralized overlap-aware modular | [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1) | Full 95-recording run complete |
| **G2-B** | Neuralized overlap-aware modular | NeMo [`diar_msdd_telephonic` v1.0.1](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/diar_msdd_telephonic) | Pending balanced pilot; telephone-domain specialization must be reported |
| **G3-A** | End-to-end discriminative | [`nvidia/diar_streaming_sortformer_4spk-v2.1`](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1) | Full CURC run complete: 95/95 successful and strict RTTM QC passed; AMI training overlap must be reported |
| **G3-B** | End-to-end discriminative | [BUT SpeechFIT DiaPer](https://github.com/BUTSpeechFIT/DiaPer), 10-attractor non-AMI-fine-tuned checkpoint | Conditional on longest-recording memory test |
| **G4-A** | Unified generative | [`OpenMOSS-Team/MOSS-Transcribe-Diarize`](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize) 0.9B | Four-domain 90-second CURC smoke passed 4/4 with strict RTTM QC; complete-recording pilots pending |
| **G4-B** | Unified generative | [`microsoft/VibeVoice-ASR-HF`](https://huggingface.co/microsoft/VibeVoice-ASR-HF) 8B | Four-domain deterministic smoke failed 0/4: three truncations and one malformed native output; do not scale under the frozen primary settings |

Pyannote `speaker-diarization-3.1` is an optional ninth run used only for a
within-pyannote version-sensitivity analysis. It is not G2-B and must not be
treated as an independent replication of the G2 architecture.

## Eligibility checklist

Each core system must satisfy every non-conditional requirement:

- [ ] Public checkpoint that can be pinned by version, commit, and/or checksum.
- [ ] Reproducible local inference with the code revision and environment saved.
- [ ] Anonymous, time-aligned speaker output that can be normalized to RTTM.
- [ ] Compatibility with mixed-speaker audio and the evaluation's four-speaker
      maximum.
- [ ] Full-recording processing for the evaluation's 49.54-minute maximum, or
      native state-preserving streaming that does not reset speaker identity.
- [ ] Fully automatic primary condition. Oracle speaker count is run separately
      only when the released interface supports it.
- [ ] Raw model output, normalized RTTM, logs, runtime, peak GPU memory, failures,
      and configuration provenance are preserved.
- [ ] The checkpoint's license, disclosed training datasets, input sample rate,
      context limit, and known domain specialization are recorded.

The locked evaluation envelope is defined by
[`dataset_metadata/final_evaluation_manifest.csv`](dataset_metadata/final_evaluation_manifest.csv):
95 recordings, 33.79 audio hours, at most four reference speakers, and no
recording longer than 49.54 minutes.

The executable-code handoff status and collaborator run commands are tracked in
[`inference/README.md`](inference/README.md). At the time of this decision note,
the proven G1-A/G1-B/G2-A runner still needs to be exported from
`levi-compute`; do not infer that this checkout can reproduce it merely from the
pilot report.

## What the collaborator should run

Run these in order on the collaborator's GPU:

1. **G3-A Streaming Sortformer** -- highest priority. It adds the missing
   end-to-end discriminative family and has a small released checkpoint with
   native long-form streaming.
2. **G4-A MOSS-Transcribe-Diarize** -- its four 90-second tests passed; run the
   four complete-recording pilots next. It adds the missing unified generative
   family, is 0.9B parameters, and supports up to 90 minutes.
3. **G2-B NeMo MSDD** -- run after the first two if time permits. It gives G2 an
   implementation independent of pyannote, but the public checkpoint is tuned
   for telephone speech and therefore requires a careful cross-domain pilot.
4. **G3-B DiaPer** -- pilot only until it processes the longest recording
   without out-of-memory failure. Do not launch the 95-file batch first.
5. **G4-B VibeVoice-ASR** -- halted at the 90-second smoke gate. It fit the
   20-GB A100 slice but produced three 4,096-token truncations and one malformed
   JSON-like output with seed 0; do not run complete pilots or the 95-file batch
   under these frozen settings.

This division avoids duplicating G1-A, G1-B, and G2-A, which are already being
run elsewhere. If only two models can be assigned, assign **G3-A and G4-A**.

## Common inference protocol

### 1. Preflight

1. Work in a separate environment for each model; do not upgrade or modify an
   existing successful environment.
2. Run `nvidia-smi`. If another process is using material GPU memory or compute,
   stop and report the PID/process rather than competing with it.
3. Record hostname, GPU, driver, CUDA reported by `nvidia-smi`, Python, PyTorch,
   model code revision, checkpoint revision/checksum, and run date.
4. Locate the collaborator's path-bearing inference manifest produced from the
   prepared datasets. Join it to the 95 path-free IDs in
   `dataset_metadata/final_evaluation_manifest.csv` using `dataset` and
   `recording_id`. Stop on missing IDs, duplicate IDs, or an ID count other than
   95. Do not copy or redistribute the complete corpora merely to match paths.

### 2. Pilot recordings

Use these fixed IDs so all model pilots cover both transfer and speaker-count
conditions:

| Dataset | Recording ID | Speakers | Pilot role |
| --- | --- | ---: | --- |
| AfriSpeech-Dialog | `5129fd8c-7b8c-4d05-a03a-196bcae4deff` | 2 | African-accented medical |
| Playlogue | `ew_42pc_22148` | 2 | Adult--child |
| AMI | `EN2002a` | 4 | Conventional multiparty meeting |
| Bangor Miami | `sastre03` | 3 | Code-switched and long-form |

For a new environment, first process a 90-second excerpt of each recording.
Then process the four complete pilot recordings. For G3-B and G4-B, add the
complete `EN2002c` recording (49.54 minutes, three speakers) as a mandatory
feasibility gate before any full batch.

### 3. Audio and inference rules

- Use the prepared 16-kHz mono waveform as the canonical input.
- A checkpoint-required model-native resample is allowed and must be logged;
  do not overwrite the canonical waveform.
- Run complete recordings. Native Sortformer streaming and VibeVoice's
  state-preserving tokenizer chunks are allowed. Do not create independent
  external chunks and stitch speaker labels unless that condition is approved
  and registered separately.
- Use automatic speaker count for the primary run. Never pass a reference
  speaker count to a model in the automatic condition.
- Use released weights without fine-tuning or evaluation-domain adaptation.
- Use deterministic decoding where available (`do_sample=False`, temperature
  zero). Do not use transcript hotwords, speaker names, or dataset-specific
  prompts.
- Preserve malformed or truncated generative outputs as failures. Do not repair
  them manually before archiving the raw output.

### 4. Model-specific settings

**G3-A:** use `nvidia/diar_streaming_sortformer_4spk-v2.1`, batch size 1, and
the released high-latency 30.4-second streaming configuration for the offline
evaluation. Save the native segment list and the normalized RTTM. The model has
four output speaker slots; it does not receive oracle speaker count.

**G4-A:** use `OpenMOSS-Team/MOSS-Transcribe-Diarize` 0.9B with the official
local inference package, pinned `trust_remote_code`, deterministic generation,
and the default timestamped speaker-diarization prompt. Save the complete raw
generated string before parsing `[start][Sxx]text[end]` segments. Detect and
report maximum-token truncation.

**G2-B:** use `diar_msdd_telephonic` v1.0.1 with the released five-scale MSDD
settings, official NeMo VAD/embedding components, `oracle_num_speakers=False`,
and maximum speakers set to 8. Do not retune thresholds on the evaluation
recordings. Save the NeMo RTTM and all intermediate configuration files.

**G3-B:** use DiaPer's 16-kHz 10-attractor checkpoint trained on simulated
conversations without AMI fine-tuning and the matching
`infer_16k_10attractors.yaml`. If the complete longest-file test fails, report
the failure and stop; do not introduce a clustering-based chunk stitcher.

**G4-B:** use `microsoft/VibeVoice-ASR-HF`, Transformers 5.3 or the pinned
compatible release, deterministic generation, and no contextual hotwords.
Save raw JSON-like output and the library's parsed output. A smaller
`acoustic_tokenizer_chunk_size` is permitted for memory because the released model
carries convolution state across those chunks; log the exact value.

### 5. Required output contract

Write each model under a separate directory such as:

```text
runs/architecture_audit/<SYSTEM_ID>/
  config/
  logs/
  raw/<dataset>/<recording_id>.*
  rttm/<dataset>/<recording_id>.rttm
  run_manifest.json
  failures.jsonl
```

Every normalized RTTM must contain standard 10-field `SPEAKER` records and
anonymous labels such as `SPEAKER_00`. Validate that starts and durations are
finite and nonnegative, and that output end times do not exceed the source by
more than 0.5 seconds. Do not replace the native raw labels in the raw archive.

`run_manifest.json` must contain the system ID, model/checkpoint and code
revisions, checkpoint checksum when practical, license, command/configuration,
environment package list, GPU/driver, seed, model-native sample rate, speaker
count mode, decoding parameters, start/end time, per-file runtime, peak GPU
memory, and counts of success, failure, malformed output, and truncation.

Do not score DER during the first handoff unless the scoring environment is
already frozen. The first handoff is complete when the raw outputs, normalized
RTTMs, validation report, and provenance manifest are available.

## Copy-paste instruction for the collaborator's coding agent

```text
You are running a speaker-diarization model pilot on a GPU machine that already
has lawful local access to the prepared evaluation audio and this repository.
Read MODEL_SELECTION_AND_INFERENCE.md completely before acting.

Your assigned models are, in order:
1. G3-A: nvidia/diar_streaming_sortformer_4spk-v2.1
2. G4-A: OpenMOSS-Team/MOSS-Transcribe-Diarize (0.9B)
3. G2-B: NVIDIA NeMo diar_msdd_telephonic v1.0.1, only after 1 and 2

Do not rerun G1-A, G1-B, G2-A, or pyannote 3.1. Do not start G3-B DiaPer or
G4-B VibeVoice unless I assign their gated pilots separately.

First inspect nvidia-smi. If the GPU is occupied, stop and report the processes.
Do not change any existing working conda environment; create one isolated
environment per assigned model. Discover the local path-bearing inference
manifest and join it to dataset_metadata/final_evaluation_manifest.csv by
dataset and recording_id. Require exactly 95 unique matched recordings.

For each assigned model, run 90-second smoke tests and then full-recording tests
for these IDs: 5129fd8c-7b8c-4d05-a03a-196bcae4deff, ew_42pc_22148, EN2002a,
and sastre03. Follow the model-specific settings and common protocol in the
brief. Do not externally chunk recordings or use reference speaker counts.

After all four full pilot recordings pass, continue that model over all 95
recordings with checkpoint and code revisions frozen. Preserve native raw
output, generate standard 10-field RTTM with anonymous SPEAKER_XX labels, and
write run_manifest.json, logs, and failures.jsonl under
runs/architecture_audit/<SYSTEM_ID>/. Never manually repair malformed output.

Stop and report instead of scaling up if there is an out-of-memory error,
unparseable output, missing/duplicate manifest IDs, speaker labels cannot remain
consistent over a complete recording, or output timing fails validation. In
your report, include completed/failed counts, total and per-file runtime, peak
GPU memory, exact model/code revisions, environment, and output paths. Do not
make accuracy claims or run DER unless the frozen scorer is already available.
```
