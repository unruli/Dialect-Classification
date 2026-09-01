# G2-A environment: pyannote/speaker-diarization-community-1

**CPU-only, confirmed structurally forced, not merely unused.** This driver
class (CUDA 12.2 max per `nvidia-smi`) cannot run `pyannote-audio` 4.x:
that package requires `torch>=2.8.0`, and no published torch wheel index at
or below cu124 carries torch>=2.8.0 (cu118 tops out at 2.7.1, cu121 at 2.5.1,
cu124 at 2.6.0) -- the earliest CUDA index that does is cu126, which needs a
newer driver than this class supports. There is no torch pin that satisfies
both pyannote 4.x's floor and a CUDA-12.2-class driver.

`run_g2a_pyannote.py` forces CPU explicitly (`pipeline.to(torch.device("cpu"))`
plus `CUDA_VISIBLE_DEVICES=""` recommended at invocation) regardless of what
`torch.cuda.is_available()` reports in this env -- that check has been
observed to give an unreliable `True` reading here after pyannote pipeline
load; ground-truth GPU usability with an actual `torch.zeros(1).cuda()`
allocation attempt, not `is_available()`.

## Setup

```bash
conda create -n <env-name> python=3.11 -y
conda activate <env-name>
pip install torch torchaudio  # resolves to a cu13x build under these constraints; CPU-only regardless
pip install pyannote-audio
```

## Full package list

See `environment.txt` (`pip freeze`). Key pins observed at last verified run:
torch 2.13.0+cu130, torchaudio 2.11.0+cu130, pyannote-audio 4.0.7.

## Authentication

Requires a Hugging Face account token (`HF_TOKEN` env var) from an account
that has separately clicked "Agree" on the model's own gate page. Set
`HF_HOME` to a writable cache dir (`--hf-home` passed through by
`run_model.py`'s `--cache-dir`).

## Checkpoint revision (at last verified run)

`pyannote/speaker-diarization-community-1` @ `3533c8cf8e369892e6b79ff1bf80f7b0286a54ee`
