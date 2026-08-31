# G4-A environment: OpenMOSS-Team/MOSS-Transcribe-Diarize (0.9B)

**UNVALIDATED.** No environment was built and no live test was run for this
system as part of this export -- only `run_g4a_moss.py`'s syntax/imports were
checked. Do not treat that as evidence it runs correctly.

## Setup (per the published model card, not independently verified here)

Requires **Python 3.12+**, a separate environment from G1-A/G1-B/G2-A/G3-A:

```bash
conda create -n <env-name> python=3.12 -y
conda activate <env-name>
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchaudio
pip install transformers
# plus the model's own helper package (clone + `pip install -e .`) for
# moss_transcribe_diarize's message-construction/parsing utilities -- confirm
# the exact repository URL from the model card at setup time.
```

## Known integration risk -- confirm before relying on GPU here

The model card's own instructions install a **cu128** torch build. Every
other GPU system in this project (G1-A, G3-A) runs on a driver class capped
at CUDA 12.2 (`nvidia-smi` reports "CUDA Version: 12.2"), and this project
already hit and documented the identical class of problem for G2-A
(`pyannote-audio` 4.x's torch>=2.8.0 floor has no cu-index build at or below
cu124). A cu128 build very likely has the same driver-compatibility problem
here. Confirm against the actual target driver (`nvidia-smi`) before assuming
GPU works; a CPU fallback is a real possibility, in which case `--device cpu`
should be used and the runtime budgeted accordingly (this 0.9B generative
model will be considerably slower on CPU than the embedding/clustering
systems in this project).

## License / gating

Apache License 2.0. No gating documented on the model card as of this
export; no `HF_TOKEN` requirement is currently coded into `run_g4a_moss.py`.

## Checkpoint

`OpenMOSS-Team/MOSS-Transcribe-Diarize` -- resolved from the HF Hub
identifier at run time; no specific commit/revision has been pinned or
verified in this export.
