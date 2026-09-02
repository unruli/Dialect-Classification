# G4-A environment: OpenMOSS-Team/MOSS-Transcribe-Diarize (0.9B)

**UNVALIDATED.** No environment was built and no live test was run for this
system as part of this export -- only `run_g4a_moss.py`'s syntax/imports were
checked. Do not treat that as evidence it runs correctly.

## Setup (per the published model card, not independently verified here)

Requires **Python 3.12+**, a separate environment from G1-A/G1-B/G2-A/G3-A:

```bash
conda create -n <env-name> python=3.12 -y
conda activate <env-name>

git clone https://github.com/OpenMOSS/MOSS-Transcribe-Diarize.git
cd MOSS-Transcribe-Diarize
git checkout 61bc29cd4120be7b5d3b761b64cd5dff57263642
uv pip install -e ".[torch-runtime]" --torch-backend=auto
```

That installs `moss_transcribe_diarize` (the package `run_g4a_moss.py` imports:
`parse_transcript`, and from `moss_transcribe_diarize.inference_utils`:
`build_transcription_messages`, `generate_transcription`, `resolve_device`)
plus a matching torch/transformers stack, per the repository's own
`torch-runtime` extra. The repo also ships `mtd-subtitle` (batch CLI) and
`mtd-subtitle-web` (web UI), neither of which this project uses --
`run_g4a_moss.py` calls the Python helpers directly.

## Known integration risk -- confirm before relying on GPU here

Two install paths are documented for this model and they may not resolve the
same torch build: the HF model card's standalone snippet pins
`--index-url https://download.pytorch.org/whl/cu128`, while the GitHub repo's
own instructions use `uv pip install -e ".[torch-runtime]" --torch-backend=auto`,
which may autodetect a driver-appropriate build instead of forcing cu128.
**Neither has been verified in this export.** If the resolved build does turn
out to be cu128: every other GPU system in this project (G1-A, G3-A) runs on
a driver class capped at CUDA 12.2 (`nvidia-smi` reports "CUDA Version: 12.2"),
and this project already hit and documented the identical class of problem
for G2-A (`pyannote-audio` 4.x's torch>=2.8.0 floor has no cu-index build at
or below cu124) -- a cu128 build would very likely have the same
driver-compatibility problem. Confirm with `python -c "import torch;
print(torch.cuda.is_available())"` after setup, before assuming GPU works. A
CPU fallback is a real possibility, in which case `--device cpu` should be
used and the runtime budgeted accordingly (this 0.9B generative model will be
considerably slower on CPU than the embedding/clustering systems in this
project).

## License / gating

Apache License 2.0. No gating documented on the model card as of this
export; no `HF_TOKEN` requirement is currently coded into `run_g4a_moss.py`.

## Checkpoint

`OpenMOSS-Team/MOSS-Transcribe-Diarize` -- resolved from the HF Hub
identifier at run time; no specific commit/revision has been pinned or
verified in this export.
