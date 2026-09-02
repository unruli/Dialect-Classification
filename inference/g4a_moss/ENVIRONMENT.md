# G4-A environment: OpenMOSS-Team/MOSS-Transcribe-Diarize (0.9B)

**SMOKE-VALIDATED 2026-09-02.** All four fixed 90-second domain recordings
completed on a 20-GB A100 MIG slice with valid normalized RTTM, no truncation,
and strict artifact/timestamp QC. Complete-recording pilots remain pending.

## Setup

Requires **Python 3.12+**, a separate environment from G1-A/G1-B/G2-A/G3-A:

```bash
conda create -n <env-name> python=3.12 -y
conda activate <env-name>

python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.8.0 torchaudio==2.8.0
python -m pip install transformers==5.6.0

git clone https://github.com/OpenMOSS/MOSS-Transcribe-Diarize.git
cd MOSS-Transcribe-Diarize
git checkout 61bc29cd4120be7b5d3b761b64cd5dff57263642
python -m pip install -e .
```

That installs `moss_transcribe_diarize` (the package `run_g4a_moss.py` imports:
`parse_transcript`, and from `moss_transcribe_diarize.inference_utils`:
`build_transcription_messages`, `generate_transcription`, `resolve_device`)
plus the pinned torch/transformers stack above. The repo also ships
`mtd-subtitle` (batch CLI) and
`mtd-subtitle-web` (web UI), neither of which this project uses --
`run_g4a_moss.py` calls the Python helpers directly.

## Validated CURC configuration and remaining gate

The passing CURC environment used Python 3.12, PyTorch 2.8.0+cu128,
Transformers 5.6.0, and official package revision
`61bc29cd4120be7b5d3b761b64cd5dff57263642` on driver 570.124.06. Peak
allocated GPU memory was 1,985.4 MiB. This does not establish that a complete
49.54-minute recording fits or finishes within the allocation; run the four
complete pilots and longest-recording gate before the 95-file batch.

## License / gating

Apache License 2.0. No gating documented on the model card as of this
export; no `HF_TOKEN` requirement is currently coded into `run_g4a_moss.py`.

## Checkpoint

`OpenMOSS-Team/MOSS-Transcribe-Diarize`, resolved checkpoint revision
`704aa4a9c304e8520be88901e0d1960158ef5b15` in the passing smoke run.
