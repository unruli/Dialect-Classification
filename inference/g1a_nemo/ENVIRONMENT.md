# G1-A environment: NeMo MarbleNet VAD + TitaNet-Large + NME-SC clustering

Also used, unmodified, by **G1-B** (VBx, same env, no extra torch requirement)
and **G3-A** (Sortformer, ships inside nemo_toolkit -- confirmed working in
this same environment via a live 90-second smoke test).

## Why this exact pin

Driver-class GPUs capped at CUDA 12.2 (`nvidia-smi` reports "CUDA Version: 12.2")
cannot run torch built for newer CUDA majors. `nemo_toolkit` 2.7.x requires
`torch>=2.6.0`; the newest torch release with a published cu118 wheel (which
this driver class can use) is 2.7.1. `nemo_toolkit[asr]` is used, not the
full `nemo_toolkit[all]`, to avoid pulling NLP/TTS/multimodal dependencies
this project never needs.

## Setup

```bash
conda create -n <env-name> python=3.10 -y
conda activate <env-name>

pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
pip install "nemo_toolkit[asr]==2.7.3"

# G1-B only:
bash ../g1b_vbx/setup_vbx.sh /path/to/vbx_checkout $(which python)
```

Verify GPU is actually usable (not just torch importable) before trusting a
smoke test:

```bash
python -c "import torch; print(torch.cuda.is_available())"  # must print True
```

## Full package list

See `environment.txt` in this directory (`pip freeze` output). Key pins:
torch 2.7.1+cu118, torchaudio 2.7.1+cu118, nemo-toolkit 2.7.3.

## Checkpoint cache

Set `NEMO_CACHE_DIR` (and, for G3-A, also `HF_HOME`) to a writable cache
directory before running -- e.g. tmpfs, not persistent disk if disk space is
constrained. `--cache-dir` on `run_model.py` sets this automatically.

## Checkpoints used (revisions at last verified run)

- `vad_marblenet`: revision `10477085f32c378938ef41e65dc2e1b3`
- `titanet_large`: revision `11ba0924fdf87c049e339adbf6899d48`
- `nvidia/diar_streaming_sortformer_4spk-v2.1` (G3-A): resolved from the HF
  Hub identifier at run time; NVIDIA Open Model License Agreement. **This
  checkpoint's training data includes the AMI Meeting Corpus** -- AMI
  recordings in the evaluation manifest are not an independent test of G3-A.
