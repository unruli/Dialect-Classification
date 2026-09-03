# G3-B environment: BUTSpeechFIT DiaPer (10-attractor, non-AMI-fine-tuned)

**SMOKE-VALIDATED on levi-compute (RTX 4090, driver 535.309.01) 2026-09-03.**
All four fixed 90-second domain smokes pass with valid normalized RTTM. See
`SMOKE_TEST_RESULTS.md` and `smoke_evidence/` for the actual outputs.

**Full-recording status: FAILED on the 24GB RTX 4090, this is why it's moving
to a larger GPU.** The model has no chunking/streaming in its inference
codepath -- full self-attention over the entire frame sequence -- and OOM'd
on 3 of the 4 pilot full recordings plus the mandatory longest-file gate
(AMI EN2002c, 49.54min, tried to allocate 52.66 GiB). This is a genuine
memory-scaling limit, not a bug: attempted allocation sizes observed range
27-53 GiB depending on recording length. **Before running the 95-file batch
on the A100, re-run at minimum the EN2002c longest-file gate there and
confirm it completes** -- an 80GB A100 has headroom for the largest observed
ask (53 GiB) plus other allocations; a 40GB A100 does not and should not be
used for this system's full/longest-file passes. If EN2002c still OOMs even
on an 80GB card, per `MODEL_SELECTION_AND_INFERENCE.md`'s own rule, report
the failure and stop -- do not introduce a chunk-stitcher workaround.

## Setup

Requires **Python 3.7** (this model's codebase predates 3.8-only syntax used
elsewhere in this project), separate from every other environment here:

```bash
conda create -n diar_g3b python=3.7 -y
conda activate diar_g3b

python -m pip install torch==1.10.0+cu113 torchaudio==0.10.0+cu113 \
  -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install -r requirements-diar_g3b.txt
```

`requirements-diar_g3b.txt` is the exact `pip freeze` of the working
levi-compute environment (includes `transformers==4.21.0`, `tokenizers==0.12.1`
-- the last transformers release with a prebuilt cp37 `tokenizers` wheel, so
no Rust/`maturin` toolchain is needed).

## Code

This directory's `diaper/` is the subset of
[BUTSpeechFIT/DiaPer](https://github.com/BUTSpeechFIT/DiaPer) actually
needed for single-file inference (`infer_single_file.py`, `infer.py`,
`train.py` -- only for its `_convert` helper that `infer.py` imports --
`process_data.py`, `backend/`, `common_utils/`). Pulled file-by-file via
`raw.githubusercontent.com`, not a full `git clone` (the upstream `models/`
tree carries many other checkpoint variants totaling several GB that this
project doesn't use).

## Checkpoints

`BUTSpeechFIT/DiaPer`'s 10-attractor, SC+LibriSpeech-2spk-adapted checkpoints
(epochs 91-100, no `checkpoint_90.tar` exists upstream -- the yaml's
`epochs: 90-100` still means "average all of 91..100"). Download:

```bash
mkdir -p models/10attractors/SC_LibriSpeech_2spk_adapted1-10/models
cd models/10attractors/SC_LibriSpeech_2spk_adapted1-10/models
for i in $(seq 91 100); do
  curl -sO "https://raw.githubusercontent.com/BUTSpeechFIT/DiaPer/main/models/10attractors/SC_LibriSpeech_2spk_adapted1-10/models/checkpoint_${i}.tar"
done
```

10 files, ~17MB each (~170MB total). Both endpoints verified reachable
(HTTP 200) 2026-09-03. Then **edit `examples/infer_16k_10attractors.yaml`'s
`models_path`** to the absolute path of the `models/` dir you just created
(currently a `__SET_ME__` placeholder). `rttms_dir` in the yaml does NOT need
editing -- `run_g3b_diaper.sh` always overrides it via `--rttms-dir`.

## Required patch: `transformers` Perceiver cross-attention

The DiaPer repo's own README points at a fork,
`pip install git+https://github.com/fnlandini/transformers`, for a
cross-attention softmax-normalization fix in
`modeling_perceiver.py`. **That fork no longer installs on a fresh
environment** -- its HEAD has drifted forward with upstream and now needs
`maturin`/Rust and Python>=3.9 to build `tokenizers`, incompatible with this
model's Python 3.7 requirement.

Fix used here: install plain `transformers==4.21.0` from PyPI (already
pinned above), then overwrite the installed
`transformers/models/perceiver/modeling_perceiver.py` with this directory's
`patches/modeling_perceiver.py.patched` (verified: the fork's actual change
vs. a real upstream release is a small, self-contained diff limited to that
one file):

```bash
cp patches/modeling_perceiver.py.patched \
  "$(python -c 'import transformers, os; print(os.path.dirname(transformers.__file__))')/models/perceiver/modeling_perceiver.py"
```

Do this AFTER `pip install -r requirements-diar_g3b.txt` (which will place
the unpatched file first).

## Known flag gotcha

`infer_single_file.py --gpu` is **number of GPUs to claim**, not a device
index. `--gpu 0` silently means CPU while checkpoint-averaging still loads
weights onto CUDA regardless, causing a cross-device RuntimeError. Always
use `--gpu 1` for a single-GPU host. `run_g3b_diaper.sh` already does this.

## Running

```bash
bash run_g3b_diaper.sh <wav_path> <uri> <path/to/this/diaper/repo/root> <out_dir>
```

Output RTTM lands at `<out_dir>/<uri>.g3b_diaper.raw.rttm` (native,
un-anonymized speaker labels -- normalize/anonymize downstream the same way
every other system in this project does, via `common/rttm_tools.py`'s
`parse_and_normalize`).

## License / gating

BUTSpeechFIT/DiaPer is openly available on GitHub, no gating. No `HF_TOKEN`
requirement.

## Checkpoint identity

Config: `10attractors/SC_LibriSpeech_2spk_adapted1-10`, `epochs: 90-100`
(checkpoints 91-100 averaged, per `examples/infer_16k_10attractors.yaml`).
Not fine-tuned on AMI, unlike G3-A's Sortformer checkpoint.
