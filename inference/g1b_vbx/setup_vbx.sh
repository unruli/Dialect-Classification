#!/usr/bin/env bash
# One-time setup for G1-B (BUT SpeechFIT VBx). Shares the G1-A conda env
# (diar_g1-equivalent: torch+cu118, no version floor from VBx's own
# requirements). Clones the upstream VBx repo (Apache-2.0, bundled ONNX
# x-vector extractor weights) into a target dir the caller chooses -- keep
# this under a cache/scratch location (e.g. tmpfs), not the git repo.
#
# Usage: setup_vbx.sh <target_dir> <python_bin>
set -euo pipefail
TARGET_DIR="$1"
PYTHON_BIN="${2:-python}"

if [ -d "$TARGET_DIR/.git" ]; then
  echo "VBx already cloned at $TARGET_DIR, skipping clone."
else
  git clone --depth 1 https://github.com/BUTSpeechFIT/VBx.git "$TARGET_DIR"
fi

# VBx's own requirements.txt lists the now-broken PyPI 'sklearn' shim; install
# scikit-learn directly and skip VBx's own dependency resolution.
"$PYTHON_BIN" -m pip install numpy scipy scikit-learn numexpr fastcluster h5py \
    onnxruntime soundfile kaldi_io tabulate intervaltree
"$PYTHON_BIN" -m pip install -e "$TARGET_DIR" --no-deps

echo "VBx installed at $TARGET_DIR (bundled ONNX x-vector extractor: VBx/models/ResNet101_16kHz/nnet/final.onnx)"
