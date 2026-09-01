#!/usr/bin/env bash
# G1-B: MarbleNet VAD (reused from a prior G1-A run) + BUT SpeechFIT VBx
# (AHC+VB-HMM over x-vectors, ONNX Runtime CPU backend -- VBx's own upstream
# default; VBx has no VAD of its own, hence the MarbleNet reuse).
#
# Extracted, path-parameterized version of the VBx invocation proven in
# diar_smoke/scripts/run_pilot_file.sh. No machine-specific paths: every path
# is a required argument.
#
# Usage: run_g1b_vbx.sh <wav_path> <vad_lab_path> <uri> <vbx_repo_dir> <work_dir> <out_rttm_dir>
set -uo pipefail

WAV="$1"
VAD_LAB="$2"
URI="$3"
VBX_REPO="$4"
WORK_DIR="$5"
OUT_RTTM_DIR="$6"

if [ ! -f "$VAD_LAB" ]; then
  echo "ERROR: VAD lab file not found: $VAD_LAB (run G1-A first and pass its .marblenet_vad.lab output)" >&2
  exit 1
fi

mkdir -p "$WORK_DIR"/{lists,xvectors,segments,vad_lab,out} "$OUT_RTTM_DIR"
WAV_DIR=$(dirname "$WAV")
WAV_BASENAME=$(basename "$WAV" .wav)

echo "$WAV_BASENAME" > "$WORK_DIR/lists/list.txt"
cp "$VAD_LAB" "$WORK_DIR/vad_lab/${WAV_BASENAME}.lab"

t0=$(date +%s.%N)
( cd "$VBX_REPO" && python VBx/predict.py \
    --in-file-list "$WORK_DIR/lists/list.txt" \
    --in-lab-dir "$WORK_DIR/vad_lab" \
    --in-wav-dir "$WAV_DIR" \
    --out-ark-fn "$WORK_DIR/xvectors/${URI}.ark" \
    --out-seg-fn "$WORK_DIR/segments/${URI}" \
    --weights VBx/models/ResNet101_16kHz/nnet/final.onnx \
    --backend onnx ) 1>&2
EXTRACT_RC=$?
if [ $EXTRACT_RC -ne 0 ]; then
  echo "ERROR: VBx x-vector extraction failed (rc=$EXTRACT_RC)" >&2
  exit 1
fi

( cd "$VBX_REPO" && python VBx/vbhmm.py \
    --init AHC+VB \
    --out-rttm-dir "$WORK_DIR/out" \
    --xvec-ark-file "$WORK_DIR/xvectors/${URI}.ark" \
    --segments-file "$WORK_DIR/segments/${URI}" \
    --xvec-transform VBx/models/ResNet101_16kHz/transform.h5 \
    --plda-file VBx/models/ResNet101_16kHz/plda \
    --threshold -0.015 --lda-dim 128 --Fa 0.3 --Fb 17 --loopP 0.99 \
) 1>&2
VBHMM_RC=$?
t1=$(date +%s.%N)
elapsed=$(python3 -c "print(round($t1-$t0,2))")
if [ $VBHMM_RC -ne 0 ]; then
  echo "ERROR: VBx vbhmm.py failed (rc=$VBHMM_RC)" >&2
  exit 1
fi

# vbhmm.py names its output by the utterance id embedded in the x-vector ark
# (taken from --in-file-list content, i.e. WAV_BASENAME), not by our
# --out-ark-fn/--out-seg-fn filenames -- find it rather than assume a name.
VBX_OUT_RTTM=$(find "$WORK_DIR/out" -maxdepth 1 -name "*.rttm" | head -1)
if [ -z "$VBX_OUT_RTTM" ] || [ ! -f "$VBX_OUT_RTTM" ]; then
  echo "ERROR: no .rttm produced under $WORK_DIR/out" >&2
  exit 1
fi

RAW_OUT="$OUT_RTTM_DIR/${URI}.g1b_vbx.raw.rttm"
cp "$VBX_OUT_RTTM" "$RAW_OUT"

echo "{\"ok\": true, \"raw_rttm_path\": \"$RAW_OUT\", \"elapsed_sec\": $elapsed, \"device\": \"cpu (onnxruntime)\"}"
