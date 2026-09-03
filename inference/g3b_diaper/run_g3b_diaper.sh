#!/usr/bin/env bash
# G3-B: BUT SpeechFIT DiaPer, 10-attractor non-AMI-fine-tuned checkpoint
# (models/10attractors/SC_LibriSpeech_2spk_adapted1-10, epochs 91-100
# averaged, matching examples/infer_16k_10attractors.yaml exactly).
#
# Runs in the diar_g3b conda env (Python 3.7, torch 1.10.0+cu113 -- CUDA
# 11.3 needs driver >=465.19.01, comfortably below this host's 535.309.01).
#
# CAUTION: per MODEL_SELECTION_AND_INFERENCE.md, this is pilot/longest-file-
# gate only. Do not wrap this in a 95-file batch until the longest recording
# (AMI EN2002c, 49.54min) has been confirmed to process without OOM.
#
# Usage: run_g3b_diaper.sh <wav_path> <uri> <diaper_repo_dir> <out_dir>
set -uo pipefail

WAV="$1"
URI="$2"
DIAPER_REPO="$3"
OUT_DIR="$4"

WAV_DIR=$(dirname "$WAV")
WAV_NAME=$(basename "$WAV" .wav)
RTTMS_DIR="$OUT_DIR/_work_g3b/$URI"
mkdir -p "$RTTMS_DIR" "$OUT_DIR"

DIAR_G3B_PYTHON="${DIAR_G3B_PYTHON:-$HOME/miniconda3/envs/diar_g3b/bin/python}"

t0=$(date +%s.%N)
( cd "$DIAPER_REPO" && PYTHONPATH="$DIAPER_REPO/diaper:$DIAPER_REPO/diaper/common_utils:${PYTHONPATH:-}" \
  "$DIAR_G3B_PYTHON" diaper/infer_single_file.py \
    -c examples/infer_16k_10attractors.yaml \
    --wav-dir "$WAV_DIR" \
    --wav-name "$WAV_NAME" \
    --rttms-dir "$RTTMS_DIR" \
    --gpu 1 \
) 1>&2
rc=$?
t1=$(date +%s.%N)
elapsed=$(python3 -c "print(round($t1-$t0,2))")

if [ $rc -ne 0 ]; then
  echo "{\"ok\": false, \"error\": \"infer_single_file.py exited rc=$rc\", \"elapsed_sec\": $elapsed}"
  exit 1
fi

# DiaPer nests output under a deep hyperparameter-encoding path
# (epochsX-Y/timeshuffle.../spk_qty.../detection_thr.../median.../
# subsampling.../rttms/), not directly under --rttms-dir -- no depth limit.
raw_rttm=$(find "$RTTMS_DIR" -iname "*${WAV_NAME}*.rttm" | head -1)
if [ -z "$raw_rttm" ] || [ ! -f "$raw_rttm" ]; then
  echo "{\"ok\": false, \"error\": \"no RTTM found under $RTTMS_DIR\", \"elapsed_sec\": $elapsed}"
  exit 1
fi

raw_out="$OUT_DIR/${URI}.g3b_diaper.raw.rttm"
cp "$raw_rttm" "$raw_out"

echo "{\"ok\": true, \"raw_rttm\": \"$raw_out\", \"elapsed_sec\": $elapsed, \"checkpoint\": \"10attractors/SC_LibriSpeech_2spk_adapted1-10 epochs91-100avg\"}"
