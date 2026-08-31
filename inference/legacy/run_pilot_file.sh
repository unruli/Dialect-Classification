#!/usr/bin/env bash
# Three-model diarization pilot driver for ONE staged recording.
# Usage: run_pilot_file.sh <corpus> <recording_id> <remote_wav_path> <audio_duration_sec>
#
# Stages the recording from levi-testing into /dev/shm, runs:
#   G1-1 NeMo MarbleNet VAD + TitaNet-Large + spectral clustering  (diar_g1, GPU)
#   G1-2 MarbleNet VAD + BUT VBx                                   (diar_g1, CPU/ONNX by upstream design)
#   G2   pyannote/speaker-diarization-community-1                  (diar_g2, forced CPU)
# parses+anonymizes each RTTM, writes a per-file JSON log, and deletes the
# staged audio ONLY if all three (inference + parse) succeeded.
set -uo pipefail

CORPUS="$1"
RECID="$2"
REMOTE_WAV="$3"
DUR_SEC="$4"

REMOTE_HOST="<redacted-for-export -- was the project's source-data host>"
BASE=/home/kelechi/Dialect-Classification/diar_smoke
# OUT_RTTM_DIR / OUT_LOGS_DIR let a batch wrapper redirect output (e.g. to
# final/ for the full run) without touching the pilot's own validated output.
PILOT_RTTM="${OUT_RTTM_DIR:-$BASE/pilot/rttm}"
PILOT_LOGS="${OUT_LOGS_DIR:-$BASE/pilot/logs}"
STAGE_DIR=/dev/shm/dialect-smoke/pilot_staging
NEMO_CACHE=/dev/shm/dialect-smoke/nemo_cache
HF_HOME_DIR=/dev/shm/dialect-smoke/hf_cache
VBX_REPO=/dev/shm/dialect-smoke/vbx_repo
DIAR_G1=/home/kelechi/miniconda3/envs/diar_g1/bin
DIAR_G2=/home/kelechi/miniconda3/envs/diar_g2/bin

mkdir -p "$PILOT_RTTM" "$PILOT_LOGS" "$STAGE_DIR"

UID_TAG="${CORPUS}__${RECID}"
LOCAL_WAV="$STAGE_DIR/${UID_TAG}.wav"
LOG_JSON="$PILOT_LOGS/${UID_TAG}.json"

log() { echo "[$(date +%H:%M:%S)] $*" >&2; }

fail_json() {
  python3 - "$@" <<'PYEOF'
import json, sys
d = json.loads(sys.argv[1])
print(json.dumps(d))
PYEOF
}

# ---- 1. Stage (one file at a time, from levi-testing, into RAM only) ----
log "Staging $CORPUS/$RECID from levi-testing ..."
t_stage0=$(date +%s.%N)
scp -o BatchMode=yes -q "$REMOTE_HOST:$REMOTE_WAV" "$LOCAL_WAV"
SCP_RC=$?
t_stage1=$(date +%s.%N)
STAGE_SEC=$(python3 -c "print(round($t_stage1-$t_stage0,2))")

if [ $SCP_RC -ne 0 ] || [ ! -s "$LOCAL_WAV" ]; then
  log "STAGE FAILED for $UID_TAG (scp rc=$SCP_RC)"
  python3 -c "
import json
print(json.dumps({'corpus':'$CORPUS','recording_id':'$RECID','stage':'stage_copy','status':'FAIL','error':'scp failed or empty file rc=$SCP_RC'}, indent=2))
" > "$LOG_JSON"
  exit 1
fi
log "Staged in ${STAGE_SEC}s -> $LOCAL_WAV"

# ---- 2. G1-1: NeMo MarbleNet VAD + TitaNet-Large + spectral clustering (GPU) ----
log "Running G1-1 NeMo (GPU) ..."
G1_NEMO_RESULT_JSON="$PILOT_LOGS/${UID_TAG}.g1_nemo.result.json"
rm -f "$G1_NEMO_RESULT_JSON"
# NeMo's logger writes INFO lines to stdout (not just stderr), so stdout alone
# is not reliably pure JSON -- the script also writes --result-json directly.
"$DIAR_G1/python" "$BASE/scripts/run_g1_nemo.py" \
  --wav "$LOCAL_WAV" \
  --out-dir "$PILOT_RTTM" \
  --nemo-cache "$NEMO_CACHE" \
  --device cuda \
  --result-json "$G1_NEMO_RESULT_JSON" \
  > "$PILOT_LOGS/${UID_TAG}.g1_nemo.stdout" 2> "$PILOT_LOGS/${UID_TAG}.g1_nemo.stderr"
G1_NEMO_RC=$?
log "G1-1 rc=$G1_NEMO_RC (see ${UID_TAG}.g1_nemo.result.json)"

G1_NEMO_PARSE="FAIL"
if [ $G1_NEMO_RC -eq 0 ] && [ -f "$G1_NEMO_RESULT_JSON" ]; then
  RAW_RTTM=$(python3 -c "import json; print(json.load(open('$G1_NEMO_RESULT_JSON'))['raw_rttm'])" 2>/dev/null)
  if [ -n "$RAW_RTTM" ] && [ -f "$RAW_RTTM" ]; then
    mv "$RAW_RTTM" "$PILOT_RTTM/${UID_TAG}.g1_nemo_titanet_spectral.raw.rttm"
    PARSE_OUT=$("$DIAR_G1/python" "$BASE/scripts/parse_rttm.py" \
      "$PILOT_RTTM/${UID_TAG}.g1_nemo_titanet_spectral.raw.rttm" \
      "$PILOT_RTTM/${UID_TAG}.g1_nemo_titanet_spectral.rttm" \
      "$UID_TAG" 2>&1)
    echo "$PARSE_OUT" | grep -q "^PARSE_OK" && G1_NEMO_PARSE="OK"
  fi
  # also grab the marblenet VAD .lab produced alongside, for VBx reuse
  VAD_LAB="$PILOT_RTTM/$(basename "$LOCAL_WAV" .wav).marblenet_vad.lab"
fi

# ---- 3. G1-2: MarbleNet VAD + VBx (CPU / ONNX, upstream default) ----
log "Running G1-2 VBx (CPU/ONNX) ..."
G1_VBX_PARSE="FAIL"
VBX_ELAPSED="null"
if [ "$G1_NEMO_PARSE" = "OK" ] && [ -f "$VAD_LAB" ]; then
  mkdir -p "$STAGE_DIR/vbx_${UID_TAG}"/{lists,xvectors,segments,vad_lab,out}
  echo "$(basename "$LOCAL_WAV" .wav)" > "$STAGE_DIR/vbx_${UID_TAG}/list.txt"
  cp "$VAD_LAB" "$STAGE_DIR/vbx_${UID_TAG}/vad_lab/$(basename "$LOCAL_WAV" .wav).lab"

  t0=$(date +%s.%N)
  ( cd "$VBX_REPO" && "$DIAR_G1/python" VBx/predict.py \
      --in-file-list "$STAGE_DIR/vbx_${UID_TAG}/list.txt" \
      --in-lab-dir "$STAGE_DIR/vbx_${UID_TAG}/vad_lab" \
      --in-wav-dir "$STAGE_DIR" \
      --out-ark-fn "$STAGE_DIR/vbx_${UID_TAG}/xvectors/${RECID}.ark" \
      --out-seg-fn "$STAGE_DIR/vbx_${UID_TAG}/segments/${RECID}" \
      --weights VBx/models/ResNet101_16kHz/nnet/final.onnx \
      --backend onnx ) > "$PILOT_LOGS/${UID_TAG}.g1_vbx_extract.stderr" 2>&1
  EXTRACT_RC=$?

  if [ $EXTRACT_RC -eq 0 ]; then
    ( cd "$VBX_REPO" && "$DIAR_G1/python" VBx/vbhmm.py \
        --init AHC+VB \
        --out-rttm-dir "$STAGE_DIR/vbx_${UID_TAG}/out" \
        --xvec-ark-file "$STAGE_DIR/vbx_${UID_TAG}/xvectors/${RECID}.ark" \
        --segments-file "$STAGE_DIR/vbx_${UID_TAG}/segments/${RECID}" \
        --xvec-transform VBx/models/ResNet101_16kHz/transform.h5 \
        --plda-file VBx/models/ResNet101_16kHz/plda \
        --threshold -0.015 --lda-dim 128 --Fa 0.3 --Fb 17 --loopP 0.99 \
    ) >> "$PILOT_LOGS/${UID_TAG}.g1_vbx_extract.stderr" 2>&1
    VBHMM_RC=$?
    t1=$(date +%s.%N)
    VBX_ELAPSED=$(python3 -c "print(round($t1-$t0,2))")

    # vbhmm.py names its output by the utterance id embedded in the x-vector
    # ark (taken from --in-file-list content, i.e. the staged wav's basename
    # = UID_TAG), not by our --out-ark-fn/--out-seg-fn filenames. Find it
    # rather than assume a name.
    VBX_OUT_RTTM=$(find "$STAGE_DIR/vbx_${UID_TAG}/out" -maxdepth 1 -name "*.rttm" | head -1)
    if [ $VBHMM_RC -eq 0 ] && [ -n "$VBX_OUT_RTTM" ] && [ -f "$VBX_OUT_RTTM" ]; then
      cp "$VBX_OUT_RTTM" "$PILOT_RTTM/${UID_TAG}.g1_vbx.raw.rttm"
      PARSE_OUT=$("$DIAR_G1/python" "$BASE/scripts/parse_rttm.py" \
        "$PILOT_RTTM/${UID_TAG}.g1_vbx.raw.rttm" \
        "$PILOT_RTTM/${UID_TAG}.g1_vbx.rttm" \
        "$UID_TAG" 2>&1)
      echo "$PARSE_OUT" | grep -q "^PARSE_OK" && G1_VBX_PARSE="OK"
    fi
  fi
  # keep VAD lab as a documented output (records the VAD reuse for VBx)
  cp "$VAD_LAB" "$PILOT_RTTM/${UID_TAG}.marblenet_vad.lab" 2>/dev/null
  rm -rf "$STAGE_DIR/vbx_${UID_TAG}"
fi
log "G1-2 parse=$G1_VBX_PARSE elapsed=${VBX_ELAPSED}s"

# ---- 4. G2: pyannote/speaker-diarization-community-1 (forced CPU) ----
log "Running G2 community-1 (forced CPU) ..."
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null)
t0=$(date +%s.%N)
CUDA_VISIBLE_DEVICES="" G2_OUT=$(CUDA_VISIBLE_DEVICES="" "$DIAR_G2/python" "$BASE/scripts/run_g2_pyannote.py" \
  --wav "$LOCAL_WAV" \
  --checkpoint pyannote/speaker-diarization-community-1 \
  --out-rttm "$PILOT_RTTM/${UID_TAG}.g2_community1.raw.rttm" \
  --hf-home "$HF_HOME_DIR" 2>"$PILOT_LOGS/${UID_TAG}.g2.stderr")
G2_RC=$?
t1=$(date +%s.%N)
G2_WALL=$(python3 -c "print(round($t1-$t0,2))")
log "G2 rc=$G2_RC wall=${G2_WALL}s"

G2_PARSE="FAIL"
if [ $G2_RC -eq 0 ] && [ -f "$PILOT_RTTM/${UID_TAG}.g2_community1.raw.rttm" ]; then
  PARSE_OUT=$("$DIAR_G2/python" "$BASE/scripts/parse_rttm.py" \
    "$PILOT_RTTM/${UID_TAG}.g2_community1.raw.rttm" \
    "$PILOT_RTTM/${UID_TAG}.g2_community1.rttm" \
    "$UID_TAG" 2>&1)
  echo "$PARSE_OUT" | grep -q "^PARSE_OK" && G2_PARSE="OK"
fi

# ---- 5. Assemble per-file log ----
python3 - "$CORPUS" "$RECID" "$DUR_SEC" "$STAGE_SEC" \
  "$G1_NEMO_RC" "$G1_NEMO_PARSE" "$G1_NEMO_RESULT_JSON" \
  "$G1_VBX_PARSE" "$VBX_ELAPSED" \
  "$G2_RC" "$G2_PARSE" "$G2_WALL" "$G2_OUT" \
  > "$LOG_JSON" <<'PYEOF'
import json, sys
(corpus, recid, dur_sec, stage_sec,
 g1n_rc, g1n_parse, g1n_result_path,
 g1v_parse, g1v_elapsed,
 g2_rc, g2_parse, g2_wall, g2_out) = sys.argv[1:14]

def safe_json(s):
    try:
        return json.loads(s)
    except Exception:
        return None

def load_json_file(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None

g1n = load_json_file(g1n_result_path) or {}
g2 = safe_json(g2_out) or {}

dur = float(dur_sec)

def rtf(elapsed):
    try:
        return round(float(elapsed) / dur, 4)
    except Exception:
        return None

record = {
    "corpus": corpus,
    "recording_id": recid,
    "audio_duration_sec": dur,
    "stage_copy_sec": float(stage_sec),
    "models": {
        "g1_nemo_titanet_spectral": {
            "device": "cuda",
            "rc": int(g1n_rc),
            "elapsed_sec": g1n.get("elapsed_sec"),
            "rtf": rtf(g1n.get("elapsed_sec")) if g1n.get("elapsed_sec") is not None else None,
            "parse": g1n_parse,
            "pass": (int(g1n_rc) == 0 and g1n_parse == "OK"),
        },
        "g1_vbx_marblenet_vad": {
            "device": "cpu (onnxruntime, upstream default)",
            "elapsed_sec": (float(g1v_elapsed) if g1v_elapsed not in ("null", "") else None),
            "rtf": rtf(g1v_elapsed) if g1v_elapsed not in ("null", "") else None,
            "parse": g1v_parse,
            "pass": (g1v_parse == "OK"),
        },
        "g2_pyannote_community1": {
            "device": "cpu (forced, CUDA_VISIBLE_DEVICES='')",
            "rc": int(g2_rc),
            "load_elapsed_sec": g2.get("load_elapsed_sec"),
            "infer_elapsed_sec": g2.get("infer_elapsed_sec"),
            "wall_elapsed_sec": float(g2_wall),
            "rtf": rtf(g2_wall),
            "cuda_actually_usable": g2.get("cuda_actually_usable"),
            "parse": g2_parse,
            "pass": (int(g2_rc) == 0 and g2_parse == "OK"),
        },
    },
}
record["all_models_pass"] = all(m["pass"] for m in record["models"].values())
print(json.dumps(record, indent=2))
PYEOF

ALL_PASS=$(python3 -c "import json; print(json.load(open('$LOG_JSON'))['all_models_pass'])")
log "all_models_pass=$ALL_PASS"

# ---- 6. Cleanup staged audio ONLY if all three validated ----
if [ "$ALL_PASS" = "True" ]; then
  rm -f "$LOCAL_WAV"
  log "Validated all 3 outputs; removed staged audio $LOCAL_WAV"
else
  log "NOT all outputs validated; leaving $LOCAL_WAV in place for inspection"
fi

cat "$LOG_JSON"
