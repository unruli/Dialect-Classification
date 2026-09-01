#!/usr/bin/env bash
# Full 95-recording batch run over the frozen final manifest, using the exact
# same validated per-file driver as the pilot (run_pilot_file.sh), writing to
# a separate final/ output tree so the pilot's own outputs stay untouched.
#
# Resilient: a single recording's failure is logged and does NOT abort the
# batch (a ~16h unattended run should not die on one bad file).
set -uo pipefail

BASE=/home/kelechi/Dialect-Classification/diar_smoke
MANIFEST="$BASE/final/final_manifest.csv"
export OUT_RTTM_DIR="$BASE/final/rttm"
export OUT_LOGS_DIR="$BASE/final/logs"
PROGRESS_CSV="$BASE/final/progress.csv"
SUMMARY_JSON="$BASE/final/progress_summary.json"

mkdir -p "$OUT_RTTM_DIR" "$OUT_LOGS_DIR"

TOTAL=$(($(wc -l < "$MANIFEST") - 1))
echo "dataset,recording_id,duration_sec,index,total,status,started_at,finished_at,elapsed_sec" > "$PROGRESS_CSV"

i=0
n_pass=0
n_fail=0
batch_start=$(date +%s)

# CSV columns (from build_final_manifest.py / manifest.csv):
# dataset,subset,recording_id,split,primary_language,domain,source_audio_path,audio_path,...,audio_duration_sec,...
python3 -c "
import csv
with open('$MANIFEST', newline='') as f:
    for row in csv.DictReader(f):
        print(f\"{row['dataset']}|{row['recording_id']}|{row['audio_path']}|{row['audio_duration_sec']}\")
" > /tmp/claude-1011/-home-kelechi/11d36598-847b-43b8-b2d1-7824ed233436/scratchpad/final_batch_rows.txt

while IFS='|' read -r dataset recid audio_path dur_sec; do
  i=$((i+1))
  echo "=========================================================="
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$i/$TOTAL] $dataset/$recid (${dur_sec}s)"
  echo "=========================================================="
  f_start=$(date +%s)
  t0_iso=$(date -Iseconds)

  bash "$BASE/scripts/run_pilot_file.sh" "$dataset" "$recid" "$audio_path" "$dur_sec"
  rc=$?

  f_end=$(date +%s)
  t1_iso=$(date -Iseconds)
  elapsed=$((f_end - f_start))

  uid_tag="${dataset}__${recid}"
  log_json="$OUT_LOGS_DIR/${uid_tag}.json"
  status="FAIL"
  if [ $rc -eq 0 ] && [ -f "$log_json" ]; then
    all_pass=$(python3 -c "import json; print(json.load(open('$log_json')).get('all_models_pass'))" 2>/dev/null)
    [ "$all_pass" = "True" ] && status="PASS"
  fi

  if [ "$status" = "PASS" ]; then
    n_pass=$((n_pass+1))
  else
    n_fail=$((n_fail+1))
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] *** FAILED: $dataset/$recid (rc=$rc) -- continuing to next file ***"
  fi

  echo "$dataset,$recid,$dur_sec,$i,$TOTAL,$status,$t0_iso,$t1_iso,$elapsed" >> "$PROGRESS_CSV"

  # rolling summary, updated after every file, for the 2-hourly check-ins
  python3 -c "
import json, time
summary = {
    'completed': $i,
    'total': $TOTAL,
    'pass': $n_pass,
    'fail': $n_fail,
    'last_recording': '$dataset/$recid',
    'last_status': '$status',
    'last_update': '$t1_iso',
    'batch_elapsed_sec': $(($(date +%s) - batch_start)),
}
json.dump(summary, open('$SUMMARY_JSON', 'w'), indent=2)
"
done < /tmp/claude-1011/-home-kelechi/11d36598-847b-43b8-b2d1-7824ed233436/scratchpad/final_batch_rows.txt

batch_end=$(date +%s)
echo "=========================================================="
echo "BATCH COMPLETE: $n_pass/$TOTAL passed, $n_fail failed. Total wall time: $((batch_end - batch_start))s"
echo "=========================================================="
