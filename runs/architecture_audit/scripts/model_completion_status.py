#!/usr/bin/env python3
"""Master cross-model completion + consistency tracker for the diarization
architecture audit. Run this BEFORE starting a new model's pilot/batch (to
confirm everything already claimed "done" is actually consistent) and AFTER
a model finishes (to record and validate its results) -- this is the gate
this project's own G1-A bug should have caught earlier: a model is not
"complete" just because a script exited 0, only once its RTTM outputs are
independently re-validated here.

For each of the 95 frozen recordings x each known model, checks:
  - does a normalized RTTM exist at that model's known output location?
  - does it parse as valid 10-field RTTM (common/rttm_tools rules)?
  - does its max end time stay within source_duration + 0.5s (catches the
    exact class of bug found in G1-A's original run)?
  - if a reference RTTM+UEM exists for that recording, DER/JER (collar=0,
    overlap retained) -- used only as a consistency signal here (e.g. a
    wildly implausible mean flags something to investigate), never as a
    ranking between models.

Writes:
  diar_smoke/final/MODEL_COMPLETION_STATUS.csv   (recording-level, one row
    per recording, one status+DER column pair per model)
  diar_smoke/final/MODEL_COMPLETION_SUMMARY.json (per-model counts + any
    flagged inconsistencies)

Exits non-zero if any implemented model shows an inconsistency (missing
expected output, malformed RTTM, or a duration-bound violation), so it can
be used as a pass/fail gate in a shell pipeline.
"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rttm_tools  # noqa: E402

BASE = "/home/kelechi/Dialect-Classification/diar_smoke"
MANIFEST = f"{BASE}/final/final_manifest.csv"
REF_ROOT = f"{BASE}/final/reference_raw"
OUT_CSV = f"{BASE}/final/MODEL_COMPLETION_STATUS.csv"
OUT_SUMMARY = f"{BASE}/final/MODEL_COMPLETION_SUMMARY.json"


def full95_path(model_dir):
    # As of the 2026-09-03 reorg, every model's full-95 output lives at
    # final/<model_dir>/rttm/<dataset>/<recid>.rttm -- one consistent
    # layout, model name in the directory name, matching what collaborators
    # pick up directly (see final/README.md).
    def _resolve(dataset, recid):
        return os.path.join(BASE, "final", model_dir, "rttm", dataset, f"{recid}.rttm")
    return _resolve


# Each entry: (display name, path-resolver(dataset, recid) -> path or None,
# "implemented" flag -- an unimplemented model is reported as not_started
# for every recording rather than treated as an inconsistency).
MODELS = [
    ("G1-A", full95_path("g1a_full95"), True),
    ("G1-B", full95_path("g1b_full95"), True),
    ("G2-A", full95_path("g2a_full95"), True),
    ("G3-A", full95_path("g3a_full95"), True),
    ("G2-B", full95_path("g2b_full95"), True),
    ("G3-B", full95_path("g3b_full95"), False),   # FAILED its mandatory longest-file OOM gate
                                  # 2026-08-31 -- see smoke/g3b_pilot/EN2002c_oomgate.stderr
                                  # (tried to allocate 52.66 GiB for one attention matrix on a
                                  # 24GB GPU). Not eligible for any batch; per
                                  # MODEL_SELECTION_AND_INFERENCE.md, do not chunk-stitch around this.
]


def load_uem(uem_path):
    from pyannote.core import Timeline, Segment
    timeline = Timeline()
    with open(uem_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            timeline.add(Segment(float(parts[2]), float(parts[3])))
    return timeline.support()


def score_der(ref_rttm, ref_uem, hyp_rttm):
    """Best-effort DER (collar=0, overlap retained) for the consistency
    check only. Returns None if pyannote.metrics isn't importable in
    whichever interpreter runs this (it's fine -- status/validity checks
    below don't depend on it)."""
    try:
        from pyannote.database.util import load_rttm
        from pyannote.metrics.diarization import DiarizationErrorRate
        ref = next(iter(load_rttm(ref_rttm).values()))
        hyp = next(iter(load_rttm(hyp_rttm).values()))
        uem = load_uem(ref_uem)
        metric = DiarizationErrorRate(collar=0.0, skip_overlap=False)
        return round(metric(ref, hyp, uem=uem), 4)
    except Exception:
        return None


def main():
    with open(MANIFEST, newline="") as f:
        manifest_rows = list(csv.DictReader(f))

    rows = []
    per_model_counts = {name: {"success": 0, "missing": 0, "malformed": 0, "duration_violation": 0, "not_started": 0}
                         for name, _, _ in MODELS}
    inconsistencies = []

    for mrow in manifest_rows:
        dataset, recid = mrow["dataset"], mrow["recording_id"]
        dur = float(mrow["audio_duration_sec"])
        ref_rttm = REF_ROOT + mrow["reference_rttm_path"] if mrow.get("reference_rttm_path") else None
        ref_uem = REF_ROOT + mrow["uem_path"] if mrow.get("uem_path") else None
        has_ref = bool(ref_rttm and ref_uem and os.path.isfile(ref_rttm) and os.path.isfile(ref_uem))

        row = {"dataset": dataset, "recording_id": recid, "audio_duration_sec": dur}

        for name, resolver, implemented in MODELS:
            path = resolver(dataset, recid)
            if not implemented and (path is None or not os.path.isfile(path)):
                status, der = "not_started", None
                per_model_counts[name]["not_started"] += 1
            elif not path or not os.path.isfile(path):
                status, der = "MISSING", None
                per_model_counts[name]["missing"] += 1
                inconsistencies.append(f"{name} {dataset}/{recid}: expected output missing at {path}")
            else:
                try:
                    with open(path) as f:
                        lines = f.readlines()
                    segments = rttm_tools.parse_and_normalize(lines, uri=recid)
                    rttm_tools.validate_against_source_duration(segments, dur)
                    status = "success"
                    per_model_counts[name]["success"] += 1
                    der = score_der(ref_rttm, ref_uem, path) if has_ref else None
                except rttm_tools.RTTMValidationError as e:
                    if "exceeds" in str(e):
                        status = "DURATION_VIOLATION"
                        per_model_counts[name]["duration_violation"] += 1
                    else:
                        status = "MALFORMED"
                        per_model_counts[name]["malformed"] += 1
                    der = None
                    inconsistencies.append(f"{name} {dataset}/{recid}: {e}")

            row[f"{name}_status"] = status
            row[f"{name}_DER"] = der

        rows.append(row)

    fieldnames = list(rows[0].keys())
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "n_recordings": len(rows),
        "per_model_counts": per_model_counts,
        "n_inconsistencies": len(inconsistencies),
        "inconsistencies": inconsistencies,
    }
    with open(OUT_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {OUT_CSV} ({len(rows)} recordings)")
    print(f"Wrote {OUT_SUMMARY}")
    print()
    for name, counts in per_model_counts.items():
        print(f"  {name}: {counts}")
    if inconsistencies:
        print(f"\n{len(inconsistencies)} INCONSISTENCY(IES) FOUND:")
        for inc in inconsistencies[:20]:
            print(f"  - {inc}")
        sys.exit(1)
    else:
        print("\nNo inconsistencies found among implemented models.")


if __name__ == "__main__":
    main()
