#!/usr/bin/env python3
"""RQ4 frozen scorer: UEM-aware DER/JER, zero collar, overlap included.

Deliberately a fresh implementation rather than a copy of the architecture
audit's scorer, because the protocol requires the RQ4 numbers to be produced by
a scorer whose settings are frozen and asserted rather than inherited. The
settings are pinned here and recorded in every output:

    collar        = 0.0     (no forgiveness band -- strict)
    skip_overlap  = False   (overlapped speech is scored, not discarded)
    UEM           = required; hypothesis and reference are both cropped to it

Cropping to the UEM is what makes scores comparable across systems: without it,
a system is rewarded or punished for regions the annotation never covered.

`--assert-frozen` re-checks the pinned settings at runtime and refuses to score
if anything drifted, so a silently edited constant cannot quietly change
published numbers.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import sys

FROZEN = {
    "collar_primary": 0.0,
    "collar_secondary": 0.25,
    "skip_overlap": False,
    "uem_required": True,
    "scorer_version": "rq4-frozen-1",
}


def load_uem(path):
    from pyannote.core import Segment, Timeline
    tl = Timeline()
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) >= 4:
                tl.add(Segment(float(p[2]), float(p[3])))
    return tl.support()


def load_annotation(path):
    from pyannote.database.util import load_rttm
    d = load_rttm(path)
    if not d:
        raise ValueError(f"no annotation parsed from {path}")
    return next(iter(d.values()))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def score_one(ref_rttm, ref_uem, hyp_rttm):
    from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate
    ref, hyp, uem = load_annotation(ref_rttm), load_annotation(hyp_rttm), load_uem(ref_uem)
    row = {
        "ref_speakers": len(ref.labels()),
        "hyp_speakers": len(hyp.labels()),
    }
    for collar, tag in ((FROZEN["collar_primary"], "collar0.0"),
                        (FROZEN["collar_secondary"], "collar0.25")):
        m = DiarizationErrorRate(collar=collar, skip_overlap=FROZEN["skip_overlap"])
        d = m(ref, hyp, uem=uem, detailed=True)
        row[f"DER_{tag}"] = round(d["diarization error rate"], 6)
        row[f"miss_{tag}_sec"] = round(d["missed detection"], 3)
        row[f"falsealarm_{tag}_sec"] = round(d["false alarm"], 3)
        row[f"confusion_{tag}_sec"] = round(d["confusion"], 3)
        row[f"total_ref_{tag}_sec"] = round(d["total"], 3)
    j = JaccardErrorRate()(ref, hyp, uem=uem, detailed=True)
    row["JER"] = round(j["jaccard error rate"], 6)
    return row


def main():
    ap = argparse.ArgumentParser(description="RQ4 frozen diarization scorer")
    ap.add_argument("--manifest", required=True,
                    help="CSV with dataset, recording_id, reference_rttm_path, uem_path")
    ap.add_argument("--hyp-dir", required=True,
                    help="<hyp-dir>/<dataset>/<recording_id>.rttm")
    ap.add_argument("--ref-root", default="", help="prefix for reference paths")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--assert-frozen", action="store_true",
                    help="fail if the pinned scorer settings have drifted")
    args = ap.parse_args()

    if args.assert_frozen:
        expected = {"collar_primary": 0.0, "collar_secondary": 0.25,
                    "skip_overlap": False, "uem_required": True,
                    "scorer_version": "rq4-frozen-1"}
        if FROZEN != expected:
            print(f"FROZEN SETTINGS DRIFTED\n  expected {expected}\n  found    {FROZEN}",
                  file=sys.stderr)
            return 2

    rows, errors = [], []
    for r in csv.DictReader(open(args.manifest)):
        ds, rid = r["dataset"], r["recording_id"]
        ref = args.ref_root + r["reference_rttm_path"]
        uem = args.ref_root + r["uem_path"]
        hyp = os.path.join(args.hyp_dir, ds, f"{rid}.rttm")
        for p, what in ((ref, "reference"), (uem, "uem"), (hyp, "hypothesis")):
            if not os.path.isfile(p):
                errors.append(f"{ds}/{rid}: missing {what} {p}")
                break
        else:
            try:
                row = {"dataset": ds, "recording_id": rid}
                row.update(score_one(ref, uem, hyp))
                row["hyp_sha256"] = sha256(hyp)[:16]
                rows.append(row)
            except Exception as e:
                errors.append(f"{ds}/{rid}: {type(e).__name__}: {e}")

    for e in errors[:20]:
        print(f"ERROR {e}", file=sys.stderr)
    if not rows:
        print("no recordings scored", file=sys.stderr)
        return 1

    rec_csv = f"{args.out_prefix}_per_recording.csv"
    with open(rec_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def agg(field):
        v = [r[field] for r in rows]
        return {"mean": round(statistics.mean(v), 6),
                "median": round(statistics.median(v), 6),
                "min": round(min(v), 6), "max": round(max(v), 6)}

    summary = {
        "frozen_settings": FROZEN,
        "n_scored": len(rows),
        "n_errors": len(errors),
        "errors": errors,
        "DER_collar0.0": agg("DER_collar0.0"),
        "DER_collar0.25": agg("DER_collar0.25"),
        "JER": agg("JER"),
        "speaker_count_exact_match": sum(
            1 for r in rows if r["ref_speakers"] == r["hyp_speakers"]),
    }
    json.dump(summary, open(f"{args.out_prefix}_summary.json", "w"), indent=2)

    print(f"scored {len(rows)} recordings ({len(errors)} error(s))")
    print(f"  DER collar=0.0  mean {summary['DER_collar0.0']['mean']:.4f}")
    print(f"  DER collar=0.25 mean {summary['DER_collar0.25']['mean']:.4f}")
    print(f"  JER             mean {summary['JER']['mean']:.4f}")
    print(f"  wrote {rec_csv} and {args.out_prefix}_summary.json")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
