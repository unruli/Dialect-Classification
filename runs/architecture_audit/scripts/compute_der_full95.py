#!/usr/bin/env python3
"""Compute DER + JER for every completed 95-recording model run against
the manifest's human reference RTTM + UEM, same method as the 5-recording
pilot's compute_der.py: pyannote.metrics, overlap retained, DER at collar=0.0
(primary/strict) and collar=0.25 (conventional), plus JER.

As of the 2026-09-03 reorg, every model's full-95 output lives at
final/<model_dir>/rttm/<dataset>/<recid>.rttm (one consistent layout,
model name in the directory name -- see final/README.md), so this scores
whichever of MODELS actually has a full-95 directory present rather than
a fixed list of 3.

Writes two CSVs: recording-level (95 x n_models rows) and model-level
(n_models rows, fixed pipeline order G1-A/G1-B/G2-A/G2-B/G3-A -- NOT a
ranking, per instructions).
"""
import csv
import os
import statistics
import sys

from pyannote.core import Timeline, Segment
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

BASE = "/home/kelechi/Dialect-Classification/diar_smoke/final"
REF_ROOT = os.path.join(BASE, "reference_raw")  # tar-extracted, preserves absolute paths
MANIFEST = os.path.join(BASE, "final_manifest.csv")
OUT_RECORDING_CSV = os.path.join(BASE, "DER_JER_recording_level_full95.csv")
OUT_MODEL_CSV = os.path.join(BASE, "DER_JER_model_level_full95.csv")

MODELS = [
    ("g1a_full95", "G1-A NeMo MarbleNet+TitaNet+spectral"),
    ("g1b_full95", "G1-B MarbleNet VAD + VBx"),
    ("g2a_full95", "G2-A pyannote community-1"),
    ("g2b_full95", "G2-B NeMo diar_msdd_telephonic"),
    ("g3a_full95", "G3-A nvidia/diar_streaming_sortformer_4spk-v2.1"),
]
MODELS = [(d, label) for d, label in MODELS if os.path.isdir(os.path.join(BASE, d, "rttm"))]


def load_single_annotation(rttm_path):
    d = load_rttm(rttm_path)
    if not d:
        raise ValueError(f"no annotation loaded from {rttm_path}")
    return next(iter(d.values()))


def load_uem(uem_path):
    timeline = Timeline()
    with open(uem_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            start, end = float(parts[2]), float(parts[3])
            timeline.add(Segment(start, end))
    return timeline.support()


def main():
    with open(MANIFEST, newline="") as f:
        manifest_rows = list(csv.DictReader(f))
    print(f"Manifest rows: {len(manifest_rows)}")

    rows = []
    errors = []
    for i, mrow in enumerate(manifest_rows, 1):
        corpus, recid = mrow["dataset"], mrow["recording_id"]
        tag = f"{corpus}__{recid}"

        ref_rttm = REF_ROOT + mrow["reference_rttm_path"]
        ref_uem = REF_ROOT + mrow["uem_path"]
        if not os.path.isfile(ref_rttm) or not os.path.isfile(ref_uem):
            errors.append(f"{tag}: missing reference ({ref_rttm} / {ref_uem})")
            continue

        try:
            reference = load_single_annotation(ref_rttm)
            uem = load_uem(ref_uem)
        except Exception as e:
            errors.append(f"{tag}: failed to load reference: {e}")
            continue
        ref_speakers = len(reference.labels())

        for model_dir, model_label in MODELS:
            hyp_rttm = os.path.join(BASE, model_dir, "rttm", corpus, f"{recid}.rttm")
            if not os.path.isfile(hyp_rttm):
                errors.append(f"{tag}/{model_dir}: missing hypothesis {hyp_rttm}")
                continue
            try:
                hypothesis = load_single_annotation(hyp_rttm)
            except Exception as e:
                errors.append(f"{tag}/{model_key}: failed to load hypothesis: {e}")
                continue
            hyp_speakers = len(hypothesis.labels())

            row = {
                "corpus": corpus, "recording_id": recid, "model": model_label,
                "ref_speakers": ref_speakers, "hyp_speakers": hyp_speakers,
            }
            try:
                for collar in (0.0, 0.25):
                    der_metric = DiarizationErrorRate(collar=collar, skip_overlap=False)
                    details = der_metric(reference, hypothesis, uem=uem, detailed=True)
                    suffix = f"collar{collar}"
                    row[f"DER_{suffix}"] = round(details["diarization error rate"], 4)
                    row[f"miss_{suffix}_sec"] = round(details["missed detection"], 2)
                    row[f"falsealarm_{suffix}_sec"] = round(details["false alarm"], 2)
                    row[f"confusion_{suffix}_sec"] = round(details["confusion"], 2)
                    row[f"total_ref_{suffix}_sec"] = round(details["total"], 2)

                jer_metric = JaccardErrorRate()
                jer_details = jer_metric(reference, hypothesis, uem=uem, detailed=True)
                row["JER"] = round(jer_details["jaccard error rate"], 4)
            except Exception as e:
                errors.append(f"{tag}/{model_key}: scoring failed: {e}")
                continue

            rows.append(row)

        if i % 20 == 0 or i == len(manifest_rows):
            print(f"  scored {i}/{len(manifest_rows)} recordings ({len(rows)} model-rows so far)")

    if errors:
        print(f"\n{len(errors)} error(s) during scoring:", file=sys.stderr)
        for e in errors[:20]:
            print(f"  - {e}", file=sys.stderr)
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more", file=sys.stderr)

    if not rows:
        print("No rows computed.", file=sys.stderr)
        sys.exit(1)

    fieldnames = list(rows[0].keys())
    with open(OUT_RECORDING_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {OUT_RECORDING_CSV}")

    numeric_fields = [
        "DER_collar0.0", "DER_collar0.25", "JER",
        "miss_collar0.0_sec", "falsealarm_collar0.0_sec", "confusion_collar0.0_sec",
    ]
    model_rows = []
    for model_dir, model_label in MODELS:
        matching = [r for r in rows if r["model"] == model_label]
        if not matching:
            continue
        agg = {"model": model_label, "n_recordings": len(matching)}
        for field in numeric_fields:
            values = [r[field] for r in matching]
            agg[f"mean_{field}"] = round(statistics.mean(values), 4)
            agg[f"median_{field}"] = round(statistics.median(values), 4)
            agg[f"min_{field}"] = round(min(values), 4)
            agg[f"max_{field}"] = round(max(values), 4)
        agg["speaker_count_exact_match_n"] = sum(
            1 for r in matching if r["ref_speakers"] == r["hyp_speakers"]
        )
        model_rows.append(agg)

    model_fieldnames = list(model_rows[0].keys())
    with open(OUT_MODEL_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=model_fieldnames)
        writer.writeheader()
        writer.writerows(model_rows)
    print(f"Wrote {len(model_rows)} rows to {OUT_MODEL_CSV} "
          f"(fixed pipeline order {'/'.join(label.split()[0] for _, label in MODELS)} "
          f"-- not a ranking)")


if __name__ == "__main__":
    main()
