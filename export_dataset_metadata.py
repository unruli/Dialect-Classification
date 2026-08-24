#!/usr/bin/env python3
"""Export a small, path-free manifest for the final evaluation selection.

This exports study metadata only. It intentionally excludes audio, transcripts,
turn text, RTTM/UEM content, absolute paths, and credentials.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = REPO_ROOT / "data" / "inference_ready" / "manifest.csv"
DEFAULT_OUTPUT = REPO_ROOT / "dataset_metadata"

PORTABLE_FIELDS = (
    "dataset",
    "subset",
    "recording_id",
    "split",
    "primary_language",
    "domain",
    "audio_duration_sec",
    "scored_duration_sec",
    "sample_rate",
    "channels",
    "num_speakers",
    "num_segments",
    "speech_duration_sec",
    "overlap_duration_sec",
    "overlap_pct",
    "median_turn_sec",
    "short_turn_pct",
    "speaker_imbalance",
    "code_switch_turns",
    "code_switch_turn_pct",
    "mixed_language_turns",
    "same_speaker_language_switches",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def final_selection(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row["dataset"] != "afrispeech_dialog" or row["domain"] == "medical"
    ]


def write_csv(path: Path, fieldnames: tuple[str, ...] | list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    with args.source.open(newline="", encoding="utf-8-sig") as stream:
        rows = final_selection(list(csv.DictReader(stream)))
    rows.sort(key=lambda row: (row["dataset"], row["recording_id"]))

    portable = [{field: row.get(field, "") for field in PORTABLE_FIELDS} for row in rows]
    write_csv(args.output / "final_evaluation_manifest.csv", PORTABLE_FIELDS, portable)

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["dataset"]].append(row)

    summary_fields = [
        "dataset",
        "recordings",
        "audio_hours",
        "scored_hours",
        "speaker_count_distribution",
        "segments",
        "speech_hours",
        "overlap_minutes",
    ]
    summaries = []
    selections = {}
    for dataset in sorted(grouped):
        selected = grouped[dataset]
        speaker_counts = Counter(int(row["num_speakers"]) for row in selected)
        summaries.append(
            {
                "dataset": dataset,
                "recordings": len(selected),
                "audio_hours": f"{sum(float(row['audio_duration_sec']) for row in selected) / 3600:.4f}",
                "scored_hours": f"{sum(float(row['scored_duration_sec']) for row in selected) / 3600:.4f}",
                "speaker_count_distribution": ";".join(
                    f"{count}spk={number}" for count, number in sorted(speaker_counts.items())
                ),
                "segments": sum(int(row["num_segments"]) for row in selected),
                "speech_hours": f"{sum(float(row['speech_duration_sec']) for row in selected) / 3600:.4f}",
                "overlap_minutes": f"{sum(float(row['overlap_duration_sec']) for row in selected) / 60:.4f}",
            }
        )
        selections[dataset] = [row["recording_id"] for row in selected]

    write_csv(args.output / "dataset_summary.csv", summary_fields, summaries)
    (args.output / "recording_selection.json").write_text(
        json.dumps(selections, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Exported {len(rows)} final-selection rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
