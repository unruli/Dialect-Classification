"""Manifest join/validation/selection logic shared by inference/run_model.py.

Joins a collaborator-local, path-bearing inference manifest to this repo's
path-free frozen selection manifest (dataset_metadata/final_evaluation_manifest.csv)
on (dataset, recording_id). No machine-specific paths are hardcoded here --
both manifest paths are supplied by the caller.
"""
import csv
import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Recording:
    dataset: str
    recording_id: str
    audio_path: str
    audio_duration_sec: float
    num_speakers_ref: Optional[int]  # reference count -- NEVER pass to a model automatically


class ManifestError(ValueError):
    """Raised on any manifest join/validation failure. Callers should treat
    this as fatal and stop, per the required behavior in inference/README.md."""


def load_selection_manifest(selection_manifest_path: str) -> List[dict]:
    with open(selection_manifest_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ManifestError(f"selection manifest is empty: {selection_manifest_path}")
    seen = set()
    dupes = set()
    for row in rows:
        key = (row["dataset"], row["recording_id"])
        (dupes if key in seen else seen).add(key)
    if dupes:
        raise ManifestError(
            f"duplicate (dataset, recording_id) key(s) in selection manifest "
            f"{selection_manifest_path}: {sorted(dupes)}"
        )
    return rows


def load_path_manifest(path_manifest_path: str) -> dict:
    """Returns {(dataset, recording_id): row} for the collaborator's local,
    path-bearing manifest. Raises on duplicate (dataset, recording_id) keys."""
    index = {}
    with open(path_manifest_path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["dataset"], row["recording_id"])
            if key in index:
                raise ManifestError(
                    f"duplicate (dataset, recording_id) in path manifest: {key}"
                )
            index[key] = row
    return index


def join_manifests(
    path_manifest_path: str,
    selection_manifest_path: str,
    expect_full_count: bool = True,
    recording_ids: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> List[Recording]:
    """Join the frozen selection manifest to the collaborator's local
    path-bearing manifest. Raises ManifestError on any missing ID, duplicate
    ID, or (when expect_full_count) a matched count other than 95 -- this
    function does not return partial results silently.

    `recording_ids`, if given, filters the selection manifest down to just
    those recording_id values BEFORE the completeness check, so a deliberate
    single-recording request (e.g. --recording-id for a smoke test or a
    retry) is not treated as an incomplete full run. `expect_full_count` is
    ignored whenever `recording_ids` or `limit` is set -- the caller asked
    for a subset on purpose.
    """
    selection_rows = load_selection_manifest(selection_manifest_path)
    path_index = load_path_manifest(path_manifest_path)

    if recording_ids:
        wanted = set(recording_ids)
        selection_rows = [r for r in selection_rows if r["recording_id"] in wanted]
        found_ids = {r["recording_id"] for r in selection_rows}
        not_in_selection = wanted - found_ids
        if not_in_selection:
            raise ManifestError(
                f"--recording-id value(s) not present in the selection manifest: {sorted(not_in_selection)}"
            )
        expect_full_count = False

    if limit is not None:
        expect_full_count = False

    recordings = []
    missing = []
    unreadable = []
    for row in selection_rows:
        key = (row["dataset"], row["recording_id"])
        if key not in path_index:
            missing.append(key)
            continue
        path_row = path_index[key]
        audio_path = path_row.get("audio_path") or path_row.get("source_audio_path")
        if not audio_path:
            raise ManifestError(
                f"path manifest row for {key} has no 'audio_path' or "
                f"'source_audio_path' column"
            )
        if not os.path.isfile(audio_path):
            unreadable.append((key, audio_path))
            continue
        num_speakers_ref = row.get("num_speakers")
        recordings.append(
            Recording(
                dataset=row["dataset"],
                recording_id=row["recording_id"],
                audio_path=audio_path,
                audio_duration_sec=float(row["audio_duration_sec"]),
                num_speakers_ref=int(num_speakers_ref) if num_speakers_ref else None,
            )
        )

    if missing:
        raise ManifestError(
            f"{len(missing)} selection-manifest ID(s) not found in the local "
            f"path manifest (dataset, recording_id): {missing[:10]}"
            + (" ... (truncated)" if len(missing) > 10 else "")
        )

    if unreadable:
        raise ManifestError(
            f"{len(unreadable)} audio_path value(s) do not exist on disk: "
            f"{unreadable[:5]}" + (" ... (truncated)" if len(unreadable) > 5 else "")
        )

    if expect_full_count and len(recordings) != 95:
        raise ManifestError(
            f"expected exactly 95 matched recordings for a full run, got "
            f"{len(recordings)}"
        )

    if limit is not None:
        recordings = recordings[:limit]

    return recordings
