#!/usr/bin/env python3
"""Prepare the four diarization evaluation datasets for common inference.

Outputs are written below ``data/inference_ready`` by default. Source media and
annotations are never modified. The preparation is deterministic and
idempotent: valid derived WAV files are reused on subsequent runs.

Datasets:
  * AMI official Mix-Headset test split
  * AfriSpeech-Dialog medical, timestamp-usable v1_47 rows (default)
  * Playlogue official participant-disjoint test split
  * Bangor Miami English-primary, non-Maria subset
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = REPO_ROOT / "data" / "datasets"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "inference_ready"

BANGOR_ENGLISH_PRIMARY = (
    "herring01", "herring06", "herring07", "herring08", "herring09",
    "herring10", "herring13", "herring15", "herring16", "herring17",
    "sastre03", "sastre04", "sastre06", "sastre07", "sastre08",
    "sastre09", "sastre10", "sastre11", "sastre12", "sastre13",
    "zeledon02", "zeledon03", "zeledon04", "zeledon06", "zeledon08",
    "zeledon09", "zeledon11", "zeledon13",
)

CHAT_TIMESTAMP_RE = re.compile(r"\x15(\d+)_(\d+)\x15")
CHAT_LANGUAGE_RE = re.compile(r"\[-\s*(eng|spa)\s*\]")
CHAT_SWITCH_TOKEN_RE = re.compile(r"(?<!\S)\S+@s(?:[:][A-Za-z-]+)?")
AFRI_TIMESTAMP_RE = re.compile(r"^\s*(\d{2}:\d{2}:\d{2})\s*$")
AFRI_SPEAKER_RE = re.compile(r"^\s*\[\s*(Speaker\s+\d+)\s*\]\s*:\s*(.*)$", re.I)


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    speaker: str
    text: str = ""
    language: str = ""
    code_switched: bool = False
    mixed_tokens: int = 0

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class ManifestRow:
    dataset: str
    subset: str
    recording_id: str
    split: str
    primary_language: str
    domain: str
    source_audio_path: str
    audio_path: str
    reference_rttm_path: str
    uem_path: str
    turn_metadata_path: str
    audio_duration_sec: float
    scored_duration_sec: float
    sample_rate: int
    channels: int
    num_speakers: int
    num_segments: int
    speech_duration_sec: float
    overlap_duration_sec: float
    overlap_pct: float
    median_turn_sec: float
    short_turn_pct: float
    speaker_imbalance: float
    code_switch_turns: int | str = ""
    code_switch_turn_pct: float | str = ""
    mixed_language_turns: int | str = ""
    same_speaker_language_switches: int | str = ""
    reference_origin: str = ""
    notes: str = ""


def require_commands() -> None:
    for command in ("ffmpeg", "ffprobe"):
        if shutil.which(command) is None:
            raise RuntimeError(f"required command is not installed: {command}")


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def probe_audio(path: Path) -> dict[str, float | int]:
    command = [
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate,channels:format=duration",
        "-of", "json", str(path),
    ]
    result = json.loads(subprocess.check_output(command, text=True))
    stream = result["streams"][0]
    return {
        "duration": float(result["format"]["duration"]),
        "sample_rate": int(stream["sample_rate"]),
        "channels": int(stream["channels"]),
    }


def ensure_pcm16k_mono(
    source: Path,
    destination: Path,
    start_sec: float = 0.0,
    end_sec: float | None = None,
) -> Path:
    source_info = probe_audio(source)
    expected_duration = (
        float(source_info["duration"]) - start_sec
        if end_sec is None
        else end_sec - start_sec
    )
    if expected_duration <= 0:
        raise ValueError(f"invalid trim interval for {source}: {start_sec}--{end_sec}")

    if destination.exists() and destination.stat().st_size > 10_000:
        try:
            current = probe_audio(destination)
            if (
                current["sample_rate"] == 16_000
                and current["channels"] == 1
                and abs(float(current["duration"]) - expected_duration) <= 0.12
            ):
                return destination
        except Exception:
            pass

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + ".part.wav")
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source)]
    if start_sec:
        command.extend(["-ss", f"{start_sec:.6f}"])
    if end_sec is not None:
        command.extend(["-t", f"{end_sec - start_sec:.6f}"])
    command.extend(["-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(temporary)])
    subprocess.run(command, check=True)
    converted = probe_audio(temporary)
    if converted["sample_rate"] != 16_000 or converted["channels"] != 1:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"audio conversion failed validation: {source}")
    if abs(float(converted["duration"]) - expected_duration) > 0.12:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"duration changed unexpectedly for {source}: expected {expected_duration}, "
            f"got {converted['duration']}"
        )
    os.replace(temporary, destination)
    return destination


def convert_many(tasks: Sequence[tuple[Path, Path, float, float | None]], workers: int) -> None:
    if not tasks:
        return
    print(f"Preparing {len(tasks)} audio files with {workers} worker(s) ...", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(ensure_pcm16k_mono, source, destination, start, end): destination
            for source, destination, start, end in tasks
        }
        completed = 0
        for future in as_completed(futures):
            destination = futures[future]
            future.result()
            completed += 1
            if completed % 10 == 0 or completed == len(tasks):
                print(f"  audio {completed}/{len(tasks)}: {destination.name}", flush=True)


def merge_same_speaker_overlaps(segments: Sequence[Segment]) -> list[Segment]:
    grouped: dict[str, list[Segment]] = defaultdict(list)
    for segment in segments:
        if segment.end > segment.start:
            grouped[segment.speaker].append(segment)
    merged: list[Segment] = []
    for speaker, speaker_segments in grouped.items():
        for segment in sorted(speaker_segments, key=lambda item: (item.start, item.end)):
            if merged and merged[-1].speaker == speaker and segment.start <= merged[-1].end:
                previous = merged.pop()
                merged.append(Segment(previous.start, max(previous.end, segment.end), speaker))
            else:
                merged.append(segment)
        # Keep subsequent speakers independent from the last item of this group.
        if merged:
            merged[-1] = merged[-1]
    return sorted(merged, key=lambda item: (item.start, item.end, item.speaker))


def write_rttm(path: Path, recording_id: str, segments: Sequence[Segment]) -> None:
    lines = [
        f"SPEAKER {recording_id} 1 {segment.start:.3f} {segment.duration:.3f} "
        f"<NA> <NA> {segment.speaker.replace(' ', '_')} <NA> <NA>"
        for segment in segments
        if segment.duration > 0
    ]
    atomic_text(path, "\n".join(lines) + ("\n" if lines else ""))


def write_turns(path: Path, dataset: str, recording_id: str, segments: Sequence[Segment]) -> None:
    lines = []
    for segment in segments:
        item = asdict(segment)
        item.update({"dataset": dataset, "recording_id": recording_id})
        lines.append(json.dumps(item, ensure_ascii=False, sort_keys=True))
    atomic_text(path, "\n".join(lines) + ("\n" if lines else ""))


def union_intervals(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted((max(0.0, a), b) for a, b in intervals if b > a):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def subtract_intervals(
    base: tuple[float, float], blocked: Iterable[tuple[float, float]]
) -> list[tuple[float, float]]:
    parts = [base]
    for block_start, block_end in union_intervals(blocked):
        next_parts = []
        for start, end in parts:
            if block_end <= start or block_start >= end:
                next_parts.append((start, end))
                continue
            if block_start > start:
                next_parts.append((start, min(block_start, end)))
            if block_end < end:
                next_parts.append((max(block_end, start), end))
        parts = next_parts
    return [(start, end) for start, end in parts if end - start > 0.001]


def write_uem(path: Path, recording_id: str, intervals: Sequence[tuple[float, float]]) -> None:
    lines = [f"{recording_id} 1 {start:.3f} {end:.3f}" for start, end in intervals if end > start]
    atomic_text(path, "\n".join(lines) + ("\n" if lines else ""))


def parse_rttm(path: Path) -> list[Segment]:
    segments = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 8 or fields[0] != "SPEAKER":
            raise ValueError(f"invalid RTTM at {path}:{line_number}")
        start = float(fields[3])
        duration = float(fields[4])
        segments.append(Segment(start, start + duration, fields[7]))
    return segments


def parse_uem(path: Path) -> list[tuple[float, float]]:
    intervals = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(f"invalid UEM at {path}:{line_number}")
        intervals.append((float(fields[2]), float(fields[3])))
    return intervals


def segment_statistics(segments: Sequence[Segment]) -> dict[str, float | int]:
    valid = [segment for segment in segments if segment.duration > 0]
    speakers = sorted({segment.speaker for segment in valid})
    durations = [segment.duration for segment in valid]
    speaker_totals = Counter()
    for segment in valid:
        speaker_totals[segment.speaker] += segment.duration

    events: dict[float, list[tuple[str, int]]] = defaultdict(list)
    for segment in valid:
        events[segment.start].append((segment.speaker, 1))
        events[segment.end].append((segment.speaker, -1))
    active = Counter()
    previous = None
    speech = 0.0
    overlap = 0.0
    for timestamp in sorted(events):
        if previous is not None:
            elapsed = timestamp - previous
            active_speakers = sum(count > 0 for count in active.values())
            if active_speakers:
                speech += elapsed
            if active_speakers >= 2:
                overlap += elapsed
        for speaker, change in sorted(events[timestamp], key=lambda item: item[1]):
            active[speaker] += change
        previous = timestamp

    total_speaker_time = sum(speaker_totals.values())
    return {
        "num_speakers": len(speakers),
        "num_segments": len(valid),
        "speech_duration": speech,
        "overlap_duration": overlap,
        "overlap_pct": 100.0 * overlap / speech if speech else 0.0,
        "median_turn": median(durations) if durations else 0.0,
        "short_turn_pct": 100.0 * sum(duration < 1.0 for duration in durations) / len(durations) if durations else 0.0,
        "speaker_imbalance": max(speaker_totals.values()) / total_speaker_time if total_speaker_time else 0.0,
    }


def make_manifest_row(
    *,
    dataset: str,
    subset: str,
    recording_id: str,
    split: str,
    primary_language: str,
    domain: str,
    source_audio: Path,
    audio: Path,
    rttm: Path,
    uem: Path,
    turns: Path | None,
    reference_origin: str,
    notes: str = "",
    code_switch: dict[str, int | float] | None = None,
) -> ManifestRow:
    audio_info = probe_audio(audio)
    segments = parse_rttm(rttm)
    intervals = parse_uem(uem)
    stats = segment_statistics(segments)
    code_switch = code_switch or {}
    return ManifestRow(
        dataset=dataset,
        subset=subset,
        recording_id=recording_id,
        split=split,
        primary_language=primary_language,
        domain=domain,
        source_audio_path=str(source_audio.resolve()),
        audio_path=str(audio.resolve()),
        reference_rttm_path=str(rttm.resolve()),
        uem_path=str(uem.resolve()),
        turn_metadata_path=str(turns.resolve()) if turns else "",
        audio_duration_sec=round(float(audio_info["duration"]), 6),
        scored_duration_sec=round(sum(end - start for start, end in intervals), 6),
        sample_rate=int(audio_info["sample_rate"]),
        channels=int(audio_info["channels"]),
        num_speakers=int(stats["num_speakers"]),
        num_segments=int(stats["num_segments"]),
        speech_duration_sec=round(float(stats["speech_duration"]), 6),
        overlap_duration_sec=round(float(stats["overlap_duration"]), 6),
        overlap_pct=round(float(stats["overlap_pct"]), 4),
        median_turn_sec=round(float(stats["median_turn"]), 6),
        short_turn_pct=round(float(stats["short_turn_pct"]), 4),
        speaker_imbalance=round(float(stats["speaker_imbalance"]), 6),
        code_switch_turns=code_switch.get("turns", ""),
        code_switch_turn_pct=code_switch.get("turn_pct", ""),
        mixed_language_turns=code_switch.get("mixed_turns", ""),
        same_speaker_language_switches=code_switch.get("same_speaker_switches", ""),
        reference_origin=reference_origin,
        notes=notes,
    )


def prepare_ami(output: Path) -> tuple[list[ManifestRow], list[str]]:
    source = DATA_ROOT / "ami"
    meetings = [
        line.strip()
        for line in (source / "AMI-diarization-setup/lists/test.meetings.txt").read_text().splitlines()
        if line.strip()
    ]
    rows = []
    for meeting in meetings:
        audio = source / "amicorpus" / meeting / "audio" / f"{meeting}.Mix-Headset.wav"
        rttm = source / "reference_rttm" / f"{meeting}.rttm"
        uem = source / "uem" / f"{meeting}.uem"
        rows.append(make_manifest_row(
            dataset="ami",
            subset="mix_headset_test",
            recording_id=meeting,
            split="test",
            primary_language="en",
            domain="adult_meeting",
            source_audio=audio,
            audio=audio,
            rttm=rttm,
            uem=uem,
            turns=None,
            reference_origin="AMI-diarization-setup word_and_vocalsounds test reference",
            notes="Official Mix-Headset test recording; source already 16-kHz mono.",
        ))
    return rows, []


def afri_time_seconds(value: str) -> float:
    minutes, seconds, milliseconds = (int(part) for part in value.split(":"))
    return minutes * 60.0 + seconds + milliseconds / 1000.0


def parse_afri_transcript(text: str) -> tuple[list[Segment], list[str]]:
    segments: list[Segment] = []
    warnings: list[str] = []
    start: float | None = None
    speaker: str | None = None
    words: list[str] = []

    for raw_line in text.replace("\r", "").splitlines():
        timestamp_match = AFRI_TIMESTAMP_RE.match(raw_line)
        speaker_match = AFRI_SPEAKER_RE.match(raw_line)
        if timestamp_match:
            timestamp = afri_time_seconds(timestamp_match.group(1))
            if start is not None and speaker is not None:
                if timestamp > start:
                    segments.append(Segment(start, timestamp, speaker.replace(" ", "_"), " ".join(words).strip()))
                else:
                    warnings.append(f"non-positive segment {start:.3f}--{timestamp:.3f} ({speaker})")
                start = None
                speaker = None
                words = []
            start = timestamp
        elif speaker_match:
            speaker = re.sub(r"\s+", " ", speaker_match.group(1).strip()).replace(" ", "_")
            words = [speaker_match.group(2).strip()] if speaker_match.group(2).strip() else []
        elif speaker is not None and raw_line.strip():
            words.append(raw_line.strip())
    return segments, warnings


def prepare_afrispeech(
    output: Path,
    workers: int,
    domain_filter: str,
) -> tuple[list[ManifestRow], list[str]]:
    dataset_root = DATA_ROOT / "afrispeech_dialog"
    source_candidates = (dataset_root / "dir_dataset", dataset_root)
    source = next(
        (candidate for candidate in source_candidates if (candidate / "afrispeech_dialog_v1_47.csv").is_file()),
        None,
    )
    if source is None:
        raise FileNotFoundError(
            "AfriSpeech CSV not found. Expected afrispeech_dialog_v1_47.csv "
            f"under one of: {', '.join(str(path) for path in source_candidates)}"
        )
    csv_path = source / "afrispeech_dialog_v1_47.csv"
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        source_rows = list(csv.DictReader(stream))

    prepared = output / "afrispeech_dialog"
    tasks = []
    parsed: list[tuple[dict[str, str], Path, list[Segment], list[str]]] = []
    warnings = []
    for source_row in source_rows:
        domain = "medical" if source_row["domain"] == "OSCE-Doctor-Patient" else "non_medical"
        if domain_filter == "medical" and domain != "medical":
            continue
        relative = Path(source_row["path"])
        source_audio = source / relative
        if not source_audio.is_file() and relative.parts[0] == "data":
            relative = Path(*relative.parts[1:])
            source_audio = source / relative
        recording_id = source_row["audio_id"]
        segments, parse_warnings = parse_afri_transcript(source_row["transcript"])
        speakers = {segment.speaker for segment in segments}
        if not segments or len(speakers) < 2:
            warnings.append(
                f"AfriSpeech excluded {recording_id}: no usable two-speaker timed reference "
                f"({len(segments)} segments, speakers={sorted(speakers)})"
            )
            continue
        destination = prepared / "audio" / f"{recording_id}.wav"
        tasks.append((source_audio, destination, 0.0, None))
        parsed.append((source_row, source_audio, segments, parse_warnings))

    convert_many(tasks, workers)
    rows = []
    for source_row, source_audio, raw_segments, parse_warnings in parsed:
        recording_id = source_row["audio_id"]
        audio = prepared / "audio" / f"{recording_id}.wav"
        segments = merge_same_speaker_overlaps(raw_segments)
        rttm = prepared / "rttm" / f"{recording_id}.rttm"
        uem = prepared / "uem" / f"{recording_id}.uem"
        turns = prepared / "turns" / f"{recording_id}.jsonl"
        write_rttm(rttm, recording_id, segments)
        write_turns(turns, "afrispeech_dialog", recording_id, raw_segments)
        max_end = max(segment.end for segment in segments)
        write_uem(uem, recording_id, [(0.0, max_end)])
        if parse_warnings:
            warnings.append(f"AfriSpeech {recording_id}: filtered {len(parse_warnings)} non-positive timestamp segment(s)")
        rows.append(make_manifest_row(
            dataset="afrispeech_dialog",
            subset="timestamped_v1_47",
            recording_id=recording_id,
            split="test",
            primary_language="en",
            domain=domain,
            source_audio=source_audio,
            audio=audio,
            rttm=rttm,
            uem=uem,
            turns=turns,
            reference_origin="AfriSpeech-Dialog CSV human timestamped speaker transcript",
            notes=(
                f"accent={source_row['accent']}; country={source_row['country']}; "
                "UEM ends at final valid transcript timestamp."
            ),
        ))
    return rows, warnings


def playlogue_audio_path(recording_id: str) -> Path:
    root = REPO_ROOT / "data" / "audio"
    prefixes = (
        ("ew_42ec_", root / "Clinical-Eng/EllisWeismer/TD/42ec"),
        ("ew_42pc_", root / "Clinical-Eng/EllisWeismer/TD/42pc"),
        ("ew_54ec_", root / "Clinical-Eng/EllisWeismer/TD/54ec"),
        ("cameron_aae_", root / "Eng-AAE/Cameron/AAE"),
        ("cameron_sae_", root / "Eng-AAE/Cameron/SAE"),
        ("gleason_father_", root / "Eng-NA/Gleason/Father"),
        ("gleason_mother_", root / "Eng-NA/Gleason/Mother"),
        ("vh_", root / "Eng-NA/VanHouten/Threes/freeplay"),
    )
    for prefix, directory in prefixes:
        if recording_id.startswith(prefix):
            return directory / f"{recording_id[len(prefix):]}.mp3"
    raise ValueError(f"unrecognized Playlogue recording ID: {recording_id}")


def prepare_playlogue(output: Path, workers: int) -> tuple[list[ManifestRow], list[str]]:
    source = DATA_ROOT / "playlogue" / "playlogue-v1"
    with (source / "metadata/splits.csv").open(newline="", encoding="utf-8-sig") as stream:
        split_rows = list(csv.DictReader(stream))
    with (source / "metadata/clip_timings.csv").open(newline="", encoding="utf-8-sig") as stream:
        timings = {row["id"]: row for row in csv.DictReader(stream)}

    test_ids = sorted(row["id"] for row in split_rows if row["split"].strip() == "test")
    prepared = output / "playlogue"
    tasks = []
    task_metadata = {}
    for recording_id in test_ids:
        source_audio = playlogue_audio_path(recording_id)
        timing = timings[recording_id]
        start = float(timing["start_time"]) / 1000.0
        end = None if float(timing["end_time"]) == -1 else float(timing["end_time"]) / 1000.0
        destination = prepared / "audio" / f"{recording_id}.wav"
        tasks.append((source_audio, destination, start, end))
        task_metadata[recording_id] = (source_audio, destination, start, end)
    convert_many(tasks, workers)

    rows = []
    warnings = []
    for recording_id in test_ids:
        source_audio, audio, start, end = task_metadata[recording_id]
        source_rttm = source / "data/speaker_diarization" / f"{recording_id}.rttm"
        rttm = prepared / "rttm" / f"{recording_id}.rttm"
        rttm.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_rttm, rttm)
        duration = float(probe_audio(audio)["duration"])
        uem = prepared / "uem" / f"{recording_id}.uem"
        write_uem(uem, recording_id, [(0.0, duration)])
        domain = recording_id.split("_", 1)[0]
        rows.append(make_manifest_row(
            dataset="playlogue",
            subset="official_test",
            recording_id=recording_id,
            split="test",
            primary_language="en",
            domain=f"adult_child_{domain}",
            source_audio=source_audio,
            audio=audio,
            rttm=rttm,
            uem=uem,
            turns=None,
            reference_origin="Playlogue v1 official RTTM aligned to curated clip",
            notes=f"Source clip trim: start={start:.3f}s, end={'EOF' if end is None else f'{end:.3f}s'}.",
        ))
    return rows, warnings


def parse_chat(path: Path) -> tuple[list[Segment], list[tuple[float, float]], int, dict[str, int | float]]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    turns: list[tuple[str, str]] = []
    current: list[str] | None = None
    for line in text.splitlines():
        if line.startswith("*") and ":" in line:
            speaker, content = line[1:].split(":", 1)
            current = [speaker.strip(), content.strip()]
            turns.append((current[0], current[1]))
        elif line.startswith("\t") and current is not None:
            speaker, previous = turns[-1]
            turns[-1] = (speaker, previous + " " + line.strip())
        elif line.startswith(("%", "@")):
            current = None

    segments = []
    blocked = []
    untimed_trusted = 0
    code_switch_turns = 0
    mixed_turns = 0
    transcribed_timed = 0
    speaker_language_sequences: dict[str, list[str]] = defaultdict(list)
    for speaker, content in turns:
        timestamps = CHAT_TIMESTAMP_RE.findall(content)
        if not timestamps:
            if speaker != "OSE":
                untimed_trusted += 1
            continue
        start_ms = min(int(start) for start, _ in timestamps)
        end_ms = max(int(end) for _, end in timestamps)
        if end_ms <= start_ms:
            continue
        start = start_ms / 1000.0
        end = end_ms / 1000.0
        if speaker == "OSE":
            blocked.append((start, end))
            continue
        language_match = CHAT_LANGUAGE_RE.search(content)
        language = language_match.group(1) if language_match else "eng"
        mixed_tokens = len(CHAT_SWITCH_TOKEN_RE.findall(content))
        code_switched = language != "eng" or mixed_tokens > 0
        segments.append(Segment(start, end, speaker, content, language, code_switched, mixed_tokens))
        if not re.search(r"(^|\s)www(?:\s|[.!?]|$)", content):
            transcribed_timed += 1
            code_switch_turns += int(code_switched)
            mixed_turns += int(mixed_tokens > 0)
            if mixed_tokens == 0:
                speaker_language_sequences[speaker].append(language)
    same_speaker_switches = sum(
        previous != current
        for sequence in speaker_language_sequences.values()
        for previous, current in zip(sequence, sequence[1:])
    )
    code_switch = {
        "turns": code_switch_turns,
        "turn_pct": round(100.0 * code_switch_turns / transcribed_timed, 4) if transcribed_timed else 0.0,
        "mixed_turns": mixed_turns,
        "same_speaker_switches": same_speaker_switches,
    }
    return segments, blocked, untimed_trusted, code_switch


def prepare_bangor(output: Path, workers: int) -> tuple[list[ManifestRow], list[str]]:
    source = DATA_ROOT / "bangor_miami"
    prepared = output / "bangor_miami_eng"
    transcript_paths = {}
    tasks = []
    for recording_id in BANGOR_ENGLISH_PRIMARY:
        matches = list((source / "transcripts/Miami/eng").glob(f"*/{recording_id}.cha"))
        if len(matches) != 1:
            raise RuntimeError(f"expected one English CHAT transcript for {recording_id}, found {matches}")
        transcript_paths[recording_id] = matches[0]
        source_audio = source / "audio" / f"{recording_id}.wav"
        destination = prepared / "audio" / f"{recording_id}.wav"
        tasks.append((source_audio, destination, 0.0, None))
    convert_many(tasks, workers)

    rows = []
    warnings = []
    for recording_id in BANGOR_ENGLISH_PRIMARY:
        transcript = transcript_paths[recording_id]
        source_audio = source / "audio" / f"{recording_id}.wav"
        audio = prepared / "audio" / f"{recording_id}.wav"
        raw_segments, blocked, untimed, code_switch = parse_chat(transcript)
        segments = merge_same_speaker_overlaps(raw_segments)
        rttm = prepared / "rttm" / f"{recording_id}.rttm"
        uem = prepared / "uem" / f"{recording_id}.uem"
        turns = prepared / "turns" / f"{recording_id}.jsonl"
        write_rttm(rttm, recording_id, segments)
        write_turns(turns, "bangor_miami_eng", recording_id, raw_segments)
        max_end = max(
            [segment.end for segment in raw_segments] + [end for _, end in blocked]
        )
        scored_intervals = subtract_intervals((0.0, max_end), blocked)
        write_uem(uem, recording_id, scored_intervals)
        if untimed:
            warnings.append(f"Bangor {recording_id}: {untimed} trusted speaker tier(s) lack timestamps")
        group = transcript.parent.name
        rows.append(make_manifest_row(
            dataset="bangor_miami_eng",
            subset="english_primary_non_maria",
            recording_id=recording_id,
            split="test",
            primary_language="en",
            domain=f"adult_codeswitch_{group}",
            source_audio=source_audio,
            audio=audio,
            rttm=rttm,
            uem=uem,
            turns=turns,
            reference_origin="Bangor Miami English CHAT speaker tiers and hidden-bullet timestamps",
            notes=(
                f"English-primary transcript; generic OSE intervals masked in UEM; "
                f"untimed trusted tiers={untimed}."
            ),
            code_switch=code_switch,
        ))
    return rows, warnings


def validate_rows(rows: Sequence[ManifestRow]) -> tuple[list[str], dict[str, dict[str, float | int]]]:
    errors = []
    seen = set()
    for row in rows:
        key = (row.dataset, row.recording_id)
        if key in seen:
            errors.append(f"duplicate manifest key: {key}")
        seen.add(key)
        for value, label in (
            (row.audio_path, "audio"),
            (row.reference_rttm_path, "RTTM"),
            (row.uem_path, "UEM"),
        ):
            if not Path(value).is_file():
                errors.append(f"missing {label}: {value}")
        if row.sample_rate != 16_000 or row.channels != 1:
            errors.append(f"non-standard audio: {key} ({row.sample_rate} Hz, {row.channels} channels)")
        segments = parse_rttm(Path(row.reference_rttm_path))
        if any(segment.start < 0 or segment.end <= segment.start for segment in segments):
            errors.append(f"invalid RTTM segment: {key}")
        if segments and max(segment.end for segment in segments) > row.audio_duration_sec + 0.12:
            errors.append(f"RTTM exceeds audio duration: {key}")
        intervals = parse_uem(Path(row.uem_path))
        if any(start < 0 or end <= start or end > row.audio_duration_sec + 0.12 for start, end in intervals):
            errors.append(f"invalid UEM interval: {key}")

    summaries = {}
    for dataset in sorted({row.dataset for row in rows}):
        selected = [row for row in rows if row.dataset == dataset]
        summaries[dataset] = {
            "recordings": len(selected),
            "audio_hours": round(sum(row.audio_duration_sec for row in selected) / 3600.0, 4),
            "scored_hours": round(sum(row.scored_duration_sec for row in selected) / 3600.0, 4),
            "segments": sum(row.num_segments for row in selected),
            "overlap_hours": round(sum(row.overlap_duration_sec for row in selected) / 3600.0, 4),
        }
    return errors, summaries


def write_manifest(path: Path, rows: Sequence[ManifestRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    fieldnames = list(asdict(rows[0]).keys())
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    os.replace(temporary, path)


def write_readiness_report(
    output: Path,
    rows: Sequence[ManifestRow],
    warnings: Sequence[str],
    errors: Sequence[str],
    summaries: dict[str, dict[str, float | int]],
) -> None:
    report = {
        "status": "ready" if not errors else "validation_failed",
        "manifest": str((output / "manifest.csv").resolve()),
        "datasets": summaries,
        "warnings": list(warnings),
        "errors": list(errors),
        "policy": {
            "audio": "16-kHz mono PCM WAV; AMI source files already conform",
            "chunking": "not yet applied; set after model-selection limits are fixed",
            "scoring": "RTTM plus dataset-specific UEM; overlap retained",
            "bangor": "28 English-primary non-Maria recordings; OSE intervals unscored",
            "playlogue": "official participant-disjoint test split and official trims",
            "afrispeech": "medical rows only by default; at least two speakers and usable timestamp pairs",
        },
    }
    atomic_text(output / "validation_report.json", json.dumps(report, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--afrispeech-domain",
        choices=("medical", "all"),
        default="medical",
        help="prepare the final medical subset (default) or every usable AfriSpeech row",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    require_commands()
    all_rows: list[ManifestRow] = []
    all_warnings: list[str] = []

    preparers = (
        ("AMI", lambda: prepare_ami(output)),
        (
            "AfriSpeech-Dialog",
            lambda: prepare_afrispeech(output, args.workers, args.afrispeech_domain),
        ),
        ("Playlogue", lambda: prepare_playlogue(output, args.workers)),
        ("Bangor Miami English-primary", lambda: prepare_bangor(output, args.workers)),
    )
    for label, prepare in preparers:
        print(f"\n== {label} ==", flush=True)
        rows, warnings = prepare()
        all_rows.extend(rows)
        all_warnings.extend(warnings)
        print(f"Prepared {len(rows)} recording(s).", flush=True)

    all_rows.sort(key=lambda row: (row.dataset, row.recording_id))
    write_manifest(output / "manifest.csv", all_rows)
    errors, summaries = validate_rows(all_rows)
    write_readiness_report(output, all_rows, all_warnings, errors, summaries)

    print("\n== Validation summary ==")
    for dataset, summary in summaries.items():
        print(
            f"{dataset}: {summary['recordings']} recordings, "
            f"{summary['audio_hours']:.2f} audio h, {summary['segments']} segments"
        )
    print(f"Warnings: {len(all_warnings)}")
    print(f"Errors: {len(errors)}")
    print(f"Manifest: {output / 'manifest.csv'}")
    print(f"Report:   {output / 'validation_report.json'}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
