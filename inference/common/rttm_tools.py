"""RTTM normalization and validation, shared across all systems.

Exported from diar_smoke/scripts/parse_rttm.py (proven on the G1-A/G1-B/G2-A
pilot and full run) and extended with the source-duration bound check
required by MODEL_SELECTION_AND_INFERENCE.md's output contract: "Validate
that starts and durations are finite and nonnegative, and that output end
times do not exceed the source by more than 0.5 seconds."
"""
import math


class RTTMValidationError(ValueError):
    pass


def parse_and_normalize(raw_lines, uri):
    """Normalize raw SPEAKER lines into (start, duration, anonymous_label)
    tuples, sorted by onset, with speaker labels remapped to SPEAKER_00,
    SPEAKER_01, ... in first-seen order. `raw_lines` is an iterable of str.
    """
    segments = []
    for lineno, line in enumerate(raw_lines, 1):
        line = line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) < 8 or fields[0] != "SPEAKER":
            raise RTTMValidationError(f"line {lineno}: not a well-formed RTTM SPEAKER line: {line!r}")
        onset = float(fields[3])
        duration = float(fields[4])
        speaker = fields[7]
        if not (math.isfinite(onset) and math.isfinite(duration)):
            raise RTTMValidationError(f"line {lineno}: non-finite onset/duration")
        if duration <= 0:
            raise RTTMValidationError(f"line {lineno}: non-positive duration {duration}")
        if onset < 0:
            raise RTTMValidationError(f"line {lineno}: negative onset {onset}")
        segments.append((onset, duration, speaker))

    if not segments:
        raise RTTMValidationError("no SPEAKER lines found")

    segments.sort(key=lambda s: (s[0], s[1]))

    label_map = {}
    normalized = []
    for onset, duration, speaker in segments:
        if speaker not in label_map:
            label_map[speaker] = f"SPEAKER_{len(label_map):02d}"
        normalized.append((onset, duration, label_map[speaker]))
    return normalized


def write_rttm(segments, uri, out_path):
    with open(out_path, "w") as f:
        for onset, duration, label in segments:
            f.write(f"SPEAKER {uri} 1 {onset:.3f} {duration:.3f} <NA> <NA> {label} <NA> <NA>\n")


def validate_against_source_duration(segments, source_duration_sec, tolerance_sec=0.5):
    """Per the output contract: output end times must not exceed the source
    duration by more than `tolerance_sec`."""
    max_end = max(onset + duration for onset, duration, _ in segments)
    if max_end > source_duration_sec + tolerance_sec:
        raise RTTMValidationError(
            f"max output end time {max_end:.3f}s exceeds source duration "
            f"{source_duration_sec:.3f}s by more than {tolerance_sec}s"
        )
    return max_end


def normalize_rttm_file(raw_rttm_path, out_rttm_path, uri, source_duration_sec=None):
    """Full pipeline: read raw RTTM, normalize+anonymize, optionally check
    against source duration, write normalized RTTM. Returns (n_segments,
    n_speakers). Raises RTTMValidationError on any problem -- callers should
    treat this file as a failed run, not silently accept a partial result.
    """
    with open(raw_rttm_path) as f:
        segments = parse_and_normalize(f, uri)
    if source_duration_sec is not None:
        validate_against_source_duration(segments, source_duration_sec)
    write_rttm(segments, uri, out_rttm_path)
    n_speakers = len({label for _, _, label in segments})
    return len(segments), n_speakers


def is_valid_normalized_rttm(rttm_path, source_duration_sec=None):
    """Cheap validity check for resume logic: does this file parse as a
    normalized RTTM without re-writing it? Returns True/False, never raises."""
    try:
        with open(rttm_path) as f:
            lines = f.readlines()
        segments = parse_and_normalize(lines, uri="_check_")
        if source_duration_sec is not None:
            validate_against_source_duration(segments, source_duration_sec)
        return True
    except (RTTMValidationError, OSError, ValueError):
        return False
