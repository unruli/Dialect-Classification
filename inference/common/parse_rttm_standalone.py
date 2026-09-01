#!/usr/bin/env python3
"""Normalize a raw RTTM (from any of the 4 diarization models) into a
validated, anonymous-speaker RTTM: canonical SPEAKER_00.. labels, sorted by
onset, fields format-checked. Speaker labels are remapped in first-seen
order so no model-internal ID (embedding index, cluster id, HF pipeline
label) leaks into the output.

Usage: parse_rttm.py <raw_rttm_in> <clean_rttm_out> <uri>
Exit code 0 + "PARSE_OK" on stdout iff the output is a valid, non-empty RTTM.
"""
import sys


def parse_and_normalize(raw_path, uri):
    segments = []
    with open(raw_path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) < 8 or fields[0] != "SPEAKER":
                raise ValueError(f"line {lineno}: not a well-formed RTTM SPEAKER line: {line!r}")
            onset = float(fields[3])
            duration = float(fields[4])
            speaker = fields[7]
            if duration <= 0:
                raise ValueError(f"line {lineno}: non-positive duration {duration}")
            if onset < 0:
                raise ValueError(f"line {lineno}: negative onset {onset}")
            segments.append((onset, duration, speaker))

    if not segments:
        raise ValueError("no SPEAKER lines found")

    segments.sort(key=lambda s: (s[0], s[1]))

    label_map = {}
    out_lines = []
    for onset, duration, speaker in segments:
        if speaker not in label_map:
            label_map[speaker] = f"SPEAKER_{len(label_map):02d}"
        anon = label_map[speaker]
        out_lines.append(
            f"SPEAKER {uri} 1 {onset:.3f} {duration:.3f} <NA> <NA> {anon} <NA> <NA>"
        )
    return out_lines, len(label_map)


def main():
    if len(sys.argv) != 4:
        print("Usage: parse_rttm.py <raw_rttm_in> <clean_rttm_out> <uri>", file=sys.stderr)
        sys.exit(2)
    raw_path, out_path, uri = sys.argv[1], sys.argv[2], sys.argv[3]
    try:
        out_lines, n_speakers = parse_and_normalize(raw_path, uri)
    except Exception as e:
        print(f"PARSE_FAIL: {e}", file=sys.stderr)
        sys.exit(1)

    with open(out_path, "w") as f:
        f.write("\n".join(out_lines) + "\n")

    print(f"PARSE_OK segments={len(out_lines)} speakers={n_speakers} out={out_path}")


if __name__ == "__main__":
    main()
