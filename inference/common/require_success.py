#!/usr/bin/env python3
"""Fail a scheduler gate unless an inference manifest is wholly successful."""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--pass-name", default=None)
    args = parser.parse_args()

    with open(args.manifest) as handle:
        manifest = json.load(handle)
    recordings = manifest.get("recordings", {})
    records = list(recordings.values()) if isinstance(recordings, dict) else list(recordings)
    if args.pass_name is not None:
        records = [record for record in records if record.get("pass") == args.pass_name]

    failures = [
        {
            "dataset": record.get("dataset"),
            "recording_id": record.get("recording_id"),
            "status": record.get("status"),
            "error": record.get("error"),
        }
        for record in records
        if record.get("status") != "success" or record.get("truncated") is True
    ]
    ok = len(records) == args.expected_count and not failures
    print(
        json.dumps(
            {
                "gate_passed": ok,
                "expected_count": args.expected_count,
                "observed_count": len(records),
                "pass_name": args.pass_name,
                "failures": failures,
            },
            indent=2,
        )
    )
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
