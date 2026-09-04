#!/usr/bin/env python3
"""Tests for RQ4 gate G1 (leakage validation).

The cases that matter most are the ones a naive implementation passes by
accident: partial participant overlap (different group strings, same child),
transitive group membership, and final-test leakage being unbypassable.
"""
import csv
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "validate_leakage.py")

COLUMNS = ["dataset", "recording_id", "source_recording_id", "participant_ids",
           "participant_group", "pool_role", "domain", "audio_duration_sec",
           "scored_duration_sec", "audio_sha256", "normalized_pcm_sha256",
           "acoustic_fingerprint", "reference_rttm_sha256", "uem_sha256",
           "source", "license"]


def rec(rid, role, participants="", src="", sha="", pcm="", fp=""):
    return {c: "" for c in COLUMNS} | {
        "dataset": "playlogue", "recording_id": rid, "pool_role": role,
        "participant_ids": participants, "source_recording_id": src,
        "audio_sha256": sha, "normalized_pcm_sha256": pcm,
        "acoustic_fingerprint": fp, "audio_duration_sec": "100",
    }


def run(records, extra=()):
    d = tempfile.mkdtemp()
    p = os.path.join(d, "recordings.csv")
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(records)
    r = subprocess.run([sys.executable, SCRIPT, "--recordings", p, *extra],
                       capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


CASES = []


def case(fn):
    CASES.append(fn)
    return fn


@case
def test_clean_passes():
    rc, out = run([
        rec("a", "target_adaptation_pool", "child1|adult1"),
        rec("b", "final_test", "child9|adult9"),
    ])
    assert rc == 0, out
    assert "GATE G1: PASS" in out


@case
def test_partial_participant_overlap_is_caught():
    """The key case: different participant-set strings, one shared child.
    A string comparison would pass this; set intersection must not."""
    rc, out = run([
        rec("a", "target_adaptation_pool", "childX|adultA"),
        rec("b", "final_test", "childX|adultB"),
    ])
    assert rc == 2, f"expected fatal final-test leakage, got rc={rc}\n{out}"
    assert "individual participant overlap" in out


@case
def test_transitive_group_membership():
    """A-B share p1, B-C share p2. A and C share nobody, but all three are one
    connected component, so splitting A and C across roles must fail."""
    rc, out = run([
        rec("a", "target_adaptation_pool", "p1"),
        rec("b", "target_adaptation_pool", "p1|p2"),
        rec("c", "final_test", "p2"),
    ])
    assert rc == 2, f"expected fatal, got {rc}\n{out}"
    assert "participant_group" in out or "individual participant overlap" in out


@case
def test_exact_recording_id_overlap():
    rc, out = run([
        rec("dup", "target_adaptation_pool", "p1"),
        rec("dup", "source_pool", "p2"),
    ])
    assert rc == 1, f"expected failure, got {rc}\n{out}"
    assert "recording_id overlap" in out


@case
def test_audio_hash_overlap():
    rc, out = run([
        rec("a", "target_adaptation_pool", "p1", sha="deadbeef"),
        rec("b", "source_pool", "p2", sha="deadbeef"),
    ])
    assert rc == 1, out
    assert "exact audio hash" in out


@case
def test_normalized_pcm_hash_overlap():
    rc, out = run([
        rec("a", "target_adaptation_pool", "p1", pcm="cafebabe"),
        rec("b", "source_pool", "p2", pcm="cafebabe"),
    ])
    assert rc == 1, out
    assert "normalized-PCM hash" in out


@case
def test_near_duplicate_fingerprint_overlap():
    rc, out = run([
        rec("a", "target_adaptation_pool", "p1", fp="fp123"),
        rec("b", "target_pseudo_audit_pool", "p2", fp="fp123"),
    ])
    assert rc == 1, out
    assert "near-duplicate" in out


@case
def test_source_recording_id_overlap():
    rc, out = run([
        rec("a", "target_adaptation_pool", "p1", src="S1"),
        rec("b", "source_pool", "p2", src="S1"),
    ])
    assert rc == 1, out
    assert "source_recording_id overlap" in out


@case
def test_missing_participants_blocks_confirmatory():
    rc, out = run([
        rec("a", "target_adaptation_pool", ""),
        rec("b", "final_test", "p9"),
    ])
    assert rc == 1, f"missing identity must fail without the flag\n{out}"
    assert "participant identity" in out


@case
def test_missing_participants_allowed_engineering_only():
    rc, out = run([
        rec("a", "target_adaptation_pool", ""),
        rec("b", "final_test", "p9"),
    ], extra=("--engineering-only",))
    assert rc == 0, f"engineering-only should downgrade to warning\n{out}"
    assert "WARN" in out and "may not be used as confirmatory" in out


@case
def test_engineering_only_cannot_bypass_real_final_test_leakage():
    """The escape hatch covers unknown identity, never actual overlap."""
    rc, out = run([
        rec("a", "target_adaptation_pool", "shared"),
        rec("b", "final_test", "shared"),
    ], extra=("--engineering-only",))
    assert rc == 2, f"final-test leakage must stay fatal, got {rc}\n{out}"
    assert "unconditional" in out


def main():
    failed = 0
    for fn in CASES:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(CASES) - failed}/{len(CASES)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
