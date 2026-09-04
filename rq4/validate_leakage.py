#!/usr/bin/env python3
"""RQ4 leakage validation (gate G1).

Rejects any overlap across experiment roles by: recording ID, *individual*
participant identity, source recording ID, exact audio hash, normalized-PCM
hash, and acoustic near-duplicate fingerprint.

Two details in the protocol are easy to get subtly wrong, so they are handled
explicitly here:

1. Participant checks compare the *sets of individual participant IDs*, not a
   serialized participant-set string. Two recordings sharing one child but
   differing in the other adult produce different group strings while still
   leaking that child -- string comparison would pass them. The spec calls this
   out directly ("comparing serialized participant-set strings is
   insufficient").

2. `participant_group` must be a connected component of the recording-
   participant bipartite graph, not a per-recording label. If A shares a
   participant with B, and B with C, then A, B and C are one group even when A
   and C share nobody. Union-Find is used so this is transitive by
   construction.

Any leakage touching `final_test` is fatal and unconditional. Other leakage is
fatal too, but `--engineering-only` may be used to downgrade *missing
participant identity* (not actual overlap) to a warning for smoke tests, per
the plan's allowance.
"""
from __future__ import annotations

import argparse
import collections
import csv
import itertools
import json
import sys

FINAL_TEST = "final_test"

# Roles that must never share data with one another.
POOL_ROLES = {
    "target_adaptation_pool",
    "target_pseudo_audit_pool",
    "source_pool",
    "rq4_confirmatory_holdout",
    FINAL_TEST,
}


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:          # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def parse_participants(value):
    """Participant IDs are stored as a delimited list. Empty/absent means the
    identity is unknown, which is materially different from 'no participants'
    and is reported separately."""
    if value is None:
        return None
    v = value.strip()
    if not v:
        return None
    return frozenset(p.strip() for p in v.replace(";", "|").split("|") if p.strip())


def derive_groups(recordings):
    """participant_group = connected component of the recording-participant graph."""
    uf = UnionFind()
    for r in recordings:
        uf.union(("rec", r["recording_id"]), ("rec", r["recording_id"]))
        for p in (r["_participants"] or ()):
            uf.union(("rec", r["recording_id"]), ("part", p))
        # Recordings cut from the same source share identity by construction.
        src = (r.get("source_recording_id") or "").strip()
        if src:
            uf.union(("rec", r["recording_id"]), ("src", src))
    return {r["recording_id"]: uf.find(("rec", r["recording_id"])) for r in recordings}


def check(recordings, role_of, engineering_only=False):
    """role_of: recording_id -> role label. Returns (violations, warnings)."""
    violations, warnings = [], []

    # --- duplicate recording IDs ------------------------------------------
    # Must be checked on the raw rows, before anything keys a dict by
    # recording_id: a duplicate ID would otherwise be silently collapsed and
    # its cross-role overlap would become invisible to every check below.
    id_roles = collections.defaultdict(set)
    for r in recordings:
        id_roles[r["recording_id"]].add(r.get("pool_role", ""))
    dupes = {k: v for k, v in id_roles.items()
             if sum(1 for r in recordings if r["recording_id"] == k) > 1}
    for rid, roles in sorted(dupes.items()):
        if len(roles) > 1:
            violations.append(
                f"recording_id overlap: {rid!r} appears in multiple pool roles {sorted(roles)}")
        else:
            violations.append(f"duplicate recording_id {rid!r} in recordings.csv")

    by_role = collections.defaultdict(list)
    for r in recordings:
        role = role_of.get(r["recording_id"])
        if role:
            by_role[role].append(r)

    # --- unknown participant identity -------------------------------------
    unknown = [r["recording_id"] for r in recordings if r["_participants"] is None]
    if unknown:
        msg = (f"{len(unknown)} recording(s) have no participant identity "
               f"(e.g. {unknown[:3]})")
        if engineering_only:
            warnings.append(msg + " -- allowed only because --engineering-only was set; "
                                  "this run may not be used as confirmatory")
        else:
            violations.append(msg + " -- confirmatory runs require participant identity")

    # --- direct recording-ID overlap --------------------------------------
    for a, b in itertools.combinations(sorted(by_role), 2):
        ids_a = {r["recording_id"] for r in by_role[a]}
        ids_b = {r["recording_id"] for r in by_role[b]}
        shared = ids_a & ids_b
        if shared:
            violations.append(f"recording_id overlap between {a} and {b}: {sorted(shared)[:5]}")

    # --- individual participant intersection (NOT set-string equality) ----
    for a, b in itertools.combinations(sorted(by_role), 2):
        pa = set().union(*[r["_participants"] for r in by_role[a] if r["_participants"]]) \
            if any(r["_participants"] for r in by_role[a]) else set()
        pb = set().union(*[r["_participants"] for r in by_role[b] if r["_participants"]]) \
            if any(r["_participants"] for r in by_role[b]) else set()
        shared = pa & pb
        if shared:
            violations.append(
                f"individual participant overlap between {a} and {b}: {sorted(shared)[:5]}")

    # --- connected-component (participant_group) crossing ------------------
    groups = derive_groups(recordings)
    group_roles = collections.defaultdict(set)
    for r in recordings:
        role = role_of.get(r["recording_id"])
        if role:
            group_roles[groups[r["recording_id"]]].add(role)
    for g, roles in group_roles.items():
        if len(roles) > 1:
            members = [r["recording_id"] for r in recordings if groups[r["recording_id"]] == g]
            violations.append(
                f"participant_group {g} spans roles {sorted(roles)}: {sorted(members)[:5]}")

    # --- source recording ID ----------------------------------------------
    for a, b in itertools.combinations(sorted(by_role), 2):
        sa = {r["source_recording_id"] for r in by_role[a] if r.get("source_recording_id")}
        sb = {r["source_recording_id"] for r in by_role[b] if r.get("source_recording_id")}
        shared = sa & sb
        if shared:
            violations.append(f"source_recording_id overlap between {a} and {b}: {sorted(shared)[:5]}")

    # --- hashes and near-duplicate fingerprints ---------------------------
    for field, label in (("audio_sha256", "exact audio hash"),
                         ("normalized_pcm_sha256", "normalized-PCM hash"),
                         ("acoustic_fingerprint", "acoustic near-duplicate fingerprint")):
        seen = collections.defaultdict(set)
        for r in recordings:
            role = role_of.get(r["recording_id"])
            v = (r.get(field) or "").strip()
            if role and v:
                seen[v].add(role)
        for v, roles in seen.items():
            if len(roles) > 1:
                violations.append(f"{label} {v[:16]}... shared across roles {sorted(roles)}")

    # --- final test is sacred ---------------------------------------------
    fatal_final = [v for v in violations if FINAL_TEST in v]
    return violations, warnings, fatal_final


def load(path):
    recs = list(csv.DictReader(open(path)))
    for r in recs:
        r["_participants"] = parse_participants(r.get("participant_ids"))
    return recs


def main():
    ap = argparse.ArgumentParser(description="RQ4 gate G1: leakage validation")
    ap.add_argument("--recordings", required=True, help="recordings.csv")
    ap.add_argument("--assignments", default=None,
                    help="assignments/seed_<n>.csv; if given, roles come from it")
    ap.add_argument("--engineering-only", action="store_true",
                    help="downgrade MISSING participant identity (never actual overlap) "
                         "to a warning; such a run can never be confirmatory")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    recs = load(args.recordings)
    if args.assignments:
        role_of = {r["recording_id"]: r["experiment_role"]
                   for r in csv.DictReader(open(args.assignments))}
        # recordings not in the seed assignment keep their immutable pool role,
        # so final_test still participates in every check
        for r in recs:
            role_of.setdefault(r["recording_id"], r["pool_role"])
    else:
        role_of = {r["recording_id"]: r["pool_role"] for r in recs}

    violations, warnings, fatal_final = check(recs, role_of, args.engineering_only)

    for w in warnings:
        print(f"WARN  {w}")
    for v in violations:
        print(f"FAIL  {v}")

    result = {
        "n_recordings": len(recs),
        "roles": sorted({v for v in role_of.values()}),
        "n_violations": len(violations),
        "n_warnings": len(warnings),
        "final_test_leakage": len(fatal_final),
        "violations": violations,
        "warnings": warnings,
        "gate_G1_pass": not violations,
    }
    if args.json_out:
        json.dump(result, open(args.json_out, "w"), indent=2)

    if fatal_final:
        print(f"\nGATE G1: FAIL -- {len(fatal_final)} violation(s) involve {FINAL_TEST}. "
              "This is unconditional; --engineering-only cannot bypass it.")
        return 2
    if violations:
        print(f"\nGATE G1: FAIL -- {len(violations)} violation(s)")
        return 1
    print(f"\nGATE G1: PASS -- {len(recs)} recordings, roles {result['roles']}, "
          f"{len(warnings)} warning(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
