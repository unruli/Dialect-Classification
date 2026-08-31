# Legacy orchestration scripts (historical, superseded)

These are unmodified copies of the exact scripts that produced the current
G1-A/G1-B/G2-A 95-recording run on `levi-compute`, preserved for provenance.
They are **not** part of the new collaborator interface and are **not**
maintained going forward -- use `inference/run_model.py` instead.

- `run_pilot_file.sh` -- per-recording driver (stage from a remote host via
  scp, run G1-A, reuse its VAD for G1-B, run G2-A, normalize+validate all
  three, delete staged audio only once all three validate).
- `run_full_batch.sh` -- iterates `run_pilot_file.sh` over all 95 rows of the
  frozen manifest, tracks per-file progress, continues past a single file's
  failure rather than aborting a ~16-hour run.

Both scripts contain machine-specific absolute paths
(`/home/kelechi/Dialect-Classification`, `/dev/shm/dialect-smoke`, a specific
remote host) and assume a remote-staging workflow specific to this project's
two-host setup (audio lives on a separate host from the GPU). `run_model.py`
generalizes the same proven per-file logic (see `g1a_nemo/`, `g1b_vbx/`,
`g2a_pyannote/`) into a single path-parameterized CLI that assumes the
collaborator already has local audio access, per `inference/README.md`.
