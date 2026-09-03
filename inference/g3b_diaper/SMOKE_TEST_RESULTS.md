# G3-B smoke test results

Run on levi-compute (RTX 4090, driver 535.309.01), `diar_g3b` env
(Python 3.7.16, torch 1.10.0+cu113, `torch.cuda.is_available()` confirmed
True immediately before this run). 90-second trimmed excerpts of the 4
fixed pilot recordings, per `MODEL_SELECTION_AND_INFERENCE.md`'s common
pilot protocol. `--gpu 1`.

| recording | wall time | segments | speakers | max end time | RTTM |
|---|---|---|---|---|---|
| afrispeech_dialog / 5129fd8c... | 5.05s | 55 | 2 | 90.00s | OK |
| ami / EN2002a | 4.82s | 117 | 5 | 89.70s | OK |
| bangor_miami_eng / sastre03 | 4.81s | 115 | 5 | 90.00s | OK |
| playlogue / ew_42pc_22148 | 5.08s | 139 | 6 | 89.90s | OK |

**4/4 passed.** Each output was independently re-parsed and validated with
`common/rttm_tools.py` (`parse_and_normalize` +
`validate_against_source_duration` against the 90.0s trim duration) --
not just a trust of the runner's own `"ok": true` -- confirming well-formed
10-field RTTM, positive durations, and max end time within the source
duration bound in every case. Raw outputs are in `smoke_evidence/`.

Segment counts and speaker counts are identical to an earlier 2026-08-31
run of the same 4 recordings on this same host, i.e. deterministic
(`seed: 3` in the yaml, no sampling at inference).

**This confirms the model/environment/patch are functionally correct on a
small clip. It does NOT confirm full-recording or longest-file feasibility
-- see `ENVIRONMENT.md`'s "Full-recording status" section for the known
OOM history that motivates running this on an A100 instead of the 4090.**
Before any 95-file batch, re-run the longest-file gate (AMI EN2002c,
49.54min) on the target A100 and confirm it completes without OOM.
