# RQ4 milestone 1 — status, and one blocker that changes the plan

Scope worked: the audio-independent parts of milestone 1 (protocol guardrails,
leakage validation, frozen scorer). Nothing here touched the final test.

## Blocker: the adaptation audio is not present

The plan states a repository audit found "97 official train recordings (about
19.09 h) and 27 official validation recordings (about 5.99 h) **locally**".
That is not the case on this machine. Verified by searching **607,531 audio
files** under `$HOME` across `.wav/.mp3/.flac/.m4a/.ogg`:

| Playlogue split | recordings | have audio | have diarization reference |
|---|---|---|---|
| train | 97 | **0** | 97 |
| val | 27 | **0** | 27 |
| test | 34 | 34 | 34 |

`data/datasets/playlogue/playlogue-v1/data/audio/` contains only a `README.md`.
Metadata (`metadata/splits.csv`, `metadata/clip_timings.csv`) and all 158
diarization references are present; the media is not, consistent with Playlogue
being gated CHILDES-derived data.

**The only Playlogue audio on this machine is the 34 final-test recordings** —
precisely the audio RQ4 forbids for training, calibration, or pseudo-labeling.

Consequence: Phase F as written (Playlogue train/validation only) cannot start.
A1 calibration, teacher inference on adaptation windows, consensus
construction, and all of A2-S/A2-C/A2-H are blocked on obtaining the gated
train/validation media. This is a data-access task, not an engineering one.

What is *not* blocked, and was built: leakage validation and the scorer, both of
which operate on metadata and RTTMs.

## Delivered

### `validate_leakage.py` — gate G1

Rejects cross-role overlap by recording ID, individual participant identity,
`participant_group`, source recording ID, exact audio hash, normalized-PCM
hash, and acoustic near-duplicate fingerprint.

Two protocol subtleties are handled explicitly because a plausible
implementation gets them wrong:

- **Participant checks intersect the sets of individual IDs**, never compare
  serialized participant-set strings. Two recordings sharing one child but
  differing in the other adult have different group strings while still
  leaking that child.
- **`participant_group` is a connected component** of the recording–participant
  graph (Union-Find), so it is transitive: if A–B share a participant and B–C
  share a different one, A and C are the same group even though they share
  nobody directly.

Final-test leakage returns exit code 2 and is unconditional — `--engineering-only`
downgrades *unknown participant identity* to a warning, never actual overlap.

**Tests: 11/11 pass.** They found a real bug during development: `role_of` was
keyed by `recording_id`, so a duplicate ID silently overwrote its predecessor
and its cross-role overlap became invisible to every downstream check. Now
detected on the raw rows before any dict keying.

### `score_diarization.py` — frozen scorer

UEM-aware, collar 0.0 primary (0.25 secondary), overlap included, both
reference and hypothesis cropped to the UEM. Settings are pinned in `FROZEN`
and re-asserted at runtime under `--assert-frozen`, so an edited constant
cannot quietly change published numbers. Emits per-recording CSV plus a summary
JSON carrying the frozen settings and a hash of every hypothesis file.

**Cross-checked against the architecture audit**, as the plan requires: on
G1-A's 95 recordings this scorer gives mean DER (collar=0) **0.4772** against
the audit's independently-implemented **0.4771** — agreement to 4 decimals, the
residual being per-recording rounding precision. 95/95 scored, 0 errors.

## Recommended next steps

1. **Obtain gated Playlogue train/validation audio.** Everything downstream
   waits on this. Worth confirming whether the plan's "locally available" claim
   referred to a different machine.
2. Re-audit AfriSpeech non-medical (29 recordings, ~5.03 h) and Bangor Maria
   (13, ~9.87 h unlabeled-only) for participant independence — those pools may
   allow partial progress without Playlogue media.
3. Build `recordings.csv` once media exists; the validator is ready to gate it.
4. The plan's participant-ID parser is still required: Playlogue metadata has
   no canonical participant-ID column, so `participant_ids` cannot yet be
   populated and any run today would be `--engineering-only` at best.

## Note on the final test

Untouched. No RQ4 code read final-test audio; the scorer cross-check used
already-published architecture-audit hypotheses purely to validate the scorer
against a known number, which is not model development.
