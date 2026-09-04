# RQ1–RQ3 status

Source of the questions: `PAPER_ONE_PAGE_CONCEPT.md`. Source of every number
below: `DER_JER_recording_level_full95.csv` (760 rows = 95 recordings × 8
systems) and `final_manifest.csv`, both regenerated 2026-09-04 after G3-B and
G4-B were fixed.

## Where the study stands in one line

**Inference is finished — all 8 systems, 95/95, zero inconsistencies.** RQ1 and
RQ2 can be answered from data already on disk; they need statistics, not more
GPU time. **RQ3 is the real remaining work**: none of its condition metrics
exist yet.

## Completion gates (from the one-pager)

| Gate | Status |
| --- | --- |
| **G1** references and metric eligibility frozen | Mostly. Bangor's 35 untimed tiers still need alignment or masking; AfriSpeech medical-only view exported |
| **G2** all 8 systems pass pilot or declared ineligible | **Met.** No system remains ineligible — G3-B and G4-B were the two conditional ones and both now pass |
| **G3** inference matrix complete | **Met.** 8/8 systems × 95 recordings, all independently re-validated |
| **G4** RQ1–RQ3 tables, CIs, sensitivity | **Not met.** Point estimates exist; CIs, contrasts and all RQ3 conditions do not |

---

## RQ1 — Have successive architectural shifts improved accuracy across domains?

**Data: complete. Analysis: partial.**

Available now: DER/JER at collar 0.0 and 0.25, missed speech, false alarm,
speaker confusion, and speaker-count accuracy, per recording per system.

Still required: planned family contrasts with multiplicity correction, relative
error reduction, and recording-clustered bootstrap confidence intervals.

### Current standing — mean DER (collar 0), by system and dataset

| System | AMI | AfriSpeech | Bangor | Playlogue | All |
| --- | --- | --- | --- | --- | --- |
| G1-A NeMo+TitaNet+NME-SC | 0.2981 | 0.3312 | 0.3634 | 0.7280 | 0.4771 |
| G1-B MarbleNet+VBx | 0.3201 | 0.3242 | 0.3896 | 0.7579 | 0.4980 |
| G2-A pyannote community-1 | 0.1691 | 0.3525 | 0.3018 | 0.5983 | 0.3946 |
| G2-B NeMo MSDD telephonic | 0.2725 | 0.3678 | 0.4722 | 0.7433 | 0.5169 |
| G3-A Sortformer | 0.2921 | 0.3597 | 0.3325 | 0.5637 | 0.4133 |
| G3-B DiaPer | 0.3769 | 0.4133 | 0.3486 | 0.8414 | 0.5414 |
| G4-A MOSS | 0.2271 | 0.6376 | 0.3529 | 0.5284 | 0.4455 |
| G4-B VibeVoice-ASR | 0.2740 | 0.2671 | 0.2878 | 0.5851 | 0.3882 |

Fixed pipeline order, **not a ranking**.

**The headline pattern is not monotonic progress.** Within-family spread exceeds
between-family separation: G2 spans 0.3946–0.5169 and G3 spans 0.4133–0.5414,
so *which* system you pick inside a family matters more than which family it
belongs to. The two strongest overall (G4-B 0.3882, G2-A 0.3946) sit in
non-adjacent families, and the weakest (G3-B 0.5414) is newer than the G1
cascades it underperforms.

**Speaker counting is the one clean generational trend** (exact-match counts):
G1-A 27/95 and G1-B 39/95, versus G2-A 75/95, G4-A 83/95 and G4-B 83/95. That
is a large, consistent improvement and is likely a genuine RQ1 finding.

---

## RQ2 — Have gains closed gaps for unconventional/less represented domains?

**Data: complete. Analysis: partial. One confound must be handled.**

### Transfer gap (target-dataset DER − AMI DER), by family

| Family | AfriSpeech | Bangor | Playlogue |
| --- | --- | --- | --- |
| G1 embedding–clustering | 0.0187 | 0.0674 | 0.4339 |
| G2 neuralized overlap-aware | 0.1393 | 0.1662 | 0.4499 |
| G3 end-to-end discriminative | 0.0520 | 0.0061 | 0.3681 |
| G4 unified generative | 0.2018 | 0.0698 | **0.3062** |

**The answer differs by domain, which is itself the interesting result.**

- **Adult–child (Playlogue): the gap does narrow across families** —
  0.4339 → 0.4499 → 0.3681 → 0.3062. This is the clearest support for the
  study's hypothesis.
- **Code-switched (Bangor): no monotonic trend** (0.0674 → 0.1662 → 0.0061 →
  0.0698).
- **African-accented medical (AfriSpeech): the gap *widens* for G4** (0.2018),
  driven almost entirely by G4-A's 0.6376 on that dataset — an outlier against
  its own 0.2271 on AMI.

Playlogue is the hardest domain for every system (0.53–0.84), so absolute
difficulty and gap-closure are separate stories.

### Confound that must be stated, not buried

**AMI is in Sortformer's (G3-A) training data** — documented in
`runs/architecture_audit/RESULTS.md:22`. RQ2 uses AMI as the *baseline* of every
gap, so G3's gaps are computed against a number that is not an independent test
for one of its two members. G3's apparent Bangor gap of 0.0061 is therefore not
directly comparable to the other families'. Options: report G3 gaps with an
explicit caveat, compute a G3-B-only variant, or add a non-AMI reference.

Still required: family-by-dataset interaction test, rank correlation across
datasets, rank-reversal analysis, clustered CIs.

---

## RQ3 — In which speech conditions have gains been most effective?

**This is the gap. Almost nothing here is computed yet.**

Recording-level stratifiers already exist in `final_manifest.csv`
(`overlap_pct`, `short_turn_pct`, `median_turn_sec`, `speaker_imbalance`,
`code_switch_turn_pct`, `num_speakers`), so *stratified* DER can be produced
quickly. But the condition-specific metrics the RQ actually asks for do not
exist and need new scoring code:

| RQ3 measure | Status | Note |
| --- | --- | --- |
| Condition-stratified DER | **Doable now** | recording-level stratifiers already in the manifest |
| Boundary precision / recall / F1 | **Not implemented** | needs new metric |
| Onset / offset error | **Not implemented** | needs new metric |
| Overlap-specific error | **Not implemented** | AMI + Bangor only; Playlogue RTTMs contain zero overlap |
| Child vs adult conditioned error | **Not implemented** | Playlogue; needs role labels, not anonymous speaker numbers |
| Per-speaker confusion | **Partial** | aggregate confusion seconds exist; per-speaker does not |
| Code-switched vs non-code-switched turns | **Not implemented** | Bangor; `code_switch_turns` is per recording, not per turn |
| Family × condition interaction | **Not implemented** | depends on the above |

---

## Challenges and open items

1. **RQ3 is a genuine implementation task**, not a scoring re-run — boundary,
   onset/offset, overlap-specific and role-conditioned metrics all need writing
   and unit-testing before any RQ3 claim.
2. **The AMI/Sortformer training-data confound** directly affects the RQ2
   baseline and needs a stated decision.
3. **No confidence intervals anywhere yet.** Every number above is a point
   estimate; the one-pager requires recording-clustered bootstrap CIs before
   claims.
4. **Bangor's 35 untimed tiers** remain unresolved (gate G1), which constrains
   Bangor overlap and code-switch analyses.
5. **The oracle-speaker-count condition (roadmap step 5) has not been run.**
   Given speaker counting varies from 27/95 to 83/95, this separates counting
   error from assignment error and is load-bearing for RQ1 and RQ3.
6. **Two systems were only just fixed.** G3-B and G4-B results are hours old.
   G4-B additionally has 3 recordings below 98% coverage
   (`zeledon06` 0.13, `ew_42ec_22118` 0.51, `IS1009a` 0.96) that survived 11
   seeds; RQ3 should treat those as a documented limitation rather than
   ordinary data.
7. **G4-A's AfriSpeech result (0.6376) is an outlier** worth auditing before it
   drives the RQ2 conclusion that G4 widens the accented-medical gap — it may
   be the chunk-recovery artifact noted in `RESULTS.md` rather than a property
   of the family.

## Suggested order of work

1. Bootstrap CIs + family contrasts on existing data → completes RQ1.
2. Decide the AMI-confound handling, then finish RQ2's interaction/rank tests.
3. Audit G4-A on AfriSpeech (item 7) — it changes an RQ2 headline.
4. Implement RQ3 metrics, starting with boundary/onset-offset and
   overlap-specific (AMI + Bangor), then Playlogue child/adult.
5. Run the oracle-speaker-count condition.
