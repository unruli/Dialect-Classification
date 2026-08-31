# Working research plan: speaker-attributed transcription under difficult speech

**Status:** planning document, updated 31 August 2026
**Primary application:** African-accented English dialogue
**Secondary transfer setting:** adult-child / child-speech interactions from TalkBank HSLLD

The current diarization-only, four-family extension is governed by
[`PAPER_ONE_PAGE_CONCEPT.md`](PAPER_ONE_PAGE_CONCEPT.md) and the frozen-candidate
record in
[`MODEL_SELECTION_AND_INFERENCE.md`](MODEL_SELECTION_AND_INFERENCE.md). This
working plan retains the broader speaker-attributed-transcription design; its
older single-diarizer recommendation is superseded by the architecture audit
below.

## 1. The project in one paragraph

Speech systems are commonly assessed either as automatic speech recognizers
(ASR) or as speaker diarizers. In real conversations the useful output is
*speaker-attributed transcription*: the words, their timing, and the speaker to
whom each word belongs. This project will test how reliably modern modular
systems and direct audio language models produce that output in difficult
interactive speech—especially African-accented medical dialogue, overlap, and
disfluencies. HSLLD is a domain-transfer stress test involving natural
adult-child interaction; it is not evidence of an African-accent effect.

## 2. Recommended paper claim and scope

### Primary claim

Speaker-attributed transcription degrades substantially when conversational
structure is difficult (overlap, rapid turns, disfluencies), and a good-looking
transcript can conceal errors in speaker identity and timing. Compare the
failure modes of a modular diarization-plus-ASR pipeline with prompted direct
audio/speech language models.

### Claims to avoid unless new evidence is collected

- Do not describe the study as a general *dialect classification* project.
  No current task labels or controls support that claim.
- Do not claim a causal accent effect without a matched non-African-accent
  control.
- Do not claim a causal medical-versus-education difference. The datasets
  differ in age, microphones, setting, interaction structure, and likely noise.
- Do not call a prompted LLM a diarization model. Call it a **prompted direct
  speaker-attributed transcription** system.

## 3. Research questions

### Core RQ1 — propagation of diarization error

How much does predicted diarization, relative to oracle human turns, degrade
the transcript and speaker attribution produced by a modular ASR pipeline?

### Core RQ2 — direct versus modular systems

Can a direct audio/speech LLM produce speaker-attributed transcripts comparable
to a modular pipeline, without an external diarizer?

### Core RQ3 — difficult interaction phenomena

How do overlap, short turns, false starts, repairs, repetitions, filled pauses,
and interruptions affect content accuracy, speaker attribution, and timing?

### Secondary RQ4 — transfer

Do the above error patterns transfer from African-accented adult dialogue to
natural adult-child interaction in HSLLD? Treat this as a robustness/domain
transfer question, not a controlled domain comparison.

## 4. Dataset plan

| Role | Dataset | What it contributes | Present status / requirement |
| --- | --- | --- | --- |
| Primary evaluation | AfriSpeech-Dialog timestamped subset | African-accented, dyadic medical and general dialogue; reference speaker times | Need an exact, versioned manifest, audio, RTTM/UEM, and turn text references. |
| ASR/accent auxiliary | AfriSpeech-200 | Large multi-accent, mainly single-speaker ASR material | Optional; it cannot evaluate overlap or speaker attribution by itself. |
| Secondary transfer | TalkBank HSLLD | Natural adult-child home interaction; linked CHAT transcripts and downloaded audio | CHAT corpus is in `data/external/TalkBank`; bulk audio must be moved from Downloads and paired with `.cha` files. |
| Disfluency auxiliary | DisfluencySpeech or a manually annotated subset | Controlled test of transcript policy and disfluencies | Prefer a manually annotated AfriSpeech-Dialog test subset to preserve the primary setting. |

### HSLLD preparation checklist

1. Move the downloaded MP3 hierarchy into `data/external/TalkBank/Eng-NA/HSLLD/`, preserving `HV*/<task>/` folders and filenames.
2. Build a manifest pairing every MP3 with its `.cha` transcript; flag missing pairs rather than silently dropping them.
3. Select a pilot sample with clear adult-child speech, then listen-check audio/transcript alignment.
4. Extract participant tiers and timestamps from CHAT where available. CHAT transcripts alone are not automatically a gold-standard word-timing reference.
5. Obtain/use the permitted TalkBank citation and follow its data-use rules; do not redistribute restricted audio.

## 5. Experimental design

Run the same recordings through three conditions. This isolates whether an
error is due to ASR, diarization, or an end-to-end/direct model.

| Condition | Input to system | Purpose |
| --- | --- | --- |
| Oracle segmentation + ASR | Human speaker turns/times | Content ceiling once speaker structure is correct. |
| Predicted diarization + ASR | Raw audio, then diarizer output | Error propagation in a modular pipeline. |
| Direct speech LLM | Raw audio and one fixed prompt | Native speaker-attributed transcript without external diarization. |

### Output contract for every system

Store raw output and a normalized parse with the fields:

`recording_id, start_time, end_time, anonymous_speaker_id, transcript`

Speaker labels are anonymous. Map predicted labels to reference speakers with a
one-to-one optimal/permutation mapping before scoring. Log malformed outputs as
an outcome (schema-validity rate), not as an invisible manual correction.

### Pilot before scale-up

Use a small, audited set (for example 10–15 recordings) balanced where possible
across overlap/disfluency presence and domain. Run one modular baseline and one
direct audio LLM. Confirm the parser, scoring, annotation policy, and data-use
approvals before expanding the model grid.

## 6. Annotation and reference policy

Build two reference targets for the pilot/test data:

- **Verbatim:** preserve repetitions, false starts, fillers, repairs, and
  incomplete words according to a written annotation guide.
- **Normalized:** apply a deterministic, published transformation policy to the
  same turns. This is not a grammaticality judgement.

For each reference turn retain speaker ID, start/end time, text, and overlap
state. Add tags for false starts, repetitions, filled pauses, repairs, and
abandoned/interrupted turns. Have a second annotator check a subset and report
agreement for the phenomenon tags and/or boundaries.

## 7. Metrics and analysis

| Target | Primary measures |
| --- | --- |
| Diarization | DER, including FA/MISS/CONF; report overlap-scored and overlap-excluded sensitivity analyses. |
| ASR content | WER against verbatim and normalized references. |
| Attributed words | cpWER / speaker-attributed WER after speaker permutation. |
| Speaker identity | Turn-level speaker-attribution accuracy or F1 after mapping. |
| Timing | Boundary-tolerance F1 or onset/offset error. |
| Operational robustness | Schema-validity, unparseable-output, and dropped-audio rates. |

Aggregate and compare **by recording**, with paired bootstrap confidence
intervals and paired effect sizes. Do not treat individual words or frames as
independent observations. Stratify results by overlap, disfluency tags, and
primary versus transfer domain.

## 8. Models

For the diarization-only architecture audit, use two systems per stratum. The
G labels are architectural evaluation strata, not a universal release-date
chronology.

| ID | Stratum | Selected system | Gate |
| --- | --- | --- | --- |
| G1-A | Embedding--clustering | NeMo MarbleNet + TitaNet-Large + NME-SC | Running |
| G1-B | Embedding--clustering | MarbleNet VAD + BUT SpeechFIT VBx | Running |
| G2-A | Neuralized modular | Pyannote Community-1 | Running |
| G2-B | Neuralized modular | NeMo MSDD `diar_msdd_telephonic` v1.0.1 | Balanced pilot and domain-specialization disclosure |
| G3-A | End-to-end discriminative | NVIDIA Streaming Sortformer 4spk v2.1 | Balanced pilot and AMI-training disclosure |
| G3-B | End-to-end discriminative | DiaPer 10-attractor, non-AMI-fine-tuned checkpoint | Complete 49.54-minute memory test |
| G4-A | Unified generative | MOSS-Transcribe-Diarize 0.9B | Parser and overlap-output pilot |
| G4-B | Unified generative | VibeVoice-ASR-HF 8B | 24-GB memory, parser, and overlap-output pilot |

Pyannote 3.1 is an optional within-lineage sensitivity run, not an independent
G2-B system. Full checkpoint links, eligibility rules, pilot IDs, output
requirements, and the collaborator handoff prompt are maintained in
[`MODEL_SELECTION_AND_INFERENCE.md`](MODEL_SELECTION_AND_INFERENCE.md).

For the broader speaker-attributed-transcription experiment, start small and
expand only after the diarization pilot and output contract are valid.

| Role | Recommended initial choice | Expansion |
| --- | --- | --- |
| Diarizer | Use the relevant frozen output from the architecture audit; Community-1 is the default modular input | Compare a G3 or G4 result only after its parser and timing validation pass. |
| ASR | One version-pinned strong ASR model/API | Add an African-accent-aware ASR baseline if available and permitted. |
| Direct audio LLM | One model that accepts raw audio and reliably returns timestamps/structured output | Compare at most one additional model; pin prompt, API/model version, chunking, date, and decoding parameters. |

For every run, save the model/checkpoint/API version, prompt, chunking rule,
speaker-count hints, decoding settings, code commit, seed, hardware, and run
date. Clinical audio must only be sent to an external provider if its data-use
terms and approval allow it.

## 9. Minimum artifacts needed before paper results

1. A versioned recording manifest with IDs, split, domain, source path, and
   permitted-use status.
2. Audio plus reference RTTM/UEM or equivalent timed speaker-turn annotations.
3. Verbatim/normalized turn-text references for the scored test set.
4. A written overlap/disfluency annotation and normalization guide.
5. One reusable runner that writes standard RTTM/JSON outputs, and one
   version-pinned scoring script whose overlap behavior matches the paper.
6. Per-recording results, not just aggregate tables.
7. A data-access/de-identification statement and all required corpus citations.

## 10. Paper structure

1. **Introduction:** ASR and diarization are insufficient in isolation; define
   speaker-attributed transcription and the stakes for accented, clinical
   dialogue.
2. **Related work:** diarization, speaker-attributed ASR/cpWER, accented ASR,
   direct audio language models, and disfluency-aware transcription.
3. **Task and datasets:** define the output schema, primary AfriSpeech setting,
   and HSLLD as transfer—not as a causal comparison.
4. **Methods:** three-condition design, models, prompts, preprocessing,
   annotation policy, and scoring/mapping details.
5. **Results:** overall performance, oracle-versus-predicted gap, direct-versus-
   modular comparison, then overlap/disfluency/domain strata.
6. **Error analysis:** show how plausible transcript text can still attach words
   to the wrong speaker or time span.
7. **Limitations and ethics:** small and confounded cohorts, API drift,
   limitations on accent claims, child/clinical data protection, and access
   constraints.
8. **Conclusion:** summarize measurable robustness gaps and the value of a
   structured evaluation protocol.

## 11. Immediate next actions

1. Move HSLLD audio from Downloads into the TalkBank corpus hierarchy.
2. Generate an MP3–CHAT pairing report and inspect a small set for alignment.
3. Recover/finalize the AfriSpeech-Dialog manifest and reference annotations.
4. Decide the single initial modular pipeline and direct audio LLM subject to
   data governance and budget.
5. Draft the annotation guide and label a small pilot subset.
6. Implement the standard output schema and scoring harness, then run the
   pilot before adding models or datasets.

## 12. Existing work to carry forward

The earlier paper, *Domain-Aware Speaker Diarization on African-Accented
English*, provides the motivation and baseline diarization evidence: a medical
penalty and improved results after accent-matched segmentation adaptation.
Treat that work as preliminary evidence. Before reusing its exact tables,
resolve the documented reproducibility issues: timestamped cohort mismatch,
missing manifests/outputs, and the disagreement between the stated
overlap-scoring protocol and code that sets `skip_overlap=True`.
