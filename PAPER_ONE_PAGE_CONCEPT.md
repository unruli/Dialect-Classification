# Has Speaker Diarization Actually Improved? Evaluating Five Years of Model Progress Beyond Standard Adult Speech

## Page 1 — Paper concept for external evaluation

### Framing and aim

Speaker diarization has moved from voice-activity detection, speaker embeddings, and clustering to overlap-aware neural pipelines, end-to-end speaker-activity models, and generative speech models that emit speaker labels and timestamps. Improvements on established adult benchmarks do not establish whether this architectural progress transfers to less represented conversational settings. This study evaluates four architectural families under one protocol on conventional adult meetings, African-accented medical conversations, adult–child interaction, and English-primary code-switched conversation. It asks **whether architectural progress produces gains across domains, whether those gains close transfer gaps, and which diarization failures have or have not improved**.

The task is anonymous, within-recording speaker attribution—*who spoke when*—not identification of a person across recordings. Automatic speech recognition is not an outcome: when a generative system also returns words, only its speaker labels and timing are scored.

### Evaluation datasets

| Dataset and role | Final evaluation subset | Speaker and interaction profile | Annotation-specific use |
| --- | --- | --- | --- |
| **AMI Mix-Headset** — conventional adult reference | Official test set: 16 sessions, 9.06 h | One 3-speaker and fifteen 4-speaker meetings | Overall and condition metrics; 63.78 min annotated overlap |
| **AfriSpeech-Dialog** — African-accented medical transfer | Medical only: 17 conversations, 1.77 h | All two-speaker conversations | Overall, boundary, short-turn, counting, and confusion analyses; overlap is insufficient |
| **Playlogue v1** — adult–child transfer | Official participant-disjoint test set: 34 sessions, 8.11 h | All two-speaker; 2.25 h child speech (40.71% of annotated speech) | Overall and child/adult-conditioned analyses; released RTTM has no reference overlap |
| **Bangor Miami** — English-primary code-switched transfer | 28 complete English-primary sessions, 14.85 h | Twenty-six 2-speaker, one 3-speaker, and one 4-speaker session | Overall and code-switch analyses; 98.48 min annotated overlap |

The final suite contains **95 sessions and 33.79 audio hours**. Across the three target datasets, 77 of 79 sessions (97.5%) are dyadic, reducing speaker-count confounding when comparing medical, adult–child, and code-switched speech. AMI provides a deliberately harder multiparty reference. Playlogue is retained because it gives the closest speaker-count match to AfriSpeech and Bangor; child and adult errors will be reported separately so adult speaking time does not conceal child-specific failures.

### Research questions and measures

| Research question | Purpose | Primary measures |
| --- | --- | --- |
| **RQ1. Have successive architectural shifts in speaker diarization led to improved accuracy across domains?** | Group comparison of architectural families overall and within each dataset | Diarization Error Rate (DER), Jaccard Error Rate (JER), missed speech, false alarm, speaker confusion, speaker-count accuracy, planned family contrasts, relative error reduction, and clustered confidence intervals |
| **RQ2. Have these gains closed performance gaps for unconventional and less represented domains?** | Transfer: test whether target-minus-AMI gaps become smaller for later families | Dataset-specific DER/JER, transfer gap, relative gap reduction, family-by-dataset interaction, rank correlation, and rank reversal |
| **RQ3. In which speech conditions have gains and transfer been most effective?** | Explain progress and persistent failure in overlap, boundaries, short turns, speaker counting, and speaker confusion | Condition-stratified error, boundary precision/recall/F1, onset/offset error, speaker-conditioned error, family-by-condition interaction, and overlap-specific error where supported |

RQ1 concerns the overall pattern of progress; RQ2 concerns whether progress transfers rather than merely lowering average error; RQ3 identifies where gains and remaining failures occur. Overlap-specific claims will be restricted to AMI and Bangor because their references explicitly contain overlap. Playlogue will support child/adult-conditioned, boundary, short-turn, counting, and confusion analyses.

### Architecture taxonomy and selected systems

The comparison strata are grounded in published reviews rather than inferred from download counts or release dates. [Park et al. (2022)](https://doi.org/10.1016/j.csl.2021.101317) distinguish traditional modular systems, neuralized or jointly optimized components, and fully end-to-end diarization. [Serafini et al. (2023)](https://doi.org/10.1016/j.csl.2023.101534) independently compare clustering-based, end-to-end neural diarization (EEND), and speech-separation-guided paradigms, including overlap-aware clustering variants. This study operationalizes those established architectural distinctions as G1--G3 and adds G4 as an explicit post-review extension for unified generative systems. The G labels are therefore evaluation strata, not a consensus chronology claimed by either review.

| Family | Distinct architectural shift | Core systems (two per family) |
| --- | --- | --- |
| **G1: Embedding–clustering cascade** | Explicit VAD/segmentation, speaker embeddings, and global clustering | **G1-A:** NeMo MarbleNet VAD + TitaNet-Large + NME-SC; **G1-B:** MarbleNet VAD + BUT SpeechFIT VBx |
| **G2: Neuralized overlap-aware modular** | Learned powerset/multiscale speaker activity improves local segmentation and overlap handling while clustering preserves global identity | **G2-A:** `pyannote/speaker-diarization-community-1`; **G2-B:** NeMo `diar_msdd_telephonic` v1.0.1 |
| **G3: End-to-end discriminative** | A sequence model directly predicts frame-level multispeaker activity without a separate clustering backend | **G3-A:** `nvidia/diar_streaming_sortformer_4spk-v2.1`; **G3-B:** BUT SpeechFIT DiaPer 10-attractor, non-AMI-fine-tuned checkpoint, conditional on a longest-recording memory test |
| **G4: Unified generative** | A long-context model generates time-aligned anonymous speaker tokens, often jointly with words | **G4-A:** `OpenMOSS-Team/MOSS-Transcribe-Diarize` 0.9B; **G4-B:** `microsoft/VibeVoice-ASR-HF` 8B, conditional on a 24-GB memory and parser test |

The core comparison therefore contains **eight individual systems**. G1-A,
G1-B, and G2-A are in the current 95-recording run. G2-B and G3-A require a
balanced pilot before scale-up; G3-B and G4-B remain conditional on the stated
feasibility gates; G4-A requires a parser and overlap-output pilot. Pyannote
`speaker-diarization-3.1` is retained only as an optional within-lineage version
sensitivity analysis. It must not be counted as an independent G2 replicate,
because it and Community-1 share the same model family and software lineage.

Final inclusion requires a public, versionable checkpoint; reproducible local
inference; anonymous time-aligned output; compatibility with mixed-speaker
audio; and a feasible common protocol on the complete evaluation suite. The
locked manifest has a maximum of four speakers and a maximum recording length
of 49.54 minutes. Checkpoint revision, inference code revision, license,
training-data disclosure, context limit, model-native resampling, compute use,
and output parser will be frozen before a full run. Download counts may be
reported only as an adoption tie-breaker, not as the primary scientific
criterion. Moshi remains related work because its user/system streams are
predefined rather than inferred from a mixed multiparty recording.

### Experiment and expected contribution

Every model will receive the same 16-kHz mono recordings and versioned manifest. A model-native resampling step is permitted only when required by the released checkpoint and will be logged explicitly. The primary condition is fully automatic; an oracle-speaker-count condition separates counting from assignment error where the released interface supports it. Unsupported oracle conditions will be reported as not applicable, not approximated. Outputs will be normalized to **Rich Transcription Time Marked (RTTM)** format. Complete recordings will be used when supported. Native streaming or state-preserving tokenizer chunks are permitted; arbitrary external chunking and cross-window speaker stitching are prohibited unless preregistered after the model pilot and reported as a separate model constraint.

Primary scoring uses zero collar with overlap included where annotated; a 0.25-second-collar analysis tests sensitivity to boundary tolerance. Results will be macro-averaged across recordings and accompanied by recording-clustered bootstrap intervals. Analyses will control or stratify for speaker count, overlap, turn duration/rate, speech proportion, speaker imbalance, signal quality, and dataset. Because accent, age, domain, and recording conditions are not independently manipulated, conclusions concern transfer and observed gaps—not isolated causal effects.

The contribution is a **cross-family, architecture-aware transfer audit**, not a new diarization model. The study adds: (1) one protocol spanning four architectural families, eight core systems, and four contrasting settings; (2) evidence of whether newer systems narrow gaps in African-accented medical, adult–child, and English-primary code-switched speech; (3) condition-level evidence connecting architectural shifts to overlap, boundary, short-turn, counting, and confusion failures; and (4) a reproducible distinction between open modular, end-to-end, and generative diarization systems.

---

## Page 2 — Dataset progress and experiment-to-paper roadmap

### Dataset collection and preparation status

| Dataset | Collection and preparation completed | Remaining reference action |
| --- | --- | --- |
| **AMI** | Sixteen official Mix-Headset test recordings, RTTM, and Un-partitioned Evaluation Map (UEM) files standardized and validated | Freeze version/checksums and confirm the official scoring collar convention used in comparison papers |
| **AfriSpeech-Dialog** | Forty-six usable conversations prepared; the final evaluation view selects the 17 medical conversations | Export the medical-only manifest; manually audit filtered nonpositive timestamp pairs and confirm no role labels are inferred from anonymous speaker numbers |
| **Playlogue** | Thirty-four official test recordings trimmed to released clip times, resampled, and paired with official RTTM/UEM | Preserve the official test split; document that all 158 released RTTMs contain zero overlap; verify child/adult role-conditioned scoring |
| **Bangor Miami** | Twenty-eight English-primary recordings selected and converted from timestamped CHAT tiers to RTTM/UEM; code-switch metadata retained | Manually align 35 untimed trusted tiers or mask their surrounding intervals; retain all sessions rather than discarding 39% of the subset |

All prepared audio is 16-kHz mono. Paths, durations, RTTM/UEM presence, and reference boundaries have passed automated validation with zero structural errors. The current prepared superset contains 124 recordings and 38.81 h; the locked medical-only evaluation view will contain **95 recordings and 33.79 h**. Source audio remains unchanged, and preparation is reproducible through `prepare_inference_datasets.py`.

### Experiments required before paper writing

1. **Lock datasets and scoring references.** Resolve Bangor’s 35 untimed tiers, export the AfriSpeech medical-only view, add checksums and version identifiers, and generate the final dataset-statistics and metric-eligibility tables.
2. **Finalize model selection.** Verify both candidates per family against the eligibility rules; freeze checkpoint and code revisions, inference settings, licenses, training-data disclosures, context limits, compute needs, native resampling, and parsing specifications. Keep pyannote 3.1 outside the core eight as a within-lineage sensitivity analysis.
3. **Run a balanced pilot.** Use representative two-, three-, and four-speaker sessions from every dataset to test audio limits, output parsing, overlap support, speaker-count behavior, runtime, memory, determinism, and failure logging. G3-B must additionally pass the longest 49.54-minute recording; G4-B must pass the same duration and emit schema-valid segments on a 24-GB GPU. Fix the full-recording or common-window policy before the main run.
4. **Run primary inference.** Evaluate every selected model on every recording with automatic speaker count. Preserve raw outputs, parsed RTTM, logs, runtime, peak memory, failures, and configuration hashes.
5. **Run controlled conditions.** Repeat with oracle speaker count where supported. Apply the preregistered chunk/stitch condition only if required, and include a common-condition sensitivity analysis if model context limits differ.
6. **Score overall performance.** Compute DER, JER, missed speech, false alarm, speaker confusion, speaker-count error, malformed-output rate, and failure rate using zero collar and the 0.25-second sensitivity collar.
7. **Score RQ3 conditions.** Measure short-turn bins, boundary/onset/offset errors, rapid speaker changes, speaker imbalance, and per-speaker confusion across all eligible datasets; evaluate overlap only on AMI and Bangor; evaluate child/adult errors on Playlogue; compare code-switched and non-code-switched turns within Bangor.
8. **Statistical analysis.** Macro-average by recording and dataset; calculate recording-clustered bootstrap confidence intervals; test family, dataset, and family-by-dataset effects; test family-by-condition effects; report planned family contrasts with multiplicity correction, relative error/gap reduction, model-rank correlation, and rank reversals.
9. **Robustness and error audit.** Repeat scoring under both collars, check results with and without highly imbalanced/incidental speakers, inspect a stratified sample of failures, and separate acoustic-model limitations from output-format or parser limitations.

### Analyses and artifacts needed for the final four-page paper

- **Table 1:** Research questions, purposes, and measures.
- **Table 2:** Architectural families, defining shifts, selected checkpoints, parameter/access status, and output capability.
- **Dataset table:** Sessions, hours, speaker-count distribution, child/adult share, code-switch prevalence, overlap coverage, turn statistics, and annotation limitations.
- **Main results table:** DER/JER and error components by family × dataset, with confidence intervals and transfer gaps.
- **Primary figure:** Architectural-family performance trajectory on AMI versus each target dataset.
- **Diagnostic figure or compact heatmap:** Relative gains for overlap, short turns, boundaries, counting, and confusion, with non-eligible cells marked rather than imputed.
- **Reproducibility package:** Locked manifests, RTTM/UEM references or lawful preparation instructions, parsers, scoring configuration, model versions, environment lockfile, seeds, and raw-to-result provenance.

### Completion gates

The project is ready for model selection after the reference-quality actions above. Full paper writing should begin only after: **(G1)** dataset references and metric eligibility are frozen; **(G2)** all eight core systems either pass the balanced pilot or are explicitly declared ineligible under a preregistered requirement; **(G3)** the inference matrix is complete or failures are explicitly documented; and **(G4)** RQ1–RQ3 tables, confidence intervals, and sensitivity analyses are generated. The final writing sequence is Methods and dataset table, Results, Discussion/limitations, Related Work and taxonomy justification, then the Abstract and Introduction after the central empirical claim is known.
