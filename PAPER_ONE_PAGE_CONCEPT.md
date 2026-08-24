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

### Architecture taxonomy and representative candidates

The comparison strata are grounded in published reviews rather than inferred from download counts or release dates. [Park et al. (2022)](https://doi.org/10.1016/j.csl.2021.101317) distinguish traditional modular systems, neuralized or jointly optimized components, and fully end-to-end diarization. [Serafini et al. (2023)](https://doi.org/10.1016/j.csl.2023.101534) independently compare clustering-based, end-to-end neural diarization (EEND), and speech-separation-guided paradigms, including overlap-aware clustering variants. This study operationalizes those established architectural distinctions as G1--G3 and adds G4 as an explicit post-review extension for unified generative systems. The G labels are therefore evaluation strata, not a consensus chronology claimed by either review.

| Family | Distinct architectural shift | Representative candidate for final selection |
| --- | --- | --- |
| **G1: Embedding–clustering cascade** | Explicit VAD/segmentation, speaker embeddings, and global clustering | NeMo MarbleNet + TitaNet-Large + spectral clustering |
| **G2: Neuralized overlap-aware modular** | Learned powerset/multiscale speaker activity improves local segmentation and overlap handling while clustering preserves global identity | pyannote Community-1 |
| **G3: End-to-end discriminative** | A sequence model directly predicts frame-level multispeaker activity without a separate clustering backend | NVIDIA Streaming Sortformer 4-speaker |
| **G4: Unified generative** | A long-context model generates time-aligned anonymous speaker tokens, often jointly with words | VibeVoice-ASR; final eligibility to be confirmed in the model pilot |

Final selection will require a public, versionable checkpoint; reproducible inference; anonymous time-aligned output; compatibility with mixed-speaker audio; and a feasible common protocol. Download counts may be reported only as an adoption tie-breaker, not as the primary scientific criterion. Moshi remains related work because its user/system streams are predefined rather than inferred from a mixed multiparty recording.

### Experiment and expected contribution

Every model will receive the same 16-kHz mono recordings and versioned manifest. The primary condition is fully automatic; an oracle-speaker-count condition separates counting from assignment error. Outputs will be normalized to **Rich Transcription Time Marked (RTTM)** format. Complete recordings will be used when supported; any required chunking and cross-window speaker stitching will be fixed after the model pilot and reported as a model constraint.

Primary scoring uses zero collar with overlap included where annotated; a 0.25-second-collar analysis tests sensitivity to boundary tolerance. Results will be macro-averaged across recordings and accompanied by recording-clustered bootstrap intervals. Analyses will control or stratify for speaker count, overlap, turn duration/rate, speech proportion, speaker imbalance, signal quality, and dataset. Because accent, age, domain, and recording conditions are not independently manipulated, conclusions concern transfer and observed gaps—not isolated causal effects.

The contribution is a **cross-family, architecture-aware transfer audit**, not a new diarization model. The study adds: (1) one protocol spanning four architectural families and four contrasting settings; (2) evidence of whether newer systems narrow gaps in African-accented medical, adult–child, and English-primary code-switched speech; (3) condition-level evidence connecting architectural shifts to overlap, boundary, short-turn, counting, and confusion failures; and (4) a reproducible distinction between open modular, end-to-end, and generative diarization systems.

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
2. **Finalize model selection.** Verify one candidate per family against the eligibility rules; freeze checkpoint/API versions, inference settings, licenses, context limits, compute needs, and parsing specifications. Decide whether a closed service is reported only as a supplemental sensitivity comparison.
3. **Run a balanced pilot.** Use representative two-, three-, and four-speaker sessions from every dataset to test audio limits, output parsing, overlap support, speaker-count behavior, runtime, memory, cost, determinism, and failure logging. Fix the full-recording or common-window policy before the main run.
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

The project is ready for model selection after the reference-quality actions above. Full paper writing should begin only after: **(G1)** dataset references and metric eligibility are frozen; **(G2)** all four model parsers pass the balanced pilot; **(G3)** the inference matrix is complete or failures are explicitly documented; and **(G4)** RQ1–RQ3 tables, confidence intervals, and sensitivity analyses are generated. The final writing sequence is Methods and dataset table, Results, Discussion/limitations, Related Work and taxonomy justification, then the Abstract and Introduction after the central empirical claim is known.
