# Domain-Aware Speaker Diarization on African-Accented English

This repository contains the experiment notebooks and exported predictions used
to evaluate speaker-diarization systems on African-accented English. The work
studies whether diarization degrades in clinical dialogue relative to general
conversation, and whether light, accent-matched adaptation can reduce that
gap.

The accompanying manuscript is **“Domain-Aware Speaker Diarization on
African-Accented English”** (Chibuzor Okocha, Kelechi Ezema, and Christan
Grant; arXiv:2509.21554v1). This repository is an experiment archive, not yet
a fully reproducible training package. In particular, the manuscript’s dataset
table reports all conversations, while the checked-in scorer uses the
timestamped subset; see
[Reproducibility status](#reproducibility-status) before using the numbers in a
new submission.

## Research question

Do contemporary diarization systems work equally well for African-accented
English in general and clinical conversations?

The study evaluates commercial and open-source systems under the same scoring
protocol, decomposes diarization error into false alarm (FA), missed speech
(MISS), and speaker confusion (CONF), and evaluates a segmentation-only
adaptation of Pyannote on AfriSpeech-Countries.

## Current four-family extension

The repository now also supports a cross-domain extension that asks whether
speaker-diarization improvements transfer across conventional adult meetings,
African-accented medical dialogue, adult--child interaction, and
English-primary code-switched conversation. Its G1--G4 labels denote
study-specific architectural strata rather than universally accepted
chronological generations.

The locked extension contains 95 recordings and 33.79 audio hours. It compares
two independently motivated systems per architectural stratum:

| ID | Architectural stratum | Selected system | Status |
| --- | --- | --- | --- |
| G1-A | Embedding--clustering cascade | NeMo MarbleNet + TitaNet-Large + NME-SC | Full 95-recording run complete |
| G1-B | Embedding--clustering cascade | MarbleNet VAD + BUT SpeechFIT VBx | Full 95-recording run complete |
| G2-A | Neuralized overlap-aware modular | `pyannote/speaker-diarization-community-1` | Full 95-recording run complete |
| G2-B | Neuralized overlap-aware modular | NeMo `diar_msdd_telephonic` v1.0.1 | Pilot pending |
| G3-A | End-to-end discriminative | `nvidia/diar_streaming_sortformer_4spk-v2.1` | Full 95-recording run complete; strict RTTM QC passed |
| G3-B | End-to-end discriminative | BUT SpeechFIT DiaPer, 10-attractor non-AMI-FT checkpoint | Longest-file gate pending |
| G4-A | Unified generative | `OpenMOSS-Team/MOSS-Transcribe-Diarize` 0.9B | Four-domain 90-second smoke passed 4/4; complete pilots pending |
| G4-B | Unified generative | `microsoft/VibeVoice-ASR-HF` 8B | Deterministic four-domain smoke failed 0/4; do not scale under frozen settings |

Pyannote `speaker-diarization-3.1` may be run as a within-lineage sensitivity
analysis, but it is not counted as an independent G2 system. The complete
selection rationale, eligibility checklist, checkpoint links, common output
contract, and collaborator-ready execution prompt are in
[`MODEL_SELECTION_AND_INFERENCE.md`](MODEL_SELECTION_AND_INFERENCE.md). The
paper-facing design is in
[`PAPER_ONE_PAGE_CONCEPT.md`](PAPER_ONE_PAGE_CONCEPT.md).

## Main findings from the manuscript

- Clinical conversations had substantially higher mean DER than general
  conversations: **33.38% vs. 15.18%** across the eight paper baselines.
- The reported paired comparison was significant: two-sided paired t-test
  `t(7) = 10.67, p < 1e-4`; Wilcoxon signed-rank `W = 0, p = 0.0078`.
- The clinical penalty was driven chiefly by false alarms, then missed speech.
  This is consistent with fast turn-taking, shorter utterances, and overlap in
  the clinical role-play dialogues.
- Fine-tuning Pyannote’s segmentation model on African-accented speech lowered
  its overall DER from **21.30% to 10.65%**, but did not eliminate the
  medical–general gap (12.43 to 6.21 percentage points).

These are findings from the manuscript. The repository captures the
timestamped evaluation subset but cannot regenerate the results without the
underlying data and annotations.

## Evaluation data

The paper uses timestamped dyadic conversations from an AfriSpeech-Dialog
subset. Audio is mono, 16-bit, 48 kHz. Participants gave verbal consent and
identifiable content was removed before analysis.

| Split | Unit | Conversations / clips | Duration | Purpose |
| --- | --- | ---: | ---: | --- |
| Evaluation—medical | conversation | 20 total / 9 timestamped | 2.07 h | OSCE-style doctor–patient role plays |
| Evaluation—general | conversation | 29 total / 21 timestamped | 4.93 h | Open-topic, topic-card conversations |
| Adaptation | clip | 21,581 | 67.73 h | AfriSpeech-Countries; country-tagged read and conversational speech |

The adaptation corpus was split 80/20 by file and stratified by country
(approximately 54.18 h train and 13.55 h development). It includes clips from
Nigeria, Kenya, South Africa, North Africa, Ghana, Uganda, and Rwanda.
Natural code-switching with African languages is present in the evaluation
speech.

## Systems evaluated in the paper

| Category | System | Configuration |
| --- | --- | --- |
| Commercial | AssemblyAI | Default diarization API; no speaker-count or manual hints |
| Commercial | Deepgram | Default diarization API; no speaker-count or manual hints |
| Commercial | Soniox | Default diarization API; no speaker-count or manual hints |
| Commercial | Reverb / Rev.ai | Default diarization API; no speaker-count or manual hints |
| Open source | Pyannote | Released `pyannote/speaker-diarization-3.1` pipeline |
| Open source | CAM++ | Released checkpoint and default inference pipeline |
| Open source | Sortformer | Released `nvidia/diar_sortformer_4spk-v1` checkpoint |
| Open source | TitaNet-L | Released checkpoint and default inference pipeline |

The open systems were run with released weights and no additional training,
except for the explicit Pyannote adaptation experiment below.

## Metric and scoring protocol

The primary metric is Diarization Error Rate (DER):

`DER = (FA + MISS + CONF) / total reference speaker time`

The paper specifies Hungarian one-to-one speaker mapping, a zero-second collar,
and scoring of overlapped speech. Audio preprocessing and RTTM formatting were
held fixed across systems. Per-recording scores are aggregated to absolute DER
for each cohort.

The exported CSVs also include component summaries in
[`strict_der_for_all_domain.csv`](strict_der_for_all_domain.csv),
[`strict_for_medical.csv`](strict_for_medical.csv), and
[`strict_for_non_medical.csv`](strict_for_non_medical.csv). See the
reproducibility note below: the current post-processing notebook must be
corrected and version-pinned before these should be treated as an exact rerun of
the manuscript’s stated protocol.

## Paper results

### Baseline DER

Lower is better. Values below are percentages from Table 7 of the manuscript.

| Model | Overall | Medical | General |
| --- | ---: | ---: | ---: |
| AssemblyAI | **12.72** | **25.66** | **9.98** |
| Deepgram | 14.21 | 29.35 | 10.92 |
| TitaNet-L | 16.27 | 34.64 | 12.28 |
| CAM++ | 19.58 | 34.63 | 16.30 |
| Soniox | 20.05 | 42.16 | 15.24 |
| Reverb | 20.23 | 31.46 | 17.68 |
| Pyannote | 21.30 | 31.46 | 19.03 |
| Sortformer | 26.82 | 39.69 | 24.04 |

### Pyannote adaptation

Only the segmentation model (`pyannote/segmentation-3.0`) was fine-tuned;
speaker embeddings were frozen. Training used frame-level BCE on 10-second
chunks, up to three speakers per chunk and two speakers per frame; Adam with
learning rate `1e-4`; batch size 1 with gradient accumulation of 4; ten epochs;
gradient clipping at 1.0; and early stopping with patience 3 on validation
loss. No augmentation was used. The best validation-loss checkpoint was scored
within the UEM on a single CUDA-capable GPU.

| Domain | Base DER | Fine-tuned DER |
| --- | ---: | ---: |
| Medical | 31.46% | 15.73% |
| General | 19.03% | 9.52% |
| Overall | 21.30% | 10.65% |

### Error and conversation profile

Across paper baselines, medical speech shifted toward FA and MISS: medical
mean DER was 33.4% (FA 18.0, MISS 9.0, CONF 8.2 percentage points), compared
with 16.5% for general speech (FA 7.0, MISS 3.2, CONF 6.3).

Medical conversations also had 78.6 ± 38.3 turns per conversation and a mean
utterance duration of 3.31 ± 1.32 seconds, versus 30.55 ± 20.3 turns and
30.71 ± 19.67 seconds for general conversations. The reported overlap ratio was
slightly higher in medical conversations (0.14% vs. 0.10%).

## Repository layout

| Path | Purpose |
| --- | --- |
| [`Afrispeech_pyannote.ipynb`](Afrispeech_pyannote.ipynb) | Pyannote inference and per-file predictions |
| [`Afrispeech_sortformer.ipynb`](Afrispeech_sortformer.ipynb) | Sortformer inference and scoring |
| [`Afrispeech CAM++.ipynb`](Afrispeech%20CAM%2B%2B.ipynb) | CAM++ inference workflow |
| [`Afrispeech_deepgram.ipynb`](Afrispeech_deepgram.ipynb) | Deepgram API workflow |
| [`Afrispeech_soniox.ipynb`](Afrispeech_soniox.ipynb) | Soniox API workflow |
| [`Afrispeech_reverb.ipynb`](Afrispeech_reverb.ipynb) | Reverb placeholder (empty) |
| [`Afrispeech_Nemo.ipynb`](Afrispeech_Nemo.ipynb) | NeMo placeholder (empty) |
| [`Afrispeech_titanet.ipynb`](Afrispeech_titanet.ipynb) | TitaNet placeholder (empty) |
| [`process_result.ipynb`](process_result.ipynb) | Aligns model outputs and computes aggregate/domain DER |
| [`Diarization results/`](Diarization%20results/) | Per-system outputs and aggregate result snapshot |
| [`PAPER_ONE_PAGE_CONCEPT.md`](PAPER_ONE_PAGE_CONCEPT.md) | Four-family cross-domain paper design and experiment roadmap |
| [`MODEL_SELECTION_AND_INFERENCE.md`](MODEL_SELECTION_AND_INFERENCE.md) | Core eight-system decision, strict eligibility gates, and collaborator inference brief |
| [`inference/README.md`](inference/README.md) | Inference code status, stable CLI contract, collaborator runbook, and compute-side export checklist |
| [`dataset_metadata/final_evaluation_manifest.csv`](dataset_metadata/final_evaluation_manifest.csv) | Path-free metadata for the locked 95-recording evaluation view |
| [`environment.yml`](environment.yml) | Conda environment definition |
| [`requirements.txt`](requirements.txt) | Python dependency list |

## Checked-in result snapshot

[`Diarization results/absolute_der_all_domains_and_models.csv`](Diarization%20results/absolute_der_all_domains_and_models.csv)
is the current aggregate artifact. It covers **seven** systems and the
post-processing notebook output shows a **30-file timestamped cohort** (9
medical, 21 non-medical). The public dataset card also lists 20 medical and 29
general conversations overall; the manuscript must distinguish those totals
from the timestamped evaluation subset. TitaNet and fine-tuned Pyannote results
are absent from the repository snapshot.

| Model | Overall DER | Medical DER | Non-medical DER |
| --- | ---: | ---: | ---: |
| AssemblyAI | **12.72%** | **25.68%** | **9.91%** |
| Deepgram | 14.21% | 29.35% | 10.92% |
| CAM++ | 19.58% | 34.64% | 16.30% |
| Soniox | 20.05% | 42.16% | 15.24% |
| Reverb | 20.23% | 31.46% | 17.68% |
| Pyannote | 21.30% | 31.46% | 19.09% |
| Sortformer | 26.82% | 39.64% | 24.04% |

## Setup and rerunning notebooks

Create the supplied environment:

```bash
conda env create -f environment.yml
conda activate diarization_env
```

Or install the base Python dependencies:

```bash
pip install -r requirements.txt
```

Then open the notebook for the system being evaluated. API notebooks require
that provider’s credentials; the Pyannote notebook prompts for a Hugging Face
access token. Do not commit credentials or raw audio containing sensitive
speech. Run [`process_result.ipynb`](process_result.ipynb) only after all model
outputs have the same `audio_id` cohort and use consistent `ref_segments` and
`pred_segments` formats.

## Reproducibility status

**What is available:** model-specific notebooks, seven exported baseline
outputs, and aggregate DER/component CSVs.

**What is not currently available:** the evaluation audio and annotations,
the AfriSpeech-Countries training manifest/split, the fine-tuning code and
checkpoint, TitaNet output, pinned API/model versions, and a machine-readable
experiment configuration. Consequently, an independent reader cannot exactly
reproduce the paper from this repository alone.

There is also a protocol discrepancy to resolve before resubmission:
[`process_result.ipynb`](process_result.ipynb) contains a helper named
`compute_strict_der_for_dataset` that sets `skip_overlap=True`, whereas the
manuscript states that overlap is scored. The manuscript protocol should be
made authoritative, the scoring code should match it, and the final tables
should be regenerated from a versioned manifest.

## Recommended resubmission work

1. Release a de-identified evaluation manifest (or a controlled-access recipe),
   RTTMs/UEMs, cohort IDs, and exact medical/general split.
2. Provide a single parameterized runner that produces normalized RTTM outputs
   for every system, followed by one version-pinned scoring script.
3. Re-run all systems on the same 49-conversation cohort, including TitaNet,
   and record API date/version, checkpoint revision, hardware, and random seed.
4. Make overlap handling explicit and add a sensitivity table: overlap scored
   versus excluded, plus collar settings. This directly addresses the central
   clinical-conversation claim.
5. Release the Pyannote adaptation manifest, configuration, checkpoint, and
   per-recording before/after scores. Report confidence intervals or paired
   effect sizes alongside p-values.
6. Strengthen causal evidence by matching clinical and general conversations on
   turn length and overlap, or modelling those variables directly. Add a
   non-African clinical comparison if data access permits.
7. Separate reproducible empirical observations from vendor-training-data
   descriptions, which may change and are often incompletely disclosed.

## Limitations

The evaluation is small and the domains differ in more than clinical content:
turn structure, duration, speaker behavior, and overlap may confound the domain
effect. The adaptation data are not clinical and were not augmented. There is
no matched non-African clinical control in the reported study. Results from
commercial APIs are inherently time- and version-dependent.

## Citation

```bibtex
@article{okocha2025domainaware,
  title={Domain-Aware Speaker Diarization On African-Accented English},
  author={Okocha, Chibuzor and Ezema, Kelechi and Grant, Christan},
  journal={arXiv preprint arXiv:2509.21554},
  year={2025}
}
```

## License

See [`LICENSE`](LICENSE). Dataset access and use remain subject to the
AfriSpeech dataset terms and participant-protection requirements.
