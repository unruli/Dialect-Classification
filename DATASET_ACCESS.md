# Dataset access and local preparation

This guide lets a collaborator reproduce the final four-dataset diarization
suite from the original sources. Raw data are intentionally not stored in this
repository. The final evaluation view is **95 recordings / 33.79 audio hours**:

| Dataset | Evaluation subset | Recordings | Audio hours |
| --- | --- | ---: | ---: |
| AMI | Official Mix-Headset test split | 16 | 9.06 |
| AfriSpeech-Dialog | Medical, timestamp-usable conversations | 17 | 1.77 |
| Playlogue v1 | Official participant-disjoint test split | 34 | 8.11 |
| Bangor Miami | Complete English-primary, non-Maria conversations | 28 | 14.85 |

The small files in [`dataset_metadata/`](dataset_metadata/) record the exact
selection and expected aggregate statistics. They are not substitutes for the
source datasets.

## 1. Access, licenses, and storage

Before downloading anything, read the terms at each source. In particular:

- [AMI](https://groups.inf.ed.ac.uk/ami/download/) is publicly released under
  CC BY 4.0.
- [AfriSpeech-Dialog](https://github.com/intron-innovation/AfriSpeech-Dialog)
  is obtained from the authors' official repository; verify the repository's
  current license and citation instructions before use.
- [Playlogue v1](https://huggingface.co/datasets/playlogue/playlogue-v1) is a
  gated dataset whose audio comes from CHILDES. Each user must accept the
  Playlogue conditions and [TalkBank Ground
  Rules](https://talkbank.org/0share/rules.html) independently.
- [Bangor Miami](https://talkbank.org/biling/access/Bangor/Miami.html) is
  accessed through BilingBank/TalkBank. Each user needs their own TalkBank
  account and must follow the corpus citation and usage conditions.

Do not commit or circulate raw TalkBank/CHILDES audio, CHAT transcripts,
Playlogue annotations, authentication cookies, Hugging Face tokens, or derived
turn text. Do not place restricted data in a public repository or third-party
web service. The download scripts read TalkBank cookies without echo and keep
them only in process memory or a temporary cookie jar.

Plan for at least **25 GB of free space** for source data and prepared 16-kHz
WAV files. The exact total depends on whether the full AfriSpeech and
Playlogue collections are downloaded.

## 2. Software and repository setup

Install the following command-line tools on the compute system:

- Git and Git LFS
- Python 3.10 or newer
- `ffmpeg` and `ffprobe`
- `curl`, `unzip`, and `sha256sum`

Clone this repository and initialize Git LFS:

```bash
git clone https://github.com/unruli/Dialect-Classification.git
cd Dialect-Classification
git lfs install
mkdir -p data/datasets
```

The preparation code expects source data under `data/datasets/` and writes
derived files under `data/inference_ready/`. Both locations remain outside
version control.

## 3. Download each source dataset

### 3.1 AMI Mix-Headset test set

The helper downloads the official 16 Mix-Headset WAV files and clones the
AMI diarization setup used for word-only RTTM references and UEM scoring
regions:

```bash
bash download_ami_test.sh
```

Sources:

- [Official AMI download page](https://groups.inf.ed.ac.uk/ami/download/)
- [AMI diarization setup](https://github.com/BUTSpeechFIT/AMI-diarization-setup)

Expected layout:

```text
data/datasets/ami/
├── AMI-diarization-setup/lists/test.meetings.txt
├── amicorpus/EN2002a/audio/EN2002a.Mix-Headset.wav
├── reference_rttm/EN2002a.rttm
└── uem/EN2002a.uem
```

The same pattern applies to all 16 meeting IDs in
`dataset_metadata/recording_selection.json`.

### 3.2 AfriSpeech-Dialog medical subset

The authors distribute the corpus through their official GitHub repository.
The commands below avoid downloading non-medical Git LFS objects while
retaining the metadata needed to select the medical conversations:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 \
  https://github.com/intron-innovation/AfriSpeech-Dialog.git \
  data/datasets/afrispeech_dialog
git -C data/datasets/afrispeech_dialog lfs pull --include="data/medical/**"
```

Expected official-repository layout:

```text
data/datasets/afrispeech_dialog/
├── afrispeech_dialog_v1_47.csv
└── data/medical/*.wav
```

The preparation script also supports the earlier local layout in which the
CSV and `medical/` folder are under
`data/datasets/afrispeech_dialog/dir_dataset/`. The default preparation mode
filters to medical conversations and excludes rows without a usable timed
two-speaker reference.

### 3.3 Playlogue v1 test set

First request and accept access on the [Playlogue Hugging Face
page](https://huggingface.co/datasets/playlogue/playlogue-v1), after reading
the linked CHILDES/TalkBank rules. Then download the lightweight annotations:

```bash
python3 -m pip install --user huggingface_hub
hf auth login
hf download playlogue/playlogue-v1 --repo-type dataset \
  --local-dir data/datasets/playlogue/playlogue-v1
```

Playlogue does not redistribute its source audio. Log into TalkBank in a local
Chrome browser, open a CHILDES media page successfully, and copy the **value**
of the cookie named `talkbank` from Developer Tools → Application → Cookies.
Then run:

```bash
bash download_playlogue_audio.sh
```

The prompt is hidden. By default the script downloads only the 34 source MP3s
needed by the official test split (about 0.75 GB here). To reproduce all 158
Playlogue recordings instead, run:

```bash
bash download_playlogue_audio.sh data/audio all
```

Do not paste the cookie into a shared shell history or commit it to the repo.

### 3.4 Bangor Miami English-primary subset

Register or log into [BilingBank](https://talkbank.org/biling/), then verify
that the authenticated [English media
index](https://media.talkbank.org/biling/Bangor/Miami/eng/0wav/) lists audio.
Copy the value of the browser cookie named `talkbank` and run:

```bash
bash download_bangor_miami.sh
```

The script downloads the official CHAT transcript archive but only the 28 WAV
files in the project's English-primary, non-Maria evaluation subset. It checks
file sizes and writes local checksums. Expected layout:

```text
data/datasets/bangor_miami/
├── audio/herring01.wav
├── transcripts/Miami/eng/herring/herring01.cha
└── manifests/
```

The selected 28 IDs are in `dataset_metadata/recording_selection.json` and in
`prepare_inference_datasets.py`.

## 4. Build the common inference-ready view

After all four downloads complete:

```bash
python3 prepare_inference_datasets.py \
  --afrispeech-domain medical \
  --workers 4
```

The pipeline does not alter source files. It resamples derived audio to 16-kHz
mono WAV where needed, trims Playlogue clips according to the official
metadata, converts timed references to Rich Transcription Time Marked (RTTM)
plus UEM files, and validates duration/reference boundaries.

Successful output should report:

```text
AMI: 16 recordings
AfriSpeech-Dialog: 17 recordings
Playlogue: 34 recordings
Bangor Miami English-primary: 28 recordings
Errors: 0
```

The local full manifest is `data/inference_ready/manifest.csv`. Refresh the
small path-free collaboration artifacts with:

```bash
python3 export_dataset_metadata.py
```

Compare the result with `dataset_metadata/dataset_summary.csv`. Differences
should be investigated before inference rather than silently accepted.

## 5. Known annotation decisions

- AMI uses the official full-corpus-ASR test partition and word-only RTTM
  references from the AMI diarization setup.
- AfriSpeech is medical only; the parser retains timestamp-usable two-speaker
  conversations and reports invalid/nonpositive timestamp pairs.
- Playlogue uses its official test split and official clip trims. Released
  RTTMs contain no annotated overlap, so no Playlogue overlap claim is made.
- Bangor uses 28 complete English-primary non-Maria sessions. Generic `OSE`
  intervals are excluded through UEM regions. Thirty-five trusted speaker
  tiers across 11 sessions lack timestamps and remain a documented manual
  audit/masking item before the reference is frozen.

## 6. Troubleshooting

**TalkBank index exposes zero files.** Refresh the exact media index while
logged in, confirm file links are visible, then copy a new `talkbank` cookie
from the same Chrome profile. The cookie can expire and is not the account
password.

**A downloaded TalkBank file is only a few bytes.** TalkBank media requires a
Range request. Both download scripts send `Range: bytes=0-` and reject small
placeholder responses; use the scripts rather than a plain `curl URL`.

**AfriSpeech WAVs are tiny text files.** They are probably unresolved Git LFS
pointers. Run `git lfs install` and repeat the scoped `git lfs pull` command.

**Preparation cannot find a file.** Compare the local tree to the expected
layouts above. Do not rename recording files; the official metadata and
preparation script provide the mapping.

**Interrupted download.** Rerun the relevant helper. Valid existing files are
skipped, while incomplete `.part` or undersized files are downloaded again.
