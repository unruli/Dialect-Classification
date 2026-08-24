# Lightweight study metadata

This directory is the redistribution-safe companion to the four source
datasets. It contains only the project's recording selection and aggregate or
recording-level numeric descriptors. It contains **no audio, transcript text,
speaker names, turn text, timestamps, RTTM/UEM contents, absolute compute
paths, credentials, or source archives**.

- `final_evaluation_manifest.csv`: path-free metadata for the 95-recording,
  medical-only evaluation view.
- `dataset_summary.csv`: dataset-level counts, hours, speaker-count
  distributions, speech, and overlap totals.
- `recording_selection.json`: stable recording identifiers used by the study.

Regenerate these files after the local inference-ready manifest changes:

```bash
python3 export_dataset_metadata.py
```

These files document the study selection but do not grant rights to, or
replace access approval for, any source corpus. Each collaborator must obtain
Playlogue and Bangor/CHILDES materials through their own approved accounts and
follow the source licenses and TalkBank Ground Rules. See
[`DATASET_ACCESS.md`](../DATASET_ACCESS.md).
