#!/usr/bin/env python3
"""Common CLI for the architecture-audit inference runners.

    python inference/run_model.py \
      --system G3-A \
      --path-manifest /local/path/to/inference_ready/manifest.csv \
      --selection-manifest dataset_metadata/final_evaluation_manifest.csv \
      --output-dir /local/path/to/runs/architecture_audit/G3-A \
      --pilot   # or --full / --validate-only

Run this under the conda/venv environment appropriate for the selected
--system (see that system's ENVIRONMENT.md) -- run_model.py itself has no
heavy ML dependencies; it only imports a system's adapter module (and that
adapter's own deps) once a system is actually selected.

No machine-specific paths, tokens, audio, or checkpoints are hardcoded here.
"""
import argparse
import importlib.util
import json
import os
import shutil
import sys
import threading
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from common.manifest import join_manifests, ManifestError  # noqa: E402
from common import rttm_tools  # noqa: E402
from common import provenance  # noqa: E402
from common.audio import trim_wav  # noqa: E402

# Fixed pilot recordings -- same four IDs used across every system's pilot,
# per MODEL_SELECTION_AND_INFERENCE.md's "Pilot recordings" table, so all
# model pilots cover the same transfer and speaker-count conditions.
PILOT_RECORDING_IDS = [
    "5129fd8c-7b8c-4d05-a03a-196bcae4deff",  # AfriSpeech-Dialog, 2spk, African-accented medical
    "ew_42pc_22148",                          # Playlogue, 2spk, adult-child
    "EN2002a",                                # AMI, 4spk, conventional multiparty meeting
    "sastre03",                               # Bangor Miami, 3spk, code-switched/long-form
]

SYSTEM_DIRS = {
    "G1-A": "g1a_nemo",
    "G1-B": "g1b_vbx",
    "G2-A": "g2a_pyannote",
    "G2-B": "g2b_msdd",
    "G3-A": "g3a_sortformer",
    "G3-B": "g3b_diaper",
    "G4-A": "g4a_moss",
    "G4-B": "g4b_vibevoice",
}


def load_adapter(system_id):
    subdir = SYSTEM_DIRS.get(system_id)
    if subdir is None:
        raise SystemExit(f"unknown --system {system_id!r}; choices: {sorted(SYSTEM_DIRS)}")
    adapter_path = os.path.join(HERE, subdir, "adapter.py")
    if not os.path.isfile(adapter_path):
        raise SystemExit(
            f"{system_id} is not implemented yet (no {subdir}/adapter.py). "
            f"See {subdir}/README.md if present for status."
        )
    spec = importlib.util.spec_from_file_location(f"{subdir}_adapter", adapter_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GPUMemSampler:
    """Best-effort peak-GPU-memory sampler via nvidia-smi polling, used as a
    fallback for adapters (e.g. subprocess-isolated ones) that don't report
    torch.cuda.max_memory_allocated() themselves. Whole-GPU memory.used, so
    it is an upper bound on this process's own usage, not an exact figure --
    treated as a fallback, never overriding an adapter-reported figure."""

    def __init__(self, interval_sec=0.5):
        self.interval_sec = interval_sec
        self.peak_mib = 0
        self._stop = threading.Event()
        self._thread = None

    def _poll(self):
        while not self._stop.is_set():
            snap = provenance.nvidia_smi_snapshot()
            if snap:
                try:
                    used_mib = int(snap["gpu_query"].split(",")[2].strip().split()[0])
                    self.peak_mib = max(self.peak_mib, used_mib)
                except (IndexError, ValueError):
                    pass
            self._stop.wait(self.interval_sec)

    def __enter__(self):
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)


def output_paths(output_dir, dataset, recording_id, system_id):
    subdir_key = system_id.lower().replace("-", "")
    raw_dir = os.path.join(output_dir, "raw", dataset)
    rttm_dir = os.path.join(output_dir, "rttm", dataset)
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(rttm_dir, exist_ok=True)
    rttm_path = os.path.join(rttm_dir, f"{recording_id}.rttm")
    return raw_dir, rttm_dir, rttm_path


def is_already_done(rttm_path, run_manifest, dataset, recording_id, source_duration_sec):
    """Resume-safe skip check: only skip if the normalized RTTM validates AND
    the last recorded status for this recording in run_manifest is success."""
    if not os.path.isfile(rttm_path):
        return False
    if not rttm_tools.is_valid_normalized_rttm(rttm_path, source_duration_sec):
        return False
    key = f"{dataset}/{recording_id}"
    return run_manifest.get("recordings", {}).get(key, {}).get("status") == "success"


def load_run_manifest(path):
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return {"recordings": {}, "counts": {"success": 0, "failure": 0, "malformed": 0, "truncated": 0}}


def save_run_manifest(path, manifest):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    os.replace(tmp, path)


def append_failure(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def process_one(adapter, system_id, rec, output_dir, cache_dir, config, device,
                 vad_lab_lookup, vbx_repo_dir, hf_home, work_dir):
    dataset, recording_id, wav_path = rec.dataset, rec.recording_id, rec.audio_path
    raw_dir, rttm_dir, rttm_path = output_paths(output_dir, dataset, recording_id, system_id)
    uri = recording_id

    kwargs = dict(
        wav_path=wav_path, raw_out_dir=raw_dir, uri=uri, cache_dir=cache_dir,
        device=device, vad_lab_path=vad_lab_lookup.get((dataset, recording_id)),
        vbx_repo_dir=vbx_repo_dir, work_dir=os.path.join(work_dir, f"{dataset}__{recording_id}"),
        hf_home=hf_home,
    )

    sampler = GPUMemSampler()
    t0 = time.time()
    with sampler:
        try:
            result = adapter.run_one(**kwargs)
        except Exception as e:  # never let one recording crash the whole run
            result = {"ok": False, "error": f"adapter raised {type(e).__name__}: {e}"}
    wall_elapsed = time.time() - t0

    peak_mem = result.get("peak_gpu_memory_mib") or (sampler.peak_mib or None)

    record = {
        "dataset": dataset, "recording_id": recording_id,
        "system_id": system_id, "wall_elapsed_sec": round(wall_elapsed, 2),
        "peak_gpu_memory_mib": peak_mem, "timestamp": datetime.now().astimezone().isoformat(),
    }

    if not result.get("ok"):
        record["status"] = "failure"
        record["error"] = result.get("error", "unknown failure")
        # Adapter results have already crossed the JSON subprocess boundary,
        # so their provenance fields are serializable. Keep them for failed
        # generations too (raw/parsed artifact paths, checkpoint revision,
        # token counts, truncation flags, device placement, seed, etc.).
        record.update({k: v for k, v in result.items() if k not in ("ok", "error")})
        return record, None

    raw_rttm_path = result.get("raw_rttm_path")
    if raw_rttm_path and os.path.isfile(raw_rttm_path):
        try:
            n_seg, n_spk = rttm_tools.normalize_rttm_file(
                raw_rttm_path, rttm_path, uri, source_duration_sec=rec.audio_duration_sec
            )
            record["status"] = "success"
            record["n_segments"] = n_seg
            record["n_speakers_predicted"] = n_spk
            record["n_speakers_reference"] = rec.num_speakers_ref  # recorded, never fed to the model
            record.update({k: v for k, v in result.items()
                            if k not in ("ok", "raw_rttm_path") and not isinstance(v, (dict, list))})
            return record, os.path.join(raw_dir, "*")
        except rttm_tools.RTTMValidationError as e:
            record["status"] = "malformed"
            record["error"] = f"RTTM normalization/validation failed: {e}"
            return record, None
    else:
        # G4-A-style raw-text output with no RTTM yet -- record as malformed
        # until that system's own RTTM conversion is implemented/validated.
        record["status"] = "malformed" if not result.get("ok") is False else "failure"
        record["error"] = result.get("error", "no raw_rttm_path produced by adapter")
        record.update({k: v for k, v in result.items() if k not in ("ok",)})
        return record, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", required=True, choices=sorted(SYSTEM_DIRS))
    ap.add_argument("--path-manifest", required=True, help="local, path-bearing inference manifest CSV")
    ap.add_argument("--selection-manifest", required=True, help="path-free frozen 95-recording selection CSV")
    ap.add_argument("--output-dir", required=True, help="runs/architecture_audit/<SYSTEM_ID> equivalent")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--pilot", action="store_true")
    mode.add_argument("--full", action="store_true")
    ap.add_argument("--recording-id", action="append", default=None,
                     help="restrict to this recording_id (repeatable)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--trim-seconds", type=float, default=None,
                     help="trim audio to this many seconds before inference (e.g. a single "
                          "90-second smoke test with --full --recording-id <id> --trim-seconds 90); "
                          "does not add a second full-recording pass the way --pilot does")
    ap.add_argument("--cache-dir", default=None, help="checkpoint/model cache dir (default: <output-dir>/cache)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--vbx-repo-dir", default=None, help="G1-B only: path to a cloned BUTSpeechFIT/VBx checkout")
    ap.add_argument("--vad-lab-manifest", default=None,
                     help="G1-B only: JSON {\"dataset/recording_id\": lab_path} produced by a prior G1-A run")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    for sub in ("config", "logs", "raw", "rttm"):
        os.makedirs(os.path.join(args.output_dir, sub), exist_ok=True)

    cache_dir = args.cache_dir or os.path.join(args.output_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    work_dir = os.path.join(args.output_dir, "_work")
    os.makedirs(work_dir, exist_ok=True)

    config_record = vars(args).copy()
    with open(os.path.join(args.output_dir, "config", f"run_{datetime.now():%Y%m%dT%H%M%S}.json"), "w") as f:
        json.dump(config_record, f, indent=2, default=str)

    expect_full = args.full and not args.recording_id and args.limit is None
    try:
        recordings = join_manifests(
            args.path_manifest, args.selection_manifest,
            expect_full_count=expect_full,
            recording_ids=args.recording_id if not args.pilot else (args.recording_id or PILOT_RECORDING_IDS),
            limit=args.limit,
        )
    except ManifestError as e:
        print(f"MANIFEST ERROR: {e}", file=sys.stderr)
        sys.exit(2)

    print(f"Joined {len(recordings)} recording(s) for --system {args.system}.")

    if args.validate_only:
        print("Validation OK. No inference was run (--validate-only).")
        for r in recordings[:5]:
            print(f"  {r.dataset}/{r.recording_id}  {r.audio_duration_sec:.1f}s  {r.audio_path}")
        if len(recordings) > 5:
            print(f"  ... and {len(recordings) - 5} more")
        return

    is_free, detail = provenance.gpu_is_free()
    if args.device == "cuda" and not is_free:
        print(f"GPU NOT FREE -- stopping rather than competing with:\n{detail}", file=sys.stderr)
        sys.exit(3)

    adapter = load_adapter(args.system)
    system_code_revision = adapter.code_revision() if hasattr(adapter, "code_revision") else {}

    vad_lab_lookup = {}
    if args.vad_lab_manifest and os.path.isfile(args.vad_lab_manifest):
        with open(args.vad_lab_manifest) as f:
            flat = json.load(f)
        vad_lab_lookup = {tuple(k.split("/", 1)): v for k, v in flat.items()}

    run_manifest_path = os.path.join(args.output_dir, "run_manifest.json")
    failures_path = os.path.join(args.output_dir, "failures.jsonl")
    run_manifest = load_run_manifest(run_manifest_path)
    run_manifest.setdefault("provenance", []).append(
        provenance.base_provenance(args.system, python_bin=sys.executable, command=" ".join(sys.argv))
    )
    # Static, system-level facts -- refreshed (not appended) every invocation
    # since they describe the code/checkpoint, not a specific recording.
    # code_revision() dicts also carry known caveats (e.g. G3-A's AMI
    # training-data overlap, G4-A's unvalidated status) -- kept verbatim here
    # rather than summarized, so a reader of run_manifest.json alone sees them.
    run_manifest["system_code_revision"] = system_code_revision
    run_manifest["last_run_config"] = config_record

    if args.pilot:
        passes = [("smoke_90s", 90.0), ("full", None)]
    elif args.trim_seconds is not None:
        passes = [(f"trim_{args.trim_seconds:g}s", args.trim_seconds)]
    else:
        passes = [("full", None)]

    for pass_name, trim_sec in passes:
        print(f"--- pass: {pass_name} ---")
        for rec in recordings:
            _, _, rttm_path = output_paths(args.output_dir, rec.dataset, rec.recording_id, args.system)
            key = f"{rec.dataset}/{rec.recording_id}" + ("" if pass_name == "full" else f"@{pass_name}")

            if pass_name == "full" and is_already_done(
                rttm_path, run_manifest, rec.dataset, rec.recording_id, rec.audio_duration_sec
            ):
                print(f"  SKIP (already valid): {rec.dataset}/{rec.recording_id}")
                continue

            wav_for_this_pass = rec.audio_path
            if trim_sec is not None:
                trimmed_path = os.path.join(work_dir, f"{rec.dataset}__{rec.recording_id}.smoke90s.wav")
                wav_for_this_pass = trim_wav(rec.audio_path, trimmed_path, trim_sec)

            eff_rec = rec if trim_sec is None else type(rec)(
                dataset=rec.dataset, recording_id=rec.recording_id,
                audio_path=wav_for_this_pass, audio_duration_sec=trim_sec,
                num_speakers_ref=rec.num_speakers_ref,
            )

            print(f"  RUN ({pass_name}): {rec.dataset}/{rec.recording_id} ({eff_rec.audio_duration_sec:.1f}s)")
            record, _ = process_one(
                adapter, args.system, eff_rec, args.output_dir, cache_dir, config_record,
                args.device, vad_lab_lookup, args.vbx_repo_dir, cache_dir, work_dir,
            )
            record["pass"] = pass_name

            status = record["status"]
            run_manifest["recordings"][key] = record
            if record.get("checkpoint_revision"):
                # Promoted to root for a quick top-level answer to "what
                # exact checkpoint produced this run" -- expected constant
                # across all recordings in one invocation; last-seen wins if
                # it somehow isn't (that itself would be worth investigating).
                run_manifest["model_revision"] = {
                    "checkpoint_id": record.get("checkpoint_id"),
                    "checkpoint_revision": record.get("checkpoint_revision"),
                }
            run_manifest["counts"][status] = run_manifest["counts"].get(status, 0) + 1
            if record.get("truncated"):
                run_manifest["counts"]["truncated"] = run_manifest["counts"].get("truncated", 0) + 1
            if status != "success":
                append_failure(failures_path, record)
            save_run_manifest(run_manifest_path, run_manifest)
            print(f"    -> {status}" + (f" ({record.get('error')})" if status != "success" else
                                         f" ({record.get('wall_elapsed_sec')}s, "
                                         f"peak_gpu={record.get('peak_gpu_memory_mib')}MiB)"))

    run_manifest["run_finished_at"] = datetime.now().astimezone().isoformat()
    save_run_manifest(run_manifest_path, run_manifest)

    counts = run_manifest["counts"]
    print(f"\nDone. success={counts.get('success', 0)} failure={counts.get('failure', 0)} "
          f"malformed={counts.get('malformed', 0)} truncated={counts.get('truncated', 0)}")
    print(f"run_manifest.json: {run_manifest_path}")
    print(f"failures.jsonl:    {failures_path}")


if __name__ == "__main__":
    main()
