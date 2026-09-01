#!/usr/bin/env python3
"""G3-A: nvidia/diar_streaming_sortformer_4spk-v2.1 (end-to-end discriminative,
4 speaker slots, native streaming). Runs inside the same NeMo/torch+cu118 env
already proven for G1-A (the SortformerEncLabelModel class ships in
nemo_toolkit; no separate G3-A environment needed as of nemo_toolkit 2.7.3).

Uses the released high-latency (30.4s) streaming configuration for offline
evaluation, per MODEL_SELECTION_AND_INFERENCE.md: chunk_len=340,
chunk_right_context=40, fifo_len=40, spkcache_update_period=300,
spkcache_len=188 (all in the model's native 80ms frames). Attribute names
confirmed against the installed nemo.collections.asr.modules.sortformer_modules
source, not guessed from documentation prose.

CAUTION: this checkpoint's card documents AMI in its training data -- AMI
recordings in the evaluation manifest are NOT an independent test of this
model and that overlap must be reported, per the eligibility checklist.

NEVER pass a reference/oracle speaker count -- the model's 4 output slots are
used automatically; this is the required primary condition.
"""
import argparse
import json
import os
import sys
import time

HIGH_LATENCY_STREAMING_CONFIG = {
    "chunk_len": 340,
    "chunk_right_context": 40,
    "fifo_len": 40,
    "spkcache_update_period": 300,
    "spkcache_len": 188,
}


def resolve_checkpoint_revision(cache_dir, checkpoint_id):
    """Read back the exact HF Hub snapshot hash that from_pretrained()
    actually resolved and cached, rather than just recording the mutable
    "nvidia/..." name. Returns None (never raises) if it can't be found --
    a missing revision should not fail an otherwise-successful run, but the
    caller must treat None as "not recorded", not as a real pin."""
    org_repo = checkpoint_id.replace("/", "--")
    snapshots_dir = os.path.join(cache_dir, "hub", f"models--{org_repo}", "snapshots")
    try:
        entries = [e for e in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, e))]
    except OSError:
        return None
    if not entries:
        return None
    # Exactly one snapshot is expected per checkpoint in a fresh cache dir;
    # if more than one exists (e.g. a reused cache across revisions), prefer
    # the most recently modified rather than guessing.
    entries.sort(key=lambda e: os.path.getmtime(os.path.join(snapshots_dir, e)), reverse=True)
    return entries[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--out-dir", required=True, help="directory for raw output + result json")
    ap.add_argument("--nemo-cache", required=True, help="dir for NEMO_CACHE_DIR / HF_HOME (checkpoint cache)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--result-json", default=None)
    args = ap.parse_args()

    os.environ["NEMO_CACHE_DIR"] = args.nemo_cache
    os.environ["HF_HOME"] = args.nemo_cache
    os.makedirs(args.out_dir, exist_ok=True)

    import torch
    from nemo.collections.asr.models import SortformerEncLabelModel

    uri = os.path.splitext(os.path.basename(args.wav))[0]

    if args.device == "cuda" and not torch.cuda.is_available():
        print("ERROR: --device cuda requested but torch.cuda.is_available() is False", file=sys.stderr)
        sys.exit(1)

    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    checkpoint_id = "nvidia/diar_streaming_sortformer_4spk-v2.1"
    model = SortformerEncLabelModel.from_pretrained(checkpoint_id)
    model.eval()
    if args.device == "cuda":
        model = model.cuda()

    # Released high-latency (30.4s) streaming configuration, for offline
    # evaluation of complete recordings (not chunk-stitched externally).
    sm = model.sortformer_modules
    for key, value in HIGH_LATENCY_STREAMING_CONFIG.items():
        setattr(sm, key, value)
    # Validates the five values just set (non-negative ints, spkcache_len
    # large enough for n_spk, chunk_len/spkcache_update_period > 0) -- raises
    # TypeError/ValueError on an illegal combination rather than silently
    # running with a bad streaming config.
    sm._check_streaming_parameters()

    checkpoint_revision = resolve_checkpoint_revision(args.nemo_cache, checkpoint_id)
    load_elapsed = time.time() - t0

    t1 = time.time()
    raw_output = model.diarize(audio=[args.wav], batch_size=1, verbose=False)
    infer_elapsed = time.time() - t1

    peak_mem_mib = None
    if torch.cuda.is_available():
        peak_mem_mib = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)

    # raw_output preserved verbatim for the raw archive before any parsing,
    # per the "preserve native raw output" requirement.
    raw_json_path = os.path.join(args.out_dir, f"{uri}.g3a_sortformer.raw.json")
    with open(raw_json_path, "w") as f:
        json.dump({"raw_output": raw_output, "type": str(type(raw_output))}, f, indent=2, default=str)

    # Confirmed by direct inspection (90s smoke test, 2026-08-31): diarize()
    # returns List[List[str]] -- one inner list per input file, each element
    # a "start end speaker_N" string (space-separated seconds + native label).
    result = {"device": args.device, "load_elapsed_sec": round(load_elapsed, 2),
              "infer_elapsed_sec": round(infer_elapsed, 2), "peak_gpu_memory_mib": peak_mem_mib,
              "raw_output_path": raw_json_path, "streaming_config": HIGH_LATENCY_STREAMING_CONFIG,
              "checkpoint_id": checkpoint_id, "checkpoint_revision": checkpoint_revision}

    try:
        lines = raw_output[0]
        if not lines:
            raise ValueError("model returned zero segments")
        raw_rttm_path = os.path.join(args.out_dir, f"{uri}.g3a_sortformer.raw.rttm")
        with open(raw_rttm_path, "w") as f:
            for line in lines:
                start_s, end_s, speaker = line.split()
                start, end = float(start_s), float(end_s)
                f.write(f"SPEAKER {uri} 1 {start:.3f} {end - start:.3f} <NA> <NA> {speaker} <NA> <NA>\n")
        result["ok"] = True
        result["raw_rttm_path"] = raw_rttm_path
    except (IndexError, ValueError) as e:
        result["ok"] = False
        result["error"] = f"could not convert raw_output to RTTM: {e}"
    if args.result_json:
        with open(args.result_json, "w") as f:
            json.dump(result, f)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
