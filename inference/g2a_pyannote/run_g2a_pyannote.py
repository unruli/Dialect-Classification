#!/usr/bin/env python3
"""G2: pyannote/speaker-diarization-3.1 and pyannote/speaker-diarization-community-1,
via pyannote.audio 4.0.7 Pipeline API. CPU-only fallback (documented reason: driver
535.309.01 caps CUDA at 12.2; pyannote-audio 4.0.7 requires torch>=2.8.0, and no
torch>=2.8.0 wheel is published for any CUDA index <=12.2 -- earliest is cu126).

Must run inside the diar_g2 conda env (torch 2.13.0+cu130, pyannote-audio 4.0.7).
Requires a Hugging Face token from an account that has accepted both models' gate terms.
"""
import argparse
import json
import os
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--checkpoint", required=True,
                     choices=["pyannote/speaker-diarization-3.1", "pyannote/speaker-diarization-community-1"])
    ap.add_argument("--out-rttm", required=True)
    ap.add_argument("--hf-home", required=True, help="dir for HF_HOME (must be under /dev/shm)")
    args = ap.parse_args()

    os.environ["HF_HOME"] = args.hf_home
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(args.hf_home, "hub"))

    import torch
    from pyannote.audio import Pipeline

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN env var not set", file=sys.stderr)
        sys.exit(1)

    t0 = time.time()
    try:
        pipeline = Pipeline.from_pretrained(args.checkpoint, token=token)
    except Exception as e:
        print(json.dumps({"stage": "load_pipeline", "checkpoint": args.checkpoint, "error": str(e)}), file=sys.stderr)
        sys.exit(1)
    load_elapsed = time.time() - t0

    if pipeline is None:
        print(json.dumps({"stage": "load_pipeline", "checkpoint": args.checkpoint,
                           "error": "Pipeline.from_pretrained returned None (gated/no access or terms not accepted)"}),
              file=sys.stderr)
        sys.exit(1)

    pipeline.to(torch.device("cpu"))

    t1 = time.time()
    try:
        diarization = pipeline(args.wav)
    except Exception as e:
        print(json.dumps({"stage": "inference", "checkpoint": args.checkpoint, "error": str(e)}), file=sys.stderr)
        sys.exit(1)
    infer_elapsed = time.time() - t1

    # pyannote.audio 4.x wraps the result in DiarizeOutput (.speaker_diarization
    # holds the pyannote.core.Annotation); older-style pipelines return the
    # Annotation directly. Handle both.
    annotation = getattr(diarization, "speaker_diarization", diarization)

    uri = os.path.splitext(os.path.basename(args.wav))[0]
    with open(args.out_rttm, "w") as f:
        annotation.write_rttm(f)

    # torch.cuda.is_available() has been observed to report inconsistent
    # values in this env after other libraries touch CUDA state; use an
    # actual allocation attempt as ground truth instead.
    try:
        torch.zeros(1).cuda()
        cuda_actually_usable = True
    except Exception:
        cuda_actually_usable = False

    result = {
        "checkpoint": args.checkpoint,
        "device": "cpu",
        "load_elapsed_sec": round(load_elapsed, 2),
        "infer_elapsed_sec": round(infer_elapsed, 2),
        "raw_rttm": args.out_rttm,
        "torch_version": torch.__version__,
        "cuda_actually_usable": cuda_actually_usable,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
