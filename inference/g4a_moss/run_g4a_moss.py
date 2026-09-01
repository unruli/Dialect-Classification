#!/usr/bin/env python3
"""G4-A: OpenMOSS-Team/MOSS-Transcribe-Diarize (0.9B), unified generative
transcription+diarization. Uses the official `moss_transcribe_diarize`
helper package (https://github.com/OpenMOSS/MOSS-Transcribe-Diarize),
verbatim per its published usage example:
  https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize

STILL UNVALIDATED as of this rewrite: no environment was built and no live
test has been run for this system. It was rewritten from written-from-scratch
transformers calls to the project's own build_transcription_messages /
generate_transcription / parse_transcript helpers because the earlier version
duplicated logic the official package already provides correctly (prompt
construction, message formatting) -- rewriting it does not itself validate
it. Do not treat a clean syntax check or --help as evidence this runs.

Preserves the complete raw generated string before parsing, and detects
max-token truncation, per MODEL_SELECTION_AND_INFERENCE.md's G4-A settings
and its "preserve malformed/truncated output as failures" rule. Writes both
the native raw text and a raw RTTM (native speaker labels, not yet anonymized
-- common/rttm_tools.py anonymizes on normalization, same as every other
system in this project).
"""
import argparse
import json
import os
import sys
import time

DEFAULT_MAX_NEW_TOKENS = 2048
LONG_AUDIO_MAX_NEW_TOKENS = 65536  # per model card, for longer recordings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-cache", required=True, help="HF_HOME-equivalent cache dir")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--result-json", default=None)
    args = ap.parse_args()

    os.environ["HF_HOME"] = args.model_cache
    os.makedirs(args.out_dir, exist_ok=True)
    uri = os.path.splitext(os.path.basename(args.wav))[0]

    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
    from moss_transcribe_diarize import parse_transcript
    from moss_transcribe_diarize.inference_utils import (
        build_transcription_messages,
        generate_transcription,
        resolve_device,
    )

    model_id = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
    device = resolve_device(args.device if args.device != "cuda" else "auto")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    if args.device == "cuda" and not torch.cuda.is_available():
        print("ERROR: --device cuda requested but torch.cuda.is_available() is False", file=sys.stderr)
        sys.exit(1)

    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, dtype="auto", attn_implementation="sdpa",
    ).to(dtype=dtype).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    load_elapsed = time.time() - t0

    # Default timestamped speaker-diarization prompt via the official helper
    # -- no transcript hotwords, speaker names, or dataset-specific prompting.
    t1 = time.time()
    messages = build_transcription_messages(args.wav)
    generated = generate_transcription(
        model, processor, messages,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        device=device,
        dtype=dtype,
    )
    raw_text = generated["text"]
    infer_elapsed = time.time() - t1

    peak_mem_mib = None
    if torch.cuda.is_available():
        peak_mem_mib = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)

    raw_text_path = os.path.join(args.out_dir, f"{uri}.g4a_moss.raw.txt")
    with open(raw_text_path, "w") as f:
        f.write(raw_text)

    result = {
        "device": str(device),
        "load_elapsed_sec": round(load_elapsed, 2),
        "infer_elapsed_sec": round(infer_elapsed, 2),
        "peak_gpu_memory_mib": peak_mem_mib,
        "raw_text_path": raw_text_path,
        "checkpoint_id": model_id,
        "max_new_tokens": args.max_new_tokens,
    }

    try:
        segments = list(parse_transcript(raw_text))
    except Exception as e:
        segments = []
        result["parse_error"] = f"{type(e).__name__}: {e}"

    # Truncation heuristic: max_new_tokens reached without the generation
    # reaching a natural close (raw text doesn't end on a segment boundary).
    truncated = bool(raw_text.strip()) and not raw_text.rstrip().endswith("]")
    result["truncated"] = truncated

    if truncated or not segments:
        result["ok"] = False
        result["error"] = (
            "output appears truncated at max_new_tokens (malformed, preserved as failure per protocol)"
            if truncated else
            "no segments parsed from raw output " + (result.get("parse_error") or "")
        )
        if args.result_json:
            with open(args.result_json, "w") as f:
                json.dump(result, f)
        print(json.dumps(result))
        return

    raw_rttm_path = os.path.join(args.out_dir, f"{uri}.g4a_moss.raw.rttm")
    with open(raw_rttm_path, "w") as f:
        for seg in segments:
            duration = seg.end - seg.start
            if duration <= 0:
                continue  # preserved in raw_text_path; not written as an invalid RTTM segment
            f.write(f"SPEAKER {uri} 1 {seg.start:.3f} {duration:.3f} <NA> <NA> {seg.speaker} <NA> <NA>\n")

    result["ok"] = True
    result["raw_rttm_path"] = raw_rttm_path
    result["n_segments_parsed"] = len(segments)

    if args.result_json:
        with open(args.result_json, "w") as f:
            json.dump(result, f)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
