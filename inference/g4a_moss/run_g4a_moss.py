#!/usr/bin/env python3
"""G4-A: OpenMOSS-Team/MOSS-Transcribe-Diarize (0.9B), unified generative
transcription+diarization. UNVALIDATED: written from the published model card
(transformers + trust_remote_code, AutoModelForCausalLM/AutoProcessor,
deterministic decoding) but has NOT been run -- no environment was built and
no live test was performed for this system in this export. Do not treat a
clean --help/syntax check as evidence this runs correctly.

Known integration risk (documented for whoever validates this): the model
card's own setup instructions install a cu128 torch build, which needs a
newer NVIDIA driver than the 535.309.01-class driver this project's other
GPU work targets (same class of problem already hit and confirmed for G2-A's
pyannote.audio 4.x torch>=2.8.0 floor -- see g2a_pyannote/ENVIRONMENT.md and
docs/diarization/ENVIRONMENT_REPORT.md history). Confirm CUDA compatibility
on the actual target driver before relying on GPU here.

Preserves the complete raw generated string before parsing, and detects
max-token truncation, per MODEL_SELECTION_AND_INFERENCE.md's G4-A settings
and its "preserve malformed/truncated output as failures" rule.
"""
import argparse
import json
import os
import re
import sys
import time

DEFAULT_MAX_NEW_TOKENS = 2048
LONG_AUDIO_MAX_NEW_TOKENS = 65536  # per model card, for longer recordings

# "[start_time][Sxx]transcribed speech[end_time]" per the model card.
SEGMENT_RE = re.compile(
    r"\[(?P<start>[\d.]+)\]\[S(?P<speaker>\d+)\](?P<text>.*?)\[(?P<end>[\d.]+)\]",
    re.DOTALL,
)


def parse_transcript(raw_text):
    """Best-effort parse of the '[start][Sxx]text[end]' format into
    (start, end, speaker_label, text) tuples. Returns (segments, truncated)
    where truncated is True if the text doesn't end on a well-formed
    closing tag (a signal of max-new-tokens truncation)."""
    segments = []
    for m in SEGMENT_RE.finditer(raw_text):
        segments.append((
            float(m.group("start")),
            float(m.group("end")),
            f"SPEAKER_{int(m.group('speaker')):02d}",
            m.group("text").strip(),
        ))
    # crude truncation heuristic: raw text doesn't end at a segment boundary
    truncated = bool(raw_text.strip()) and not raw_text.rstrip().endswith("]")
    return segments, truncated


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

    model_id = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, dtype="auto",
    ).to(dtype=dtype).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    load_elapsed = time.time() - t0

    # Default timestamped speaker-diarization prompt per the model card; no
    # transcript hotwords, speaker names, or dataset-specific prompting.
    inputs = processor(audio=[args.wav], return_tensors="pt").to(device)

    t1 = time.time()
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            do_sample=False,
            temperature=0.0,
            max_new_tokens=args.max_new_tokens,
        )
    raw_text = processor.batch_decode(generated, skip_special_tokens=True)[0]
    infer_elapsed = time.time() - t1

    raw_path = os.path.join(args.out_dir, f"{uri}.g4a_moss.raw.txt")
    with open(raw_path, "w") as f:
        f.write(raw_text)

    segments, truncated = parse_transcript(raw_text)

    result = {
        "ok": len(segments) > 0 and not truncated,
        "device": str(device),
        "load_elapsed_sec": round(load_elapsed, 2),
        "infer_elapsed_sec": round(infer_elapsed, 2),
        "raw_output_path": raw_path,
        "n_segments_parsed": len(segments),
        "truncated": truncated,
        "max_new_tokens": args.max_new_tokens,
    }
    if truncated:
        result["error"] = "output appears truncated at max_new_tokens (malformed, preserved as failure per protocol)"
    if not segments:
        result["error"] = result.get("error") or "no [start][Sxx]text[end] segments parsed from raw output"

    if args.result_json:
        with open(args.result_json, "w") as f:
            json.dump(result, f)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
