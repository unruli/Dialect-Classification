#!/usr/bin/env python3
"""G4-B: microsoft/VibeVoice-ASR-HF (8B), unified transcription,
timestamping, and diarization.

Uses the model card's native Transformers API, deterministic greedy decoding,
no prompt/hotwords, and a 64,000-sample acoustic-tokenizer chunk (20 acoustic-token
hops) to reduce peak memory while retaining the released convolution state.
The optional mean-latent mode disables the acoustic VAE noise, matching the
official VibeVoice vLLM implementation's ``VIBEVOICE_USE_MEAN=1`` behavior.
The complete native text and parsed JSON-like output are archived before RTTM
conversion. Malformed or maximum-token-truncated output is a failed smoke.
"""
import argparse
import json
import math
import os
import sys
import time

DEFAULT_MAX_NEW_TOKENS = 32768
DEFAULT_TOKENIZER_CHUNK_SIZE = 64000


def _write_result(path, result):
    if path:
        with open(path, "w") as handle:
            json.dump(result, handle)
    print(json.dumps(result))


def _eos_ids(model, processor):
    value = getattr(model.generation_config, "eos_token_id", None)
    if value is None:
        value = getattr(processor.tokenizer, "eos_token_id", None)
    if value is None:
        return set()
    if isinstance(value, int):
        return {value}
    return {int(item) for item in value}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-cache", required=True)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--tokenizer-chunk-size", type=int, default=DEFAULT_TOKENIZER_CHUNK_SIZE)
    parser.add_argument(
        "--acoustic-latent-mode",
        choices=("sample", "mean"),
        default="sample",
        help="sample the released acoustic VAE, or use its deterministic mean latent",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--result-json", default=None)
    args = parser.parse_args()

    if args.tokenizer_chunk_size <= 0 or args.tokenizer_chunk_size % 3200:
        parser.error("--tokenizer-chunk-size must be a positive multiple of 3200")

    os.environ["HF_HOME"] = args.model_cache
    os.makedirs(args.out_dir, exist_ok=True)
    uri = os.path.splitext(os.path.basename(args.wav))[0]

    import torch
    from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration

    if args.device == "cuda" and not torch.cuda.is_available():
        print("ERROR: --device cuda requested but torch.cuda.is_available() is False", file=sys.stderr)
        raise SystemExit(1)

    model_id = "microsoft/VibeVoice-ASR-HF"
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    load_started = time.time()
    processor = AutoProcessor.from_pretrained(model_id)
    if args.device == "cuda":
        model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
            model_id,
            dtype=dtype,
            device_map="auto",
            # Transformers 5.6 cannot dispatch the nested VibeVoice acoustic
            # tokenizer through SDPA. Eager is its supported fallback.
            attn_implementation="eager",
        )
    else:
        model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
            model_id,
            dtype=dtype,
            attn_implementation="eager",
        ).to("cpu")
    model.eval()
    load_elapsed = time.time() - load_started

    acoustic_config = model.config.acoustic_tokenizer_encoder_config
    original_vae_std = float(acoustic_config.vae_std)
    if args.acoustic_latent_mode == "mean":
        # The Transformers composite ASR model does not expose the acoustic
        # tokenizer's public sample=False switch. Its get_audio_features()
        # multiplies both random terms by this configured standard deviation,
        # so setting it to zero is exactly the mean-latent path. Microsoft
        # exposes the same inference choice as VIBEVOICE_USE_MEAN=1 in its
        # official vLLM implementation.
        acoustic_config.vae_std = 0.0

    device_map = getattr(model, "hf_device_map", None)
    if args.device == "cuda" and device_map:
        placements = {str(value) for value in device_map.values()}
        if "cpu" in placements or "disk" in placements:
            result = {
                "ok": False,
                "error": f"model was offloaded outside the assigned GPU: {device_map}",
                "checkpoint_id": model_id,
            }
            _write_result(args.result_json, result)
            return

    checkpoint_revision = getattr(model.config, "_commit_hash", None)
    model_device = model.device

    infer_started = time.time()
    inputs = processor.apply_transcription_request(audio=args.wav).to(model_device, dtype)
    # Loading a large checkpoint can consume framework RNG state. Reset here,
    # immediately before the first forward pass where the acoustic tokenizer
    # samples its VAE latent, so a recorded seed actually reproduces that
    # sample. Mean mode remains independent of the seed.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            acoustic_tokenizer_chunk_size=args.tokenizer_chunk_size,
        )
    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    infer_elapsed = time.time() - infer_started

    raw_text = processor.decode(generated_ids)[0]
    generated_tokens = int(generated_ids.shape[-1])
    last_token = int(generated_ids[0, -1].item()) if generated_tokens else None
    hit_limit = generated_tokens >= args.max_new_tokens
    ended_with_eos = last_token in _eos_ids(model, processor)
    truncated = hit_limit and not ended_with_eos

    raw_text_path = os.path.join(args.out_dir, f"{uri}.g4b_vibevoice.raw.txt")
    parsed_path = os.path.join(args.out_dir, f"{uri}.g4b_vibevoice.parsed.json")
    with open(raw_text_path, "w") as handle:
        handle.write(raw_text)

    peak_mem_mib = None
    if torch.cuda.is_available():
        peak_mem_mib = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)

    result = {
        "device": str(model_device),
        "device_map": device_map,
        "load_elapsed_sec": round(load_elapsed, 2),
        "infer_elapsed_sec": round(infer_elapsed, 2),
        "peak_gpu_memory_mib": peak_mem_mib,
        "raw_text_path": raw_text_path,
        "parsed_output_path": parsed_path,
        "checkpoint_id": model_id,
        "checkpoint_revision": checkpoint_revision,
        "max_new_tokens": args.max_new_tokens,
        "generated_tokens": generated_tokens,
        "tokenizer_chunk_size": args.tokenizer_chunk_size,
        "acoustic_latent_mode": args.acoustic_latent_mode,
        "original_acoustic_vae_std": original_vae_std,
        "effective_acoustic_vae_std": float(acoustic_config.vae_std),
        "seed": args.seed,
        "truncated": truncated,
    }

    if truncated:
        with open(parsed_path, "w") as handle:
            json.dump(
                {"parse_error": "not attempted: generation reached max_new_tokens without EOS"},
                handle,
                indent=2,
            )
        result.update(ok=False, error="generation reached max_new_tokens without EOS")
        _write_result(args.result_json, result)
        return

    # Preserve the native generation before invoking the library parser. The
    # released parser currently lets JSONDecodeError escape for malformed
    # model output, even though its documentation says parsing failures return
    # the original string. Such output remains a failed smoke by protocol.
    try:
        parsed = processor.decode(generated_ids, return_format="parsed")[0]
        parsed_artifact = parsed
    except Exception as exc:
        parsed = None
        result["parse_error"] = f"{type(exc).__name__}: {exc}"
        parsed_artifact = {"parse_error": result["parse_error"]}
    with open(parsed_path, "w") as handle:
        json.dump(parsed_artifact, handle, indent=2, default=str)

    if parsed is None:
        result.update(ok=False, error=f"processor parser failed: {result['parse_error']}")
        _write_result(args.result_json, result)
        return
    if not isinstance(parsed, list) or not parsed:
        result.update(ok=False, error="processor did not return a non-empty parsed segment list")
        _write_result(args.result_json, result)
        return

    segments = []
    try:
        for index, segment in enumerate(parsed):
            if not isinstance(segment, dict):
                raise ValueError(f"segment {index} is not an object")
            start = float(segment["Start"])
            end = float(segment["End"])
            speaker = str(segment["Speaker"])
            if not math.isfinite(start) or not math.isfinite(end):
                raise ValueError(f"segment {index} has a non-finite timestamp")
            if start < 0 or end <= start:
                raise ValueError(f"segment {index} has invalid interval {start}..{end}")
            segments.append((start, end - start, speaker))
    except (KeyError, TypeError, ValueError) as exc:
        result.update(ok=False, error=f"malformed parsed output: {exc}")
        _write_result(args.result_json, result)
        return

    raw_rttm_path = os.path.join(args.out_dir, f"{uri}.g4b_vibevoice.raw.rttm")
    with open(raw_rttm_path, "w") as handle:
        for start, duration, speaker in segments:
            handle.write(
                f"SPEAKER {uri} 1 {start:.3f} {duration:.3f} "
                f"<NA> <NA> {speaker} <NA> <NA>\n"
            )

    result.update(ok=True, raw_rttm_path=raw_rttm_path, n_segments_parsed=len(segments))
    _write_result(args.result_json, result)


if __name__ == "__main__":
    main()
