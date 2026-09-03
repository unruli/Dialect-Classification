"""run_model.py adapter for G4-B: microsoft/VibeVoice-ASR-HF (8B)."""
import json
import os
import subprocess
import sys
import tempfile

SYSTEM_ID = "G4-B"
HERE = os.path.dirname(os.path.abspath(__file__))


def run_one(wav_path, raw_out_dir, uri, cache_dir, device="cuda",
            max_new_tokens=None, g4b_acoustic_latent_mode="sample",
            g4b_tokenizer_chunk_size=None, **_):
    runner = os.path.join(HERE, "run_g4b_vibevoice.py")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        result_json = tf.name
    try:
        command = [sys.executable, runner,
                   "--wav", wav_path,
                   "--out-dir", raw_out_dir,
                   "--model-cache", cache_dir,
                   "--device", device,
                   "--acoustic-latent-mode", g4b_acoustic_latent_mode,
                   "--result-json", result_json]
        if max_new_tokens is not None:
            command.extend(["--max-new-tokens", str(max_new_tokens)])
        if g4b_tokenizer_chunk_size is not None:
            command.extend(["--tokenizer-chunk-size", str(g4b_tokenizer_chunk_size)])
        proc = subprocess.run(
            command,
            capture_output=True, text=True, timeout=3600,
        )
        if proc.returncode != 0:
            return {"ok": False, "error": f"rc={proc.returncode}: {proc.stderr[-4000:]}"}
        with open(result_json) as f:
            return json.load(f)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timed out after 3600s"}
    finally:
        if os.path.exists(result_json):
            os.remove(result_json)


def code_revision():
    return {
        "checkpoint": "microsoft/VibeVoice-ASR-HF",
        "license": "MIT",
        "validated": True,
        "smoke_gate_passed": False,
        "validation_note": "four fixed deterministic 90-second domain smokes ran on CURC A100 "
                           "MIG on 2026-09-02; three truncated at 4096 tokens and one failed "
                           "the official processor JSON parser; do not scale under frozen settings",
    }
