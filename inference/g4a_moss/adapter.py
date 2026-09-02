"""run_model.py adapter for G4-A: OpenMOSS-Team/MOSS-Transcribe-Diarize.

The four fixed 90-second domain smokes passed on CURC on 2026-09-02. Complete
recording pilots remain pending.
"""
import json
import os
import subprocess
import sys
import tempfile

SYSTEM_ID = "G4-A"
HERE = os.path.dirname(os.path.abspath(__file__))


def run_one(wav_path, raw_out_dir, uri, cache_dir, device="cuda",
            max_new_tokens=None, **_):
    runner = os.path.join(HERE, "run_g4a_moss.py")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        result_json = tf.name
    try:
        command = [sys.executable, runner,
                   "--wav", wav_path,
                   "--out-dir", raw_out_dir,
                   "--model-cache", cache_dir,
                   "--device", device,
                   "--result-json", result_json]
        # Token-budget precedence: explicit kwarg > MOSS_MAX_NEW_TOKENS env > runner default.
        # The env fallback lets a batch job raise the ceiling (long recordings) without
        # threading a new argument through run_model.py.
        if max_new_tokens is None:
            env_mnt = os.environ.get("MOSS_MAX_NEW_TOKENS")
            max_new_tokens = int(env_mnt) if env_mnt else None
        if max_new_tokens is not None:
            command.extend(["--max-new-tokens", str(max_new_tokens)])
        timeout = int(os.environ.get("MOSS_TIMEOUT", "3600"))
        proc = subprocess.run(
            command,
            capture_output=True, text=True, timeout=timeout,
        )
        if proc.returncode != 0:
            return {"ok": False, "error": f"rc={proc.returncode}: {proc.stderr[-4000:]}"}
        with open(result_json) as f:
            return json.load(f)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timed out after {timeout}s"}
    finally:
        if os.path.exists(result_json):
            os.remove(result_json)


def code_revision():
    return {
        "checkpoint": "OpenMOSS-Team/MOSS-Transcribe-Diarize",
        "inference_package_revision": "61bc29cd4120be7b5d3b761b64cd5dff57263642",
        "license": "Apache-2.0",
        "validated": True,
        "smoke_gate_passed": True,
        "validation_note": "four fixed 90-second domain smokes passed with strict RTTM QC on "
                           "CURC A100 MIG on 2026-09-02; complete-recording pilots pending",
    }
