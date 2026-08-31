"""run_model.py adapter for G4-A: OpenMOSS-Team/MOSS-Transcribe-Diarize.
UNVALIDATED -- see run_g4a_moss.py's module docstring. No environment was
built and no live test was run for this system in this export."""
import json
import os
import subprocess
import sys
import tempfile

SYSTEM_ID = "G4-A"
HERE = os.path.dirname(os.path.abspath(__file__))


def run_one(wav_path, raw_out_dir, uri, cache_dir, device="cuda", **_):
    runner = os.path.join(HERE, "run_g4a_moss.py")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        result_json = tf.name
    try:
        proc = subprocess.run(
            [sys.executable, runner,
             "--wav", wav_path,
             "--out-dir", raw_out_dir,
             "--model-cache", cache_dir,
             "--device", device,
             "--result-json", result_json],
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
        "checkpoint": "OpenMOSS-Team/MOSS-Transcribe-Diarize",
        "license": "Apache-2.0",
        "validated": False,
        "validation_note": "no GPU smoke test was run for G4-A in this export; code is written from "
                            "the published model card and is unvalidated",
    }
