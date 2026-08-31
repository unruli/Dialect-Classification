"""run_model.py adapter for G3-A: nvidia/diar_streaming_sortformer_4spk-v2.1.
Runs in-process import is avoided (subprocess isolation, matching the other
adapters) so a crash in this system never takes down the orchestrator."""
import json
import os
import subprocess
import sys
import tempfile

SYSTEM_ID = "G3-A"
HERE = os.path.dirname(os.path.abspath(__file__))


def run_one(wav_path, raw_out_dir, uri, cache_dir, device="cuda", **_):
    runner = os.path.join(HERE, "run_g3a_sortformer.py")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        result_json = tf.name
    try:
        proc = subprocess.run(
            [sys.executable, runner,
             "--wav", wav_path,
             "--out-dir", raw_out_dir,
             "--nemo-cache", cache_dir,
             "--device", device,
             "--result-json", result_json],
            capture_output=True, text=True, timeout=1800,
        )
        if proc.returncode != 0:
            return {"ok": False, "error": f"rc={proc.returncode}: {proc.stderr[-4000:]}"}
        with open(result_json) as f:
            return json.load(f)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timed out after 1800s"}
    finally:
        if os.path.exists(result_json):
            os.remove(result_json)


def code_revision():
    return {
        "checkpoint": "nvidia/diar_streaming_sortformer_4spk-v2.1",
        "license": "NVIDIA Open Model License Agreement",
        "known_caveat": "AMI Meeting Corpus is documented in this checkpoint's training data -- "
                         "AMI recordings in this evaluation are not an independent test of G3-A",
    }
