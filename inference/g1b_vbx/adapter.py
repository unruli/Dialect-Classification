"""run_model.py adapter for G1-B: MarbleNet VAD + BUT SpeechFIT VBx.

Requires the caller to have already run G1-A for the same recording in this
session (or a prior one) to obtain the MarbleNet VAD .lab file -- VBx has no
VAD of its own. `vad_lab_path` must be supplied by the orchestrator.
"""
import json
import os
import subprocess

SYSTEM_ID = "G1-B"
HERE = os.path.dirname(os.path.abspath(__file__))


def run_one(wav_path, raw_out_dir, uri, vad_lab_path, vbx_repo_dir, work_dir, **_):
    if not vad_lab_path or not os.path.isfile(vad_lab_path):
        return {"ok": False, "error": "G1-B requires a MarbleNet VAD .lab file from a G1-A run; none provided/found"}

    script = os.path.join(HERE, "run_g1b_vbx.sh")
    try:
        proc = subprocess.run(
            ["bash", script, wav_path, vad_lab_path, uri, vbx_repo_dir, work_dir, raw_out_dir],
            capture_output=True, text=True, timeout=900,
        )
        if proc.returncode != 0:
            return {"ok": False, "error": f"rc={proc.returncode}: {proc.stderr[-4000:]}"}
        result = json.loads(proc.stdout.strip().splitlines()[-1])
        return result
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timed out after 900s"}


def code_revision():
    return {"upstream_repo": "https://github.com/BUTSpeechFIT/VBx", "license": "Apache-2.0"}
