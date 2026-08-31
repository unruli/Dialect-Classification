"""run_model.py adapter for G2-A: pyannote/speaker-diarization-community-1.

Invokes the proven run_g2a_pyannote.py (byte-identical export of
diar_smoke/scripts/run_g2_pyannote.py) as a subprocess. CPU-forced: this
project's diar_g2-equivalent env carries torch built for CUDA 13, which the
"535.309.01"-class driver cannot use -- see g2a_pyannote/ENVIRONMENT.md.
Requires HF_TOKEN in the environment and prior acceptance of the model's
gate terms on huggingface.co.
"""
import json
import os
import subprocess
import sys

SYSTEM_ID = "G2-A"
HERE = os.path.dirname(os.path.abspath(__file__))


def run_one(wav_path, raw_out_dir, uri, hf_home, checkpoint="pyannote/speaker-diarization-community-1", **_):
    runner = os.path.join(HERE, "run_g2a_pyannote.py")
    out_rttm = os.path.join(raw_out_dir, f"{uri}.g2a_pyannote.raw.rttm")
    env = dict(os.environ)
    if "HF_TOKEN" not in env:
        return {"ok": False, "error": "HF_TOKEN not set in environment; required for gated checkpoint access"}
    try:
        proc = subprocess.run(
            [sys.executable, runner,
             "--wav", wav_path,
             "--checkpoint", checkpoint,
             "--out-rttm", out_rttm,
             "--hf-home", hf_home],
            capture_output=True, text=True, timeout=3600, env=env,
        )
        if proc.returncode != 0:
            return {"ok": False, "error": f"rc={proc.returncode}: {proc.stderr[-4000:]}"}
        result = json.loads(proc.stdout.strip().splitlines()[-1])
        result["ok"] = True
        result["raw_rttm_path"] = out_rttm
        return result
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timed out after 3600s"}


def code_revision():
    return {"checkpoint": "pyannote/speaker-diarization-community-1",
            "note": "revision pinned by whatever the HF cache resolved at run time -- see run_manifest.json's hf_cache_snapshot"}
