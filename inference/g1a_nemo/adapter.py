"""run_model.py adapter for G1-A: NeMo MarbleNet VAD + TitaNet-Large + NME-SC
spectral clustering. Invokes the proven run_g1a_nemo.py (byte-identical export
of diar_smoke/scripts/run_g1_nemo.py) as a subprocess using the current
interpreter, and reads its dedicated --result-json (NOT stdout -- NeMo's
logger writes INFO lines to stdout, so stdout alone is not reliable JSON).
"""
import json
import os
import subprocess
import sys
import tempfile

SYSTEM_ID = "G1-A"
HERE = os.path.dirname(os.path.abspath(__file__))


def run_one(wav_path, raw_out_dir, uri, cache_dir, device="cuda", **_):
    """Returns a dict: {ok, raw_rttm_path, elapsed_sec, device, error}."""
    runner = os.path.join(HERE, "run_g1a_nemo.py")
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
            result = json.load(f)
        return {
            "ok": True,
            "raw_rttm_path": result["raw_rttm"],
            "elapsed_sec": result["elapsed_sec"],
            "device": result["device"],
            "vad_lab_path": None,  # written alongside raw_rttm by run_g1a_nemo.py if present
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timed out after 1800s"}
    finally:
        if os.path.exists(result_json):
            os.remove(result_json)


def code_revision():
    return {"model_checkpoints": ["vad_marblenet", "titanet_large"],
            "note": "revisions logged by run_g1a_nemo.py's --nemo-cache directory layout"}
