"""Environment/GPU/checkpoint provenance capture, shared across all systems.

Required by MODEL_SELECTION_AND_INFERENCE.md's output contract: run_manifest.json
must contain "system ID, model/checkpoint and code revisions, checkpoint checksum
when practical, license, command/configuration, environment package list,
GPU/driver, seed, model-native sample rate, speaker count mode, decoding
parameters, start/end time, per-file runtime, peak GPU memory, and counts of
success, failure, malformed output, and truncation."
"""
import datetime
import platform
import subprocess


def nvidia_smi_snapshot():
    """Returns (gpu_name, driver_version, cuda_version, processes) or None if
    nvidia-smi is unavailable. `processes` is the raw process-list text, so a
    preflight check can report exactly what's using the GPU."""
    try:
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=True,
        )
        procs = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=True,
        )
        return {
            "gpu_query": query.stdout.strip(),
            "processes": procs.stdout.strip(),
        }
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


def gpu_is_free(threshold_mib=100):
    """Best-effort check: is the GPU free of *other* material usage right
    now? Returns (is_free: bool, detail: str). Never assume free on error --
    per the protocol, an inability to check means stop and report, not
    proceed."""
    snap = nvidia_smi_snapshot()
    if snap is None:
        return False, "nvidia-smi unavailable -- cannot confirm GPU is free"
    procs = snap["processes"]
    if procs:
        return False, f"GPU has running compute processes:\n{procs}"
    return True, snap["gpu_query"]


def pip_freeze(python_bin):
    try:
        out = subprocess.run(
            [python_bin, "-m", "pip", "freeze"],
            capture_output=True, text=True, timeout=60, check=True,
        )
        return out.stdout
    except Exception as e:
        return f"<pip freeze failed: {e}>"


def base_provenance(system_id, python_bin=None, command=None, config=None):
    """Common provenance fields every run_manifest.json should carry,
    independent of whether the model used the GPU."""
    rec = {
        "system_id": system_id,
        "hostname": platform.node(),
        "python_version": platform.python_version(),
        "run_started_at": datetime.datetime.now().astimezone().isoformat(),
        "command": command,
        "config": config or {},
    }
    snap = nvidia_smi_snapshot()
    rec["gpu"] = snap["gpu_query"] if snap else "unavailable"
    if python_bin:
        rec["pip_freeze"] = pip_freeze(python_bin)
    return rec
