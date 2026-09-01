"""Environment/GPU/checkpoint provenance capture, shared across all systems.

Required by MODEL_SELECTION_AND_INFERENCE.md's output contract: run_manifest.json
must contain "system ID, model/checkpoint and code revisions, checkpoint checksum
when practical, license, command/configuration, environment package list,
GPU/driver, seed, model-native sample rate, speaker count mode, decoding
parameters, start/end time, per-file runtime, peak GPU memory, and counts of
success, failure, malformed output, and truncation."
"""
import datetime
import os
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
    proceed.

    "Free" means BOTH: any listed compute process, AND memory.used below
    threshold_mib (previously this only checked for a nonempty process list
    and ignored threshold_mib entirely -- a GPU showing e.g. 4000 MiB used
    with no attributable compute-app entry read as "free", which is wrong).
    """
    snap = nvidia_smi_snapshot()
    if snap is None:
        return False, "nvidia-smi unavailable -- cannot confirm GPU is free"

    # On a Slurm-managed GPU/MIG allocation, CUDA_VISIBLE_DEVICES identifies
    # the device assigned exclusively to this job.  `nvidia-smi --query-gpu`
    # may still report every *physical* GPU on the shared node and aggregate
    # memory used by other MIG tenants.  Treating that node-wide figure as
    # usage on our slice creates a false positive (for example, an otherwise
    # empty 20 GiB MIG slice can appear as 19 GiB used on its parent GPU).
    # The scheduler allocation is the correct isolation boundary here.
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "").strip()
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if slurm_job_id and visible_devices and visible_devices not in {"-1", "NoDevFiles"}:
        return True, (
            f"Slurm GPU allocation confirmed: job={slurm_job_id}, "
            f"CUDA_VISIBLE_DEVICES={visible_devices}. Physical-GPU memory/process "
            "figures may include other isolated MIG allocations and are recorded "
            f"for provenance only.\n{snap['gpu_query']}"
        )

    procs = snap["processes"]
    if procs:
        return False, f"GPU has running compute processes:\n{procs}"

    try:
        used_mib = int(snap["gpu_query"].split(",")[2].strip().split()[0])
    except (IndexError, ValueError):
        return False, f"could not parse memory.used from nvidia-smi output: {snap['gpu_query']!r}"

    if used_mib >= threshold_mib:
        return False, (
            f"GPU memory.used={used_mib}MiB >= threshold_mib={threshold_mib}MiB "
            f"(no compute-app was individually listed, but this much memory is in "
            f"use by something):\n{snap['gpu_query']}"
        )
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
