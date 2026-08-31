#!/usr/bin/env python3
"""G1 model 1: NeMo MarbleNet VAD + TitaNet-Large embeddings + NME-SC spectral
clustering, via nemo's ClusteringDiarizer. Also writes a VBx-format .lab VAD
file from the same MarbleNet VAD output, for reuse by run_g1_vbx.sh.

Must run inside the diar_g1 conda env (torch+cu118, nemo_toolkit 2.7.3).
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--nemo-cache", required=True, help="dir for NEMO_CACHE_DIR (must be under /dev/shm)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--result-json", default=None,
                     help="write the result dict here too (NeMo's logger writes INFO lines to "
                          "stdout, so stdout alone is not reliably parseable as pure JSON)")
    args = ap.parse_args()

    os.environ["NEMO_CACHE_DIR"] = args.nemo_cache
    os.makedirs(args.out_dir, exist_ok=True)
    work_dir = os.path.join(args.out_dir, "_work")
    os.makedirs(work_dir, exist_ok=True)

    import torch
    from omegaconf import OmegaConf
    from nemo.collections.asr.models import ClusteringDiarizer

    uri = os.path.splitext(os.path.basename(args.wav))[0]

    if args.device == "cuda" and not torch.cuda.is_available():
        print("ERROR: --device cuda requested but torch.cuda.is_available() is False", file=sys.stderr)
        sys.exit(1)

    manifest_path = os.path.join(work_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        f.write(json.dumps({
            "audio_filepath": os.path.abspath(args.wav),
            "offset": 0,
            "duration": None,
            "label": "infer",
            "text": "-",
            "num_speakers": None,
            "rttm_filepath": None,
            "uem_filepath": None,
        }) + "\n")

    config = OmegaConf.create({
        "diarizer": {
            "manifest_filepath": manifest_path,
            "out_dir": work_dir,
            "oracle_vad": False,
            "collar": 0.25,
            "ignore_overlap": True,
            "vad": {
                "model_path": "vad_marblenet",
                "parameters": {
                    "window_length_in_sec": 0.15,
                    "shift_length_in_sec": 0.01,
                    "smoothing": "median",
                    "overlap": 0.5,
                    "onset": 0.1,
                    "offset": 0.1,
                    "pad_onset": 0.1,
                    "pad_offset": 0,
                    "min_duration_on": 0,
                    "min_duration_off": 0.2,
                    "filter_speech_first": True,
                },
            },
            "speaker_embeddings": {
                "model_path": "titanet_large",
                "parameters": {
                    "window_length_in_sec": [1.5, 1.25, 1.0, 0.75, 0.5],
                    "shift_length_in_sec": [0.75, 0.625, 0.5, 0.375, 0.25],
                    "multiscale_weights": [1, 1, 1, 1, 1],
                    "save_embeddings": False,
                },
            },
            "clustering": {
                "parameters": {
                    "oracle_num_speakers": False,
                    "max_num_speakers": 8,
                    "enhanced_count_thres": 80,
                    "max_rp_threshold": 0.25,
                    "sparse_search_volume": 30,
                    "maj_vote_spk_count": False,
                }
            },
        },
        "device": args.device,
        "verbose": True,
        "num_workers": 1,
        "sample_rate": 16000,
        "batch_size": 64,
    })

    t0 = time.time()
    diarizer = ClusteringDiarizer(cfg=config)
    diarizer.diarize(batch_size=64)
    elapsed = time.time() - t0

    raw_rttm_candidates = glob.glob(os.path.join(work_dir, "pred_rttms", "*.rttm"))
    if not raw_rttm_candidates:
        print("ERROR: no RTTM produced by ClusteringDiarizer", file=sys.stderr)
        sys.exit(1)
    raw_rttm = raw_rttm_candidates[0]
    raw_out = os.path.join(args.out_dir, f"{uri}.g1_nemo.raw.rttm")
    shutil.copy(raw_rttm, raw_out)

    # Convert MarbleNet VAD speech segments (offset+duration jsonl) to a
    # VBx-format .lab file (start_sec end_sec sp), for reuse by run_g1_vbx.sh.
    vad_out_json = os.path.join(work_dir, "vad_outputs", "vad_out.json")
    lab_written = False
    if os.path.isfile(vad_out_json):
        lab_path = os.path.join(args.out_dir, f"{uri}.marblenet_vad.lab")
        with open(vad_out_json) as f_in, open(lab_path, "w") as f_out:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                start = float(rec["offset"])
                end = start + float(rec["duration"])
                f_out.write(f"{start:.3f} {end:.3f} sp\n")
        lab_written = True

    result = {
        "elapsed_sec": round(elapsed, 2),
        "device": args.device,
        "raw_rttm": raw_out,
        "vad_lab_written": lab_written,
        "nemo_cache_dir": args.nemo_cache,
    }
    if args.result_json:
        with open(args.result_json, "w") as f:
            json.dump(result, f)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
