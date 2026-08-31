"""Minimal WAV trimming for the 90-second smoke-test excerpt used before a
new environment's first complete-recording pilot run. Stdlib-only (`wave`)
-- no ffmpeg/sox dependency -- since the canonical evaluation audio is
already 16-kHz mono PCM WAV.
"""
import os
import wave


def trim_wav(src_path, dst_path, duration_sec=90.0):
    with wave.open(src_path, "rb") as src:
        params = src.getparams()
        n_frames = min(int(duration_sec * params.framerate), params.nframes)
        frames = src.readframes(n_frames)
    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    with wave.open(dst_path, "wb") as dst:
        dst.setparams(params)
        dst.writeframes(frames)
    return dst_path
