# G4-B environment: microsoft/VibeVoice-ASR-HF (8B)

Use a clean Python 3.12 environment. The released Transformers-native model
requires Transformers 5.3 or newer; this audit pins 5.6.0 and uses PyTorch
2.8.0 with CUDA 12.8 wheels.

```bash
conda create -n g4b_vibevoice python=3.12 pip -y
conda activate g4b_vibevoice
python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.8.0 torchaudio==2.8.0
python -m pip install transformers==5.6.0 accelerate soundfile librosa
```

The primary condition uses:

- `microsoft/VibeVoice-ASR-HF`, BF16, eager attention;
- `do_sample=False` and no contextual prompt or hotwords;
- `acoustic_tokenizer_chunk_size=64000` (a multiple of the released
  3200-sample hop; exposed as `--tokenizer-chunk-size` by this runner);
- batch size 1 and no oracle speaker count;
- raw native text plus the processor's parsed output retained before RTTM
  normalization.

Eager attention is intentional: Transformers 5.6 rejects SDPA for the nested
`VibeVoiceAcousticTokenizerEncoderModel` and identifies eager attention as the
supported fallback.

The 8B BF16 checkpoint must first pass a 90-second smoke test on a GPU with at
least 24 GB usable memory. Do not start complete-recording pilots until the
memory, parser, timestamp-bound, and maximum-token gates pass.
