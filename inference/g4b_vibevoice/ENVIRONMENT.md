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
- seed 0 for the acoustic tokenizer's released stochastic latent-noise step;
- `acoustic_tokenizer_chunk_size=64000` (a multiple of the released
  3200-sample hop; exposed as `--tokenizer-chunk-size` by this runner);
- batch size 1 and no oracle speaker count;
- raw native text plus the processor's parsed output retained before RTTM
  normalization.

Raw text is written before parsing. If the released processor raises while
decoding malformed JSON-like model output, the parser error is retained and
the recording fails the smoke gate without attempting output repair.

Eager attention is intentional: Transformers 5.6 rejects SDPA for the nested
`VibeVoiceAcousticTokenizerEncoderModel` and identifies eager attention as the
supported fallback.

Use `--max-new-tokens 4096` with `run_model.py` for the bounded 90-second
smoke set. Retain the runner's 32768-token default for complete recordings.

The 8B BF16 checkpoint loaded and inferred without offload on a 20-GB A100 MIG
slice (16,241.5 MiB peak), so GPU capacity passed. The deterministic four-domain
smoke nevertheless failed 0/4: three outputs hit the 4,096-token ceiling and
one failed the official processor's JSON parser. Do not start complete-recording
pilots under the frozen primary settings.

The smoke resolved checkpoint revision
`f22241c2062b3b25272bf117397e03d73381037a`.
