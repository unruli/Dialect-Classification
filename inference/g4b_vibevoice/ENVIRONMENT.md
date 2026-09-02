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

## Deterministic recovery condition

The initial seed-0 smoke failures were not caused by the 4,096-token safety
ceiling: three generations had already entered repetition loops with invalid
timestamp resets. Greedy text decoding alone is not deterministic because the
Transformers implementation still samples noise in the acoustic VAE
(`vae_std=0.625`).

The runner therefore exposes `--g4b-acoustic-latent-mode mean`. In this mode it
sets the acoustic VAE standard deviation to zero immediately after loading,
which is the Transformers equivalent of Microsoft's official vLLM
`VIBEVOICE_USE_MEAN=1` option. This is recorded as a distinct controlled
inference condition; it is not silently substituted for the failed sampled
condition. For sampled conditions, the RNG is reset immediately before
`generate()` so the recorded seed controls the acoustic draw.

Use the official 1,440,000-sample tokenizer chunk on a full 40-GB A100 for the
primary recovery test:

```bash
python inference/run_model.py \
  --system G4-B \
  --path-manifest /local/path/to/path_manifest.csv \
  --selection-manifest dataset_metadata/final_evaluation_manifest.csv \
  --output-dir /local/path/to/G4-B-mean-smoke \
  --cache-dir /local/path/to/model-cache \
  --device cuda \
  --max-new-tokens 4096 \
  --g4b-acoustic-latent-mode mean \
  --g4b-tokenizer-chunk-size 1440000 \
  --full \
  --recording-id 5129fd8c-7b8c-4d05-a03a-196bcae4deff \
  --recording-id ew_42pc_22148 \
  --recording-id EN2002a \
  --recording-id sastre03 \
  --trim-seconds 90
```

Keep the 4,096-token ceiling for the 90-second gate; valid earlier outputs used
only 854--1,102 tokens. If all four pass, complete-recording pilots and the
95-file run use the checkpoint's released 32,768-token ceiling. The helper
`inference/common/require_success.py` enforces those scheduler gates.
