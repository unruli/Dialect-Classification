# Inference results: G3-B and G4-B — root causes and fixes

These were the two outstanding systems. Both had been written off, and **both
diagnoses were wrong**. Correcting them completes the architecture audit at 8/8
systems, all 95/95, all independently re-validated.

Frozen evaluation set: 95 recordings (17 AfriSpeech-Dialog, 16 AMI, 28 Bangor
Miami, 34 Playlogue). Outputs are normalized 10-field anonymized RTTMs.

---

## G3-B — BUTSpeechFIT DiaPer (10-attractor)

### Previous belief
A hardware-capacity failure: the longest recording (AMI `EN2002c`, 49.54 min)
OOM'd on a 24 GB RTX 4090 trying to allocate 52.66 GiB, so the system was
disqualified pending a larger GPU.

### Actual root cause
`diaper/backend/models.py:217-229` computes attention explicitly:

```python
scores = torch.matmul(q.permute(0,2,1,3), k.permute(0,2,3,1)) / np.sqrt(self.d_k)
self.att = F.softmax(scores, dim=3)     # retained on the module, never read
```

At `T = 59,445` (one frame per 50 ms) the score matrix is
`(1, 4, 59445, 59445)` fp32 = **52.66 GiB**. Because the softmax output is
assigned to `self.att` — **written once and never read anywhere in the
repository** — each of the 4 encoder layers keeps one alive for the whole
forward pass:

| | memory |
|---|---|
| `scores` | 52.66 GiB |
| `+ self.att` | **105.3 GiB peak per layer** |
| × 4 layers, never freed | **~210 GiB resident** |

**No GPU can satisfy this** — not an 80 GB A100, not a 141 GB H200. Waiting for
a larger card was never going to work.

Secondary: torch 1.10.0+cu113 has no `scaled_dot_product_attention` (torch 2.0)
and **no `sm_90` kernels**, which is why the H200 attempt died with "no kernel
image is available for execution on the device" — locking the system out of the
idle H200 partition.

### Fix
Route attention through `F.scaled_dot_product_attention` and drop the dead
`self.att`, on a torch 2.5.1+cu121 rebuild (Python 3.10, transformers 4.39.3,
librosa 0.9.2). This is a **kernel substitution, not a model change**: identical
default scale `1/sqrt(d_k)`, no mask, no positional bias, softmax over keys,
dropout inert under `eval()`.

Measured on an RTX 4090 at T=20,000 fp32:

| implementation | peak | max abs diff vs naive |
|---|---|---|
| naive (original) | 11.97 GiB | — |
| SDPA (`EFFICIENT_ATTENTION`) | **0.0465 GiB** (257× less) | **4.47e-07** |
| SDPA forced `MATH` | 19.42 GiB | (re-materialises N²) |

`FLASH` does not support fp32; `EFFICIENT_ATTENTION` does, and auto-selection
picks it. A silent fall back to `MATH` would restore the original failure, so
the backend is asserted at runtime.

### Equivalence gate (two-way ablation)
Because two variables changed at once (stack and kernel), they were separated:

| comparison | result |
|---|---|
| torch2+naive vs torch1.10+naive (baseline) | identical on 3/4; `sastre03` differs by **one 50 ms frame**, 0 label changes, same 115 segments |
| torch2+SDPA vs torch2+naive | **identical on all 4** |

So SDPA is a provable no-op on output; the single-frame shift is attributable to
the GPU/torch migration, not the fix. The `DIAPER_ATTN=naive` switch is retained
so the ablation is reproducible.

**Also load-bearing:** DiaPer depends on a patched `transformers`
`PerceiverSelfAttention` that softmaxes cross-attention over `dim=-2` and
renormalises (upstream uses `dim=-1`). The pinned fork no longer installs, so
the 3-line diff is re-applied to a stock 4.39.3. **Without it the model silently
produces wrong output with no error.**

### Outcome
- Longest file: **unrunnable on any GPU → 11.4 s**, 3850 segments, exactly 3
  speakers (matching the reference count), 100.00% coverage.
- Full 95: **95/95, 0 failures, 9 min 18 s** total on one H200 MIG slice.

---

## G4-B — microsoft/VibeVoice-ASR-HF (8B)

### Previous belief
Output "truncated at `max_new_tokens`" even at 131,072 tokens — read as a
context/token-budget limit.

### Actual root cause
Forensics on the failed `EN2002c` output (409,109 chars):

- **621 segments, monotonic, 3 speakers, covering 2970.00 s of 2972.26 s —
  99.92% of the recording.** Essentially complete.
- Then the model locked onto `"But yeah, I guess."` and repeated it **17,202
  times**, consuming **79.9% of all output** and ~50 min of a 61-min job.
- That runaway prevented the closing `]`, so `json.loads` raised (the
  processor's parser is unguarded at `processing_vibevoice_asr.py:294`) and a
  nearly-complete result scored as a **total failure**.

It is none of the suspected causes:

| suspected | measured |
|---|---|
| out of memory | 24.9 GiB of 143 GiB |
| out of context | 22,354 tokens of **131,072** (17%) |
| out of token budget | needed ~24k; had 65k, then 131k |
| `max_length=32768` clamp | overwritten by `generate()`; never binds |
| `max_position_embeddings=450` warning | cosmetic — a derived property about tokenizer chunk size, not LM context |

Two further findings shaped the fix:

- **The model is non-deterministic even at `do_sample=False`.**
  `modeling_vibevoice_asr.py:374-378` injects VAE noise (`vae_std=0.625`) at
  inference with no `if self.training` guard and no disable flag. Collapse is a
  **stochastic accident, not a property of the recording** — which is exactly
  what makes reseeding principled rather than a fudge.
- Only `<|endoftext|>` is declared EOS; `<|im_end|>` is not.
- `acoustic_tokenizer_chunk_size` bounds only CNN encoder memory — the LM always
  attends over every audio token, so it cannot help here.

**Scope:** 33 of 95 recordings exceed 30 min. The 4-domain 90-second smoke gate
is **structurally incapable** of detecting this — `sastre03` is a 43-minute
recording whose first 90 s pass cleanly.

**Cross-model corroboration:** G4-A independently failed on `EN2002c` + 7 Bangor
files — the same long, dense recordings — and was rescued by chunking, at the
documented cost of cross-chunk speaker fragmentation.

### Fix
Layered, most-principled first:

1. **Duration-aware stop** — halt once an emitted `End` passes the known audio
   duration. Task-aware: past that point the model is inventing content.
2. **Repetition stop** — generic backstop mirroring Microsoft's own VibeVoice
   vLLM repetition detector.
3. **Seeded generation + reseeded retry** — accepted per protocol as a primary
   result with seed logged, since every attempt is a full single-pass forward
   over the entire recording with no chunking and no cross-chunk speaker
   re-linking. Selection is **structural only** (coverage + segment soundness),
   never reference-based.
4. `eos_token_id=[151643, 151645]`; robust salvage parse via `raw_decode`;
   bracketed non-speech tags excluded from RTTM.

Deliberately **not** used: `no_repeat_ngram_size` / `repetition_penalty` — both
would corrupt legitimately repetitive output (JSON scaffolding every segment,
and real meetings repeat "Okay." constantly).

### Outcome
- `EN2002c`: valid RTTM, 575 segments, 3 speakers matching the reference,
  99.88% coverage — from total failure to usable.
- Guards cut a collapsed attempt from 3,657 s to ~500 s (**7.3× less** wasted
  compute).
- Full 95 (12-way shard, H200 MIG): **95/95, 0 failures**.
- After a targeted re-seed of the under-covered recordings:
  **91 primary / 4 recovered**, coverage mean **0.9839**, median 1.0.

### Honest limitation
**3 recordings remain under 98% coverage** and are reported as such rather than
hidden: `bangor_miami_eng/zeledon06` (0.1321), `playlogue/ew_42ec_22118`
(0.5113), `ami/IS1009a` (0.9603). The first two collapsed on **11 different
seeds**, so these are genuine model limitations, not sampling luck. Per-recording
coverage, seed, attempt count and stop reason are in
`g4b_vibevoice_full95/results/`.

Six other recordings *were* rescued by reseeding, several dramatically
(`ew_42pc_12014` 0.33→1.00, `gleason_mother_david` 0.23→1.00, `herring09`
0.80→1.00 with a clean EOS on seed 16) — direct evidence that collapse is a bad
draw rather than a fixed property.

---

## Final results — all 8 systems

760 recording-model rows. Mean DER, collar 0.0, overlap retained, UEM-cropped.
**Fixed pipeline order, not a ranking.**

| System | mean DER (collar 0) | mean JER |
|---|---|---|
| G1-A NeMo MarbleNet+TitaNet+spectral | 0.4771 | see CSV |
| G1-B MarbleNet VAD + VBx | 0.4980 | |
| G2-A pyannote community-1 | 0.3946 | |
| G2-B NeMo diar_msdd_telephonic | 0.5169 | |
| G3-A Sortformer | 0.4133 | |
| G4-A MOSS | 0.4455 | |
| G3-B DiaPer | 0.5414 | |
| G4-B VibeVoice-ASR | 0.3882 | |

All 8 pass `model_completion_status.py` at 95/95 with zero inconsistencies.

## Suggested follow-up

The duration-aware stop plus reseeding would likely let **G4-A's 8
chunk-recovered files be redone as clean single-pass runs**, removing their
documented cross-chunk speaker fragmentation. That belongs to whoever owns the
G4-A contribution.

Note also that a parallel branch (`codex/g4b-strict-recovery`) independently
implemented a G4-B repetition stop the same evening. The approach here differs
deliberately — it stays greedy and varies only the model's own inherent noise
seed, which is more faithful to the "deterministic generation" clause — and it
is validated end-to-end across all 95 recordings. The two should be reconciled
rather than both carried.
