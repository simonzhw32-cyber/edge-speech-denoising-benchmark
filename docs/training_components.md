# Waveform training components

Phase 3.2 implements the loss, paired crops and collator used by the
[training draft](training_protocol.md). There is still no training loop,
optimizer step, resume pipeline or validation/selection orchestration.
`train.py` and `validate.py` remain separate configuration-only entry points.

## Loss

`EnhancementLoss.forward(enhanced_audio, clean_audio, lengths=None)` accepts aligned
mono `[batch, samples]` tensors in FP32 or FP64, on the same device. Instantiate
the module before calling it. Valid lengths are an integer `[batch]` tensor
on that device. Omitting lengths means the complete waveform is valid.

For each valid slice, subtract its own mean from estimate and reference, then
project the estimate onto the reference. The projection divides by the clean
energy without epsilon. Negative SI-SNR is
`-10*log10((projected_energy + 1e-12)/(residual_energy + 1e-12))`.
This matches the existing evaluation definition; the evaluation implementation
is unchanged. Mean reduction weights utterances equally, not by length.
`reduction="none"` returns one value per utterance for validation aggregation.

Masking happens before centering and projection. Invalid tails do not affect the
loss or its gradients, even if those tails contain NaNs or infinities. Valid
non-finite samples are rejected. Silent/constant clean references (zero-mean
energy <= epsilon), invalid lengths, dtype/device mismatches and energy overflow
produce errors rather than silent fallback scores. Half precision is unsupported
in this FP32 training draft. A single-sample reference cannot define SI-SNR.

```python
from speech_denoising.losses.loss import EnhancementLoss

criterion = EnhancementLoss()
loss = criterion(enhanced_audio, batch["clean_audio"], batch["lengths"])
```

## Paired crops and batches

`paired_crop(sample, segment_samples=..., seed=..., epoch=...)` accepts an
uncropped dataset audio-pair dictionary. Noisy and clean use one shared offset.
The offset is SHA-256-based on the crop algorithm version, seed, epoch, utterance
ID, original length and segment length. It does not depend on Python hashing,
global RNG state, access order or DataLoader worker count. Offsets are not
guaranteed to differ between every pair of epochs.

Short utterances are right-padded to the segment width; `valid_samples` retains
their true length. Crops own their storage and do not modify the original sample.
There is no resampling, clipping, independent normalization or additional noise.
The draft segment length is 64,000 samples; these utilities also accept shorter
synthetic segments for tests.

`collate_audio_pairs(samples)` handles raw pairs or crops. It pads to the maximum
physical width and returns paired `[batch, samples]` tensors, valid `lengths`,
ordered utterance IDs and crop offsets. All items must share dtype/device and be
finite 16 kHz mono pairs. A DataLoader may use this function as `collate_fn`.
When moving a batch to an accelerator, move both audio tensors and lengths.

Valid-length masking controls the loss, not model attention or STFT boundaries.
Padding can still affect a model's predictions within a valid slice. Full-length
validation therefore stays batch=1, as the draft specifies. This stage does not
add attention masks, alter frontends or promise padding-invariant model outputs.
The collator also does not filter train IDs or shuffle data; a later trainer must
consume the accepted split plan and keep test data out of learning and selection.

## Check

```sh
python -m scripts.smoke_training_components
python -m scripts.smoke_training_plan
python -m scripts.smoke_benchmark
```

The new check compares FP64 loss to the existing NumPy metric, tests FP32 values,
masked padding and gradients, runs finite-difference `gradcheck`, and checks
paired crop alignment and deterministic batching. It then backpropagates one
synthetic shared loss through each random-weight model in eval mode. There are
no optimizer steps, parameter changes, asset downloads or VoiceBank scores.
Backward checks do not establish convergence, memory feasibility or a successful
training pipeline. The six-layer TF-GridNet default is used, not the DNS profile.
