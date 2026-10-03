# GTCRN pretrained integration — Phase 2.3

Upstream: https://github.com/Xiaobin-Rong/gtcrn
Pinned commit: 502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73
Checkpoint: checkpoints/model_trained_on_vctk.tar (737323 bytes)
Upstream README labels this checkpoint as trained on VCTK-DEMAND.
Checkpoint epoch: 92. Loading selects only the model state, not optimizer state.
Checkpoint SHA-256: a0f0e04421d9fc1efe734d44191f5dc4b973ce59d6546af8d64ad9167e2a9944
Git blob SHA-1: 1cec3e5fdede31fff264dd58766aeff48824b923

Only the network state is loaded into model.network, with strict=True, after
SHA-256 verification and safe weights_only=True deserialization. All 271
state entries must match. Counts include 48245 total parameters, with frozen
ERB parameters separately excluded from trainable parameter count.
Source/license attribution remains in SOURCE.md and LICENSE.upstream.

The original migration installer bundled the checkpoint and one upstream noisy
demo (`test_wavs/mix.wav`). Current checkouts use the fetcher below to restore them.
Demo SHA-256: 8d47e1d03eeb457c2549be79c8ec33a349ccd79f21c3add05f946f8f760c5a99
Demo git blob SHA-1: bab7cfd86b80c50533281f980e2f0343b5262223
It is 156302 samples of mono 16 kHz PCM16 audio. It is a listening demo,
not an utterance selected from this benchmark's fixed 824-item test manifest.
It has no clean reference here, so no reference-based quality scores are given.
Weights and demo/output audio stay in ignored checkpoints/ and results/.
They are not tracked by Git.

Commands:

    python -m scripts.smoke_gtcrn_pretrained
    python -m scripts.inference

The inference command writes results/gtcrn_demo/enhanced.wav and enhanced.json.
For personal mono 16 kHz PCM16 WAV:

    python -m scripts.inference --input YOUR_AUDIO.wav --output results/enhanced.wav

To restore the checkpoint and optional demo:

    python -m scripts.fetch_gtcrn_checkpoint --demo

The output timing is a single cold CPU run, not a controlled RTF comparison.
Whole-utterance centered-STFT inference does not claim stateful streaming.
Single-WAV inference is a listening demonstration. The committed full VoiceBank
evaluation is indexed in [results](../../../docs/results.md); training conditions
remain unmatched across pretrained checkpoints.
