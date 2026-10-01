"""Shared evaluation contract; metric backends are intentionally unimplemented."""

METRIC_NAMES = (
    "pesq", "stoi", "si_snr", "si_snr_improvement", "rtf", "parameter_count",
)


def compute_audio_metrics(clean_audio, enhanced_audio, noisy_audio, sample_rate=16000):
    """Return per-utterance PESQ, STOI, SI-SNR and SI-SNR improvement.

    Inputs: aligned mono [samples] tensors at 16 kHz.
    SI-SNR improvement = SI-SNR(enhanced, clean) - SI-SNR(noisy, clean).
    """
    raise NotImplementedError("Phase 2: implement shared quality metric backends.")


def evaluate(model, dataset):
    """Return aggregate metrics for a model and paired dataset.

    Future implementation calls compute_audio_metrics for every utterance,
    averages quality metrics per utterance, measures RTF with synchronized
    inference timing, and counts model.parameters(). Record protocol metadata
    alongside scores. Run in eval/inference mode and restore prior model mode.
    """
    raise NotImplementedError("Phase 2: implement the unified evaluation runner.")
