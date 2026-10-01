# Unified VoiceBank-DEMAND-16k evaluation — Phase 2.4

Dataset source: https://huggingface.co/datasets/JacobLinCool/VoiceBank-DEMAND-16k
Revision: 20879f4f9aab3d0b9263993667e7711a3ae1416d
Dataset card: cc-by-4.0; 11572 train, 824 test; id/clean/noisy at 16 kHz.
configs/voicebank.json pins all six parquet file sizes and SHA-256 hashes.
The test shard is 132200447 bytes (about 132 MB); train shards total about
2.15 GB. The initial preparation command downloads only the full test shard.
Training is not performed. Validation must later be derived reproducibly from
train; the official test is never used for selection/tuning.

Preparation validates parquet SHA-256, unique IDs, filename pairing, mono
16 kHz audio, finite samples and equal clean/noisy lengths. Audio bytes are
preserved. Sorted manifest rows have a stable SHA-256; every audio file has its
own hash. The loader verifies these on access and forbids paths outside the
manifest root. Full manifests require the protocol count; fixtures require an
explicit opt-out and cannot produce a full benchmark report.

Quality metrics use aligned complete utterances with no independent gain
normalization, clipping, truncation, resampling or silence trimming:
- PESQ: pesq 16 kHz wideband MOS-LQO.
- STOI: pystoi classical STOI, extended=False.
- SI-SNR: remove signal means, project estimate onto clean reference, then
  10*log10((projected energy + 1e-12)/(residual energy + 1e-12)). A silent
  clean reference is undefined and yields an explicit metric error.
- SI-SNR improvement: enhanced SI-SNR minus noisy SI-SNR.
Each quality mean is a per-utterance macro mean. Noisy baseline scores use the
same implementation. Errors have null scores and named failures, and coverage
is recorded; a report with failures is not accepted as complete. Missing
backends stop evaluation before inference unless those metrics were explicitly
omitted. Omitted metrics are recorded in not_computed and all_quality_metrics_complete=False.

RTF is sum of per-utterance median inference time / total audio duration.
Default: CPU, FP32, batch=1, one thread, one warmup per utterance, three timed
runs per utterance. STFT/network/iSTFT are included; file decoding, host/device
transfer and metric calculation are excluded. CUDA timing synchronizes if a
CUDA model is supplied to evaluate(). Platform, precision, threads, warmups,
repeats and package versions are stored. Whole-utterance centered-STFT RTF is
not streaming latency. Parameter count includes frozen ERB parameters; the
trainable count is reported separately.

## Windows first run

    python -m pip install -r requirements-benchmark.txt --index-url https://pypi.org/simple
    python -m scripts.smoke_benchmark
    python -m scripts.prepare_data --split test
    python -m scripts.evaluate --metrics stoi si_snr si_snr_improvement --limit 5 --output results/gtcrn_test5.json

The five-utterance run is a wiring check and is labeled subset_or_fixture.
For a full 824-utterance report with the available metrics:

    python -m scripts.evaluate --metrics stoi si_snr si_snr_improvement

PESQ backend source: https://github.com/ludlows/PESQ (requires a C compiler,
NumPy and Cython). On Windows, installing pesq may require Microsoft C++ Build
Tools. Use the documented pesq installation for your platform; once installed:

    python -m scripts.evaluate

No different metric is substituted for PESQ. A report omitting PESQ is explicitly
partial in metric coverage even when all 824 utterances were processed.
STOI backend: https://github.com/mpariente/pystoi

For download problems, manually obtain the pinned test parquet from the source
resolve URL and use --parquet PATH with prepare_data; SHA-256 is still enforced.
Data/manifests remain under ignored data/ and report files under ignored results/.
GTCRN results are upstream-pretrained results using the VCTK-DEMAND checkpoint;
they do not represent a controlled comparison with the other two models.
