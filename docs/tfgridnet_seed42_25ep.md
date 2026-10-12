# TF-GridNet seed 42 25-epoch pilot

Completed 25 training epochs and 25 independent validations.
Random initialization; local_6layer profile; FP32; micro-batch 1 with
gradient accumulation 4. Training utterances: 10,778; validation: 794.
Final optimizer steps: 67,375. Validation selected epoch 23.

All 824 fixed test utterances were evaluated without failed metrics.

| Metric | Result |
|---|---:|
| PESQ-WB | 2.795010139785924 |
| STOI | 0.9482537844712103 |
| SI-SNR dB | 18.322921437478385 |
| SI-SNR improvement dB | 9.877385312096893 |
| GPU whole-utterance RTF | 0.08426015152218175 |

Timing includes STFT, network and iSTFT; excludes I/O, transfer and
quality metrics. Each utterance has one warmup and three timed repeats.

These are single-seed results under the draft shared training protocol,
not an exact reproduction of the original paper. The fixed test set
was previously evaluated at the five-epoch stage.
Host migration differences are recorded alongside the reports.
Selected epoch 23 is for inference; epoch 25 is the final resume state.

Evidence: training_reports/tfgridnet_random_seed42_accum4_25ep/
Weights: Release tfgridnet-seed42-pilot-25ep.
