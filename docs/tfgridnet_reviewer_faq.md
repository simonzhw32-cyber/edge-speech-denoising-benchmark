# TF-GridNet memory-policy paper — reviewer FAQ

Working notes for the draft in `docs/tfgridnet_paper_draft.md`. These are
answers to the objections most likely to decide the paper's fate.

## 1. "This is just `torch.utils.checkpoint`."

The contribution is not the API call. It is the measured cost model and the
selection rule: attention is the only sub-block whose memory saving is cheap
enough to take first, and the remaining budget should go to whole blocks.
The paper reports the full 11-policy Pareto, the negative intra/inter-RNN
result, and a 4-layer boundary check. The rule is validated point-by-point,
not assumed additive.

## 2. "Only one GPU. Does this generalize?"

The draft is explicitly scoped to one 16 GB RTX 5060 Ti. The 4-layer DNS
profile is a second model size, not a second GPU. E5 (24 GB / data-center GPU)
is listed as missing in `docs/tfgridnet_submission_checklist.md`; do not claim
generalization without it.

## 3. "The quality proxy is too short."

The 200-step pair is an optimization-equivalence check, not a final PESQ
result. Both 1,000-step arms reach PESQ 2.123 on a fixed 20-utterance subset;
the attention arm matches the uniform trajectory at all ten checkpoints. Final
quality with the full Wav+Mag+MC objective remains future work.

## 4. "Why 8 ms if 16 ms is the repository default?"

8 ms is TF-GridNet's paper-faithful hop. The 16 ms profile is a different STFT
configuration and is not a reproduction. The paper studies the 8 ms case
because that is the configuration that does not fit in FP32 on the target GPU.

## 5. "The attention arm is slower in the long run."

Two A/B/A controls (30 steps each) bracket the 1,000-step run: in a fast
session uniform/attention/uniform measured 5.601/5.624/5.611 s; in a slow
session 7.297/6.868/7.406 s. The attention arm is not slower in either. The
long arm itself alternated between 7.66 and 5.65 s/step with no code change,
so absolute step time is environment-sensitive. The paper claims memory and
trajectory equivalence, not a step-time speedup.

## 6. "Why not compare against the paper's full training recipe?"

The repository's unified benchmark uses negative SI-SNR for all models, and
the compute-policy comparison is objective-agnostic. A full Wav+Mag+MC run is
outside the compute budget of this study. The deviation is stated in the
abstract, §5, and the submission checklist.

## 7. "What is actually new compared with Checkmate / selective recomputation?"

Checkmate solves optimal rematerialization for a generic graph; this paper
measures the cost structure of TF-GridNet's block and finds a cheap-first
ordering that a generic solver would not exploit without per-section cost
measurements. The negative result for RNN sub-blocks is also specific to this
architecture. The claim is a systems recipe and measured Pareto, not a new
general-purpose solver.

## 8. "Doesn't FP16 change the quality?"

At 16 ms, a 200-step FP32/FP16 pair gives final PESQ 1.891 vs 1.889 and STOI
0.8739 vs 0.8749, while FP16 saves 4.27 GiB allocated. The loss trajectories
are not bit-identical (mean abs difference 0.37; final SI-SNR 1.14 dB lower in
FP16), so we call the result quality-neutral within proxy noise, not exact
equivalence. At 8 ms, FP32 does not fit at all; FP16 is a feasibility
requirement, not an optional speedup.
