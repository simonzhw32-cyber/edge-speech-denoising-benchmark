# Reproducing the TF-GridNet 8 ms memory-policy results

This note lists the exact commands behind the paper draft. Paths are relative
to the repository root. The environment used for the reported numbers is
Python 3.14.9, PyTorch 2.11.0+cu128, CUDA 12.8, RTX 5060 Ti 16 GB, Windows 11.

## 1. Model and data check

```powershell
.\.conda\python.exe -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
.\.conda\python.exe -c "from speech_denoising.models.tfgridnet.model import TFGridNetModel; m=TFGridNetModel(n_layers=6, stride=128); print(sum(p.numel() for p in m.parameters()))"
```

Expected six-layer parameter count: `8,381,504`.

## 2. Pareto benchmark

Each row runs in a fresh process; one warm-up then 20 measured AdamW steps:

```powershell
.\.conda\python.exe scripts\bench_tfgridnet_policies.py --child `
  --config configs\tfgridnet.yaml --stride 128 --amp fp16 --steps 20 `
  --policy none
```

The full set of policies is in `results/paper_runs/tfgridnet_pareto_8ms_fp16.json`.
The figure is regenerated with:

```powershell
.\.conda\python.exe scripts\plot_tfgridnet_pareto.py
```

## 3. 200-step paired equivalence

```powershell
.\.conda\python.exe scripts\train_tfgridnet_policy.py `
  --run-dir results\paper_runs\tfgridnet_policy_proxy\p3_whole3_200_paired `
  --steps 200 --seed 42 --stride 128 --amp fp16 --checkpoint-blocks 0,1,2 `
  --train-pool 400 --validation-size 40 --validation-limit 20 `
  --validation-every 50 --skip-audio-verify --log-every 25

.\.conda\python.exe scripts\train_tfgridnet_policy.py `
  --run-dir results\paper_runs\tfgridnet_policy_proxy\p4_attn_whole3_200_paired `
  --steps 200 --seed 42 --stride 128 --amp fp16 --checkpoint-blocks 0,1,2 `
  --train-pool 400 --validation-size 40 --validation-limit 20 `
  --validation-every 50 --skip-audio-verify --log-every 25 `
  --subblock-attention
```

The two `learning_curve.csv` loss columns must be bit-identical; the
consolidated check is `quality_equivalence_200.json`.

## 4. 100-step attention-only check

```powershell
.\.conda\python.exe scripts\train_tfgridnet_policy.py `
  --run-dir results\paper_runs\tfgridnet_policy_proxy\q_none_100 `
  --steps 100 --seed 42 --stride 128 --amp fp16 --checkpoint-blocks 0 `
  --train-pool 400 --validation-size 40 --validation-limit 20 `
  --validation-every 50 --skip-audio-verify

.\.conda\python.exe scripts\train_tfgridnet_policy.py `
  --run-dir results\paper_runs\tfgridnet_policy_proxy\q_attn_100_cache `
  --steps 100 --seed 42 --stride 128 --amp fp16 --checkpoint-blocks 0 `
  --train-pool 400 --validation-size 40 --validation-limit 20 `
  --validation-every 50 --skip-audio-verify --subblock-attention
```

The conditional cache release is in `tfgridnet_policy.py`; without it the
attention run degrades after validation (see the paper's cache-cliff section).

## 5. 1,000-step quality pair

```powershell
.\.conda\python.exe scripts\train_tfgridnet_policy.py `
  --run-dir results\paper_runs\tfgridnet_policy_proxy\p3_whole3_1000 `
  --steps 1000 --seed 42 --stride 128 --amp fp16 --checkpoint-blocks 0,1,2 `
  --train-pool 400 --validation-size 40 --validation-limit 20 `
  --validation-every 100 --skip-audio-verify
```

Repeat with `--subblock-attention` for
`p4_attn_whole3_1000`. The trajectory figure is regenerated with:

```powershell
.\.conda\python.exe scripts\plot_tfgridnet_quality.py `
  --series "whole 0-2 (200-step)=results/paper_runs/tfgridnet_policy_proxy/p3_whole3_200_paired" `
  --series "whole 0-2 (1,000-step)=results/paper_runs/tfgridnet_policy_proxy/p3_whole3_1000" `
  --out results/paper_runs/tfgridnet_quality.svg
```

## 6. 4-layer boundary

```powershell
.\.conda\python.exe scripts\bench_tfgridnet_policies.py --child `
  --config configs\tfgridnet_dns.yaml --stride 128 --amp fp16 --steps 20 `
  --policy dns4_attn --checkpoint-blocks 0 --subblock-attention```

Consolidated values: `results/paper_runs/tfgridnet_dns4_8ms.json`.

## 7. Precision pair at 16 ms

The 8 ms model does not fit in FP32, so the precision control uses the 16 ms
profile. Both runs are 200 steps, no checkpointing, the same pool/seed, and a
fresh run directory:

```powershell
.\.conda\python.exe -m scripts.train_tfgridnet_policy `
  --run-dir results\paper_runs	fgridnet_policy_proxy\prec16_fp32_200 `
  --steps 200 --stride 256 --amp none --checkpoint-blocks 0 `
  --train-pool 400 --validation-size 40 --validation-limit 20 `
  --validation-every 50 --segment-seconds 4 --batch-size 1 `
  --learning-rate 1e-3 --seed 42 --skip-audio-verify --log-every 50

.\.conda\python.exe -m scripts.train_tfgridnet_policy `
  --run-dir results\paper_runs	fgridnet_policy_proxy\prec16_fp16_200 `
  --steps 200 --stride 256 --amp fp16 --checkpoint-blocks 0 `
  --train-pool 400 --validation-size 40 --validation-limit 20 `
  --validation-every 50 --segment-seconds 4 --batch-size 1 `
  --learning-rate 1e-3 --seed 42 --skip-audio-verify --log-every 50
```

Consolidated values: `results/paper_runs/tfgridnet_policy_proxy/precision_equivalence_200.json`.
FP16 saves 4.27 GiB allocated; final PESQ differs by -0.003, but losses are not
bit-identical (mean abs diff 0.37), so the paper calls this quality-neutral
within proxy noise, not exact equivalence.

## 8. Checks

```powershell
.\.conda\python.exe -m py_compile speech_denoising\training\tfgridnet_policy.py
git diff --check
```

The paper's absolute step times include per-process kernel setup; compare
rows within one protocol, not across processes. Two 30-step A/B/A controls
(fast and slow sessions) find no attention penalty for the 0.52 GiB saving;
absolute step time varies with session state, so do not compare across
sessions. See `timing_controls.json` and `timing_analysis.json`.
