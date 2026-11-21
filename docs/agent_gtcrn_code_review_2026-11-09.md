# GTCRN paper track — 独立代码审计（2026-11-09）

> **Status:** Findings B1–B7 below are pre-fix. See the Addendum at the end: shared aligned crop, wrap-padding, resume checks, validation count, source/audio checksums and config guards were implemented; 15/15 smoke checks pass. No real GTCRN run exists because the paper gives no VCTK epoch budget.

范围：只读审计 `configs/gtcrn_paper.yaml`、`speech_denoising/training/paper_gtcrn.py`、
`scripts/smoke_gtcrn_paper.py`、`speech_denoising/losses/gtcrn.py`、`.research/gtcrn_loss.py`、
`.research/gtcrn_scribd.txt`、`docs/gtcrn_paper_track_plan_2026-11-09.md`。
未训练、未占 GPU、未安装依赖、未改动既有文件；唯一写入是本报告。探针均用 `./.conda/python.exe`（torch 2.11.0+cu128，CPU）。

## 0. 结论速览

| # | 结论 | 级别 | 位置 |
|---|---|---|---|
| B1 | **noisy/clean 各自独立抽随机起点，训练对完全错位**（实测 8/8 错位，偏移最大 16,649 samples） | confirmed / 致命 | `paper_gtcrn.py:502-503` + `:393-399` |
| B2 | `--resume` 与 `_prepare_run_dir` 互斥：同一 run dir 必然抛错，plan 规定的 resume 流程不可能跑通 | confirmed / 高 | `paper_gtcrn.py:578-589, 610, 645` |
| B3 | resume 不校验数据集身份（manifest 路径/sha）与 git dirty，换一份同长度 manifest 可通过 | confirmed / 高 | `paper_gtcrn.py:648-655` |
| B4 | `--validation-limit` 实际截断监控集，但 `run.json` 仍写 794；`val_stats["count"]` 从未落盘 | discrepancy / 中 | `:543, :671, :556-562` |
| B5 | 短音频零填充进 loss（无 mask），S=4.0 时 10,294/11,572（89.0%）被填零，与模块注释/plan §4.4 的"loss clean"理由矛盾 | discrepancy / 中 | `:397-399, :14-15` |
| B6 | `run.json` 缺 plan 要求的 `split.val_sha256`（及任何 split 指纹）；无 `source.sha256`；未校验 wav 的 `*_sha256` | discrepancy / 中 | `:657-696, :350-359` |
| B7 | `n_fft/hop_length/window`、`validation.*`、`checkpoint.*`、`results_root` 等配置项无人读取；前端参数也不进 `run.json` | discrepancy / 中 | `configs/gtcrn_paper.yaml:19-21,44-57`；`paper_gtcrn.py:52-55` |
| P1 | Loss 与官方 `loss.py` 逐项一致（含 eps 位置、`length=` 行为、100× 关系） | 通过 | 见 §1 |
| P2 | Adam 1e-3 + ReduceLROnPlateau 每 epoch 在 val loss 上 step 一次，首 epoch 用 1e-3 | 通过 | 见 §2 |

**总判断：当前 track 不能用于真实训练。** 训练信号本身被 B1 破坏，产出的 loss/曲线/checkpoint 全部不可解释。

---

## 1. HybridLoss vs 官方 `loss.py`（重点 1）

逐行对照 `.research/gtcrn_loss.py:14-30`，`paper_gtcrn.py:286-339` 全部对齐：

| 项 | 官方 | 本实现 | 判定 |
|---|---|---|---|
| 幅度 | `sqrt(re²+im²+1e-12)`，eps **在 sqrt 内** | `:298-299` 同 | 一致 |
| RI 归一化 | 每个信号除以**自己**的 `‖X‖^0.7` | `:301-307` 同（pred 用 pred_mag，true 用 true_mag） | 一致 |
| magnitude | `(‖X‖)^0.3` 后 MSE | `:308-310` | 一致 |
| 窗 | sqrt-Hann, 512/256, center=True | `:270-272, 274-284`，`torch.hann_window(512).sqrt()` | 一致 |
| ISTFT | **不带 `length=`** → `(T−1)·256` | `:274-284` 无 `length` | 一致（smoke #2 实测 16,000→15,872） |
| SI-SNR | 投影用 `<y,ŷ>`；分子 `‖proj‖²`、分母 `‖ŷ−proj‖²+1e-8`，外层 `+1e-8`，**log10**（非 10·log10） | `:314-323` 同 | 一致 |
| 权重 | `30·RI + 70·mag + 1·sisnr` | `:325-329` `0.01·sisnr + 0.7·mag + 0.3·ri`，`official_x100` = ×100 | 精确等价：30=100×0.3、70=100×0.7、1=100×0.01 |

- 交叉验证：smoke #14（`smoke_gtcrn_paper.py:223-233`）拿独立实现 `losses/gtcrn.py:108-141` 当 oracle，容差 1e-3，能抓住归一化用错对象这类错误。
- 小瑕疵（不影响数值）：`losses/gtcrn.py:5-7` 与 `paper_gtcrn.py:5-6` 称 100× "只通过梯度裁剪影响 Adam"。严格说 Adam 的 ε 也会破坏尺度不变性（相对 ε 效应缩小 100×）；当前 `grad_clip_norm: null`、`weight_decay: 0.0`，实际无差异，但注释应加 "up to Adam ε"。
- `HybridLoss` 的 n_fft/hop 是模块常量（`:274-283`），`settings["n_fft"]` 从不传入 → 改配置只会改指纹、不改行为（见 B7）。

## 2. 优化器 / 调度器（重点 2）

- `build_optimizer`（`:413-420`）：Adam，lr 1e-3、betas (0.9,0.999)、eps 1e-8、wd 0，只喂 `requires_grad` 参数（203 tensors / 23,669 params，与 plan §3.1 一致）✅
- `build_scheduler`（`:423-433`）：`mode=min, factor=0.5, patience=5, threshold=1e-4, threshold_mode=rel, cooldown=0, min_lr=0.0`，与 config `:25-33` 一致 ✅
- step 顺序：**先跑完 epoch → validate → `scheduler.step(val_stats["total"])`（`:743`）**，因此第一个 epoch 全程用 1e-3，不存在"第一轮就用错 LR" ✅
- `verbose`：生产代码**完全不传** `verbose`（正确做法）。实测 torch 2.11.0 传入会 `TypeError: ... unexpected keyword argument 'verbose'`，与 plan §3.2 一致 ✅
- 但 smoke #11（`smoke_gtcrn_paper.py:163-181`）把"本机 torch 拒绝 verbose"写成断言：在仍接受该弃用参数的 torch 构建上会 `AssertionError("this torch build accepts verbose=; docstring expectation is stale")`，属于**测试绑定 torch 版本的脆弱点**，不是训练代码缺陷。
- patience 语义：torch 在 `num_bad_epochs > patience` 时才降 → 第 6 个连续不降 epoch 才首次减半，比论文散文"5 consecutive"晚一轮；plan §3.2 已明示并选择保留作者原值，属**已知偏差**而非 bug。
- `row["lr"] = lr_before`（`:750`），CSV 里看到的是"该 epoch 实际使用的 LR"，减半体现在生效的那一轮，与 plan §4.5 "drop on the epoch it happens" 的措辞略有出入；print 行同时给出 `lr_before -> lr_after`，可读性 OK。

## 3. 数据选择与 crop（重点 3）

**选择逻辑本身正确。** 本机实测（只读）：

```
rows 11572 | train 10778 | val 794 | overlap 0 | union==parent True | speakers 26/2 disjoint
algorithm speaker_sha256_v1 seed 42 | samples 17,478–241,793（中位 2.679 s）
```

- `paper_full`：`tuple(range(len(rows)))` = 11,572，监控集是同一批里的 794 → **in-sample**，`run.json` 记 `in_sample_monitor: true`（`:671-673`），与 plan §4.3 的声明一致 ✅
- `held_out`：10,778 / 794，靠 `data.py:42-45` 强制 `train ∩ val = ∅` 且 `train ∪ val = parent` ✅
- 与论文的偏差：论文是 11,572 中抽 1,572 作验证（`.research/gtcrn_scribd.txt:588`），即 10,000/1,572。仓库两个选项**都不等于论文**：`paper_full` 池子对但监控在样本内；`held_out` 干净但只用 10,778 训练。报告结果时必须写明用的是哪个，否则"paper recipe"是虚标。
- **B1（致命）**：`train_epoch` 对同一句话调用两次 `_crop_or_pad`：

```python
noisy, padded_noisy = _crop_or_pad(noisy, segment)   # :502
clean, padded_clean = _crop_or_pad(clean, segment)   # :503
```

而 `_crop_or_pad` 内部各自 `torch.randint(0, numel-length+1)`（`:395`）。实测 100,000 样本、segment 64,000、连续 8 次：8/8 错位，偏移 −16,649…+14,337 samples。等价于用 A 段噪声去回归 B 段干净语音，梯度是噪声，loss 会停在高位且不收敛。
- **B5**：短音频用尾部零填充（`:397-399`），无 mask，仅计数。实测：S=2.0 → 12.9%、S=3.0 → 63.8%、S=4.0 → **89.0%**、S=5.0 → 96.3% 的句子被填充。模块注释（`:14-15`）说"固定长度随机 crop 是为了不把 padded 帧当信号"，但短句仍然被填零进入 loss → 与大段静音一起算 MSE/SI-SNR，梯度被稀释；`plan §4.4` 的 "loss clean" 只在 S ≤ 1.09 s 时成立。
- 其他：每个 epoch 用 `soundfile` 串行读盘（`:501`）、最后一个 batch 可能小于 4（10,778 % 4 = 2），仅影响吞吐/梯度噪声，非正确性问题。

## 4. Provenance / resume / guard（重点 4）

`run.json` 已有：schema_version、scope、status、recipe、`config.sha256`、`dataset{path,sha256,count}`、`split{train_pool,count,in_sample_monitor,purpose,selection}`、epochs+source、segment+source、batch、loss+scale、optimizer、scheduler、grad_clip、seed、`git{commit,dirty}`、environment、command、started_at、best/last。这部分比多数仓库扎实，但仍缺：

1. **B6** plan §4.6 要求的 `split.val_sha256` 没有；`data["plan"]["split_sha256"]` 明明可取（`data.py:32-45`）却没写 → 事后无法复原"这 794 条到底是哪 794 条"。
2. 没有 `source.sha256`（训练模块/脚本的哈希）。一旦 `git.dirty=true`，`commit` 不再能定位代码。
3. `config.sha256` 只哈希覆盖文件，不含 `extends` 解析后的 `gtcrn.yaml`/`default.yaml`。
4. manifest 行里带 `noisy_sha256/clean_sha256`，`_read_pair`（`:350-359`）只读波形不校验 → 音频漂移不可发现。
5. `checksums.json` 只在循环正常结束后写（`:780-784`），中断就没有；`status` 只有 running/completed，无 failed。
6. **B7** 前端参数（512/256/sqrt-Hann）不进 `run.json`，同时 config 里的 `n_fft/hop_length/window` 又不生效。
7. **B2** resume 死路：`run()` 在 `:610` 先 `_prepare_run_dir`，该函数对"已存在且非空"直接抛错（`:583-585`）；而 plan §4.6 要求"same run dir + `--resume`"。唯一可行路径是换一个空目录，但那会丢掉旧 checkpoint、把 `started_at` 重置为现在（`:693`），末尾 `checksums.json` 只覆盖 resume 之后的 epoch。
8. **B3** `settings_fingerprint`（`:476-478`）含 epochs/segment/train_pool/loss/scheduler/seed，**不含 manifest 路径与 sha**，也不含 git dirty；因此"改 epochs/segment 被拒"成立 ✅，但"换数据集"不设防 ❌。另外 `torch.load(..., weights_only=False)`（`:647`）会执行 pickle。
9. **B4** `--validation-limit` 会截断监控集（`:543`），但 `run.json.validation_count` 写的是完整 794（`:671`），曲线里也没有 count 列。
10. Guard：`--device cuda` 且 `device_count()==0` 抛错 ✅（`:440-441`）；`--fixture` 强制 CPU ✅（`:603-604`）。但 `environment_record`（`:111-113`）无条件探测 CUDA 并取 device 0 属性 → 标称"CPU-only fixture"的运行在有 GPU 的机器上仍会初始化 CUDA 上下文。另缺 `epochs >= 1` 校验：`--epochs 0` 会写出 `status=completed`、零 checkpoint 的 run。

## 5. 其他偏差（低）

- `GTCRNPaper.forward` 没有 `GTCRNModel.forward` 的短输入 `F.pad`（`models/gtcrn/model.py:30`），`stft` 用 reflect padding 时 L < 257 会报错；常规句长无关紧要，但 #4 的 parity 只在 L ≥ 257 成立。
- `validate` 用整句、训练用 4 s crop，网络含 BatchNorm2d（`network.py:102,119,125,129`）→ 训练/验证的 running stats 分布不一致，监控值系统性偏乐观或偏保守；plan 只提了 step-0 BN 的注意事项，没提长度不一致。
- smoke #5 把容差收到 1e-5（`smoke_gtcrn_paper.py:81-91`），plan §4.7 写的是 1e-3 → 换 torch/BLAS 版本可能误报。
- smoke #8 的注释"mse divides by 1*257*1"与真实 `1*257*5` 不符（结论仍对，靠 `×257` 抵消）。

## 6. 结论：能否用于真实训练

**不能。** 必须先修 B1，否则任何训练曲线、checkpoint、验证 loss 都不可解释；B1 修好后 B2/B3 会让"可复现"这一条也不成立。

必修（按顺序）：

1. **B1**：`_crop_or_pad` 改为一次抽起点、同时切 noisy/clean（例如 `_crop_pair(noisy, clean, length)` 返回同一 `start`），并加一条断言：`(noisy_crop == 原始noisy[start:start+L])` 且 clean 同 start；smoke 里补一个"aligned crop"检查。
2. **B2**：`_prepare_run_dir` 增加 `resume=True` 分支——允许已存在目录但要求 `run.json` 存在且 `settings_sha256`/`dataset.sha256` 一致；或在 resume 时跳过创建逻辑。
3. **B3**：把 `manifest.path`、`dataset.sha256`、`git.commit`、`split_sha256` 纳入 resume 比对；`run.json` 追加 `resumed_at`/`resume_from`，不要覆盖 `started_at`。
4. **B4/B6**：`run.json` 写 `val_sha256`（或 `split_sha256`）、实际参与验证的 `count`、前端参数（`n_fft/hop/win/window`），并落 `source.sha256`。
5. **B5**：至少二选一——(a) 短句按 `min(len)` 截断而不是补零；(b) 保持补零但在 `run.json` 记 `padded_fraction`，并把 S 限制在 1.0–2.0 s（此时填充率 ≤ 12.9%）。当前默认无 S，operator 很可能选 4.0（统一协议的习惯值），恰好是最差的 89%。

修完 1–4 可以开跑；第 5 项决定结果能否被称为"paper recipe"。

## Addendum — fixes applied

The findings above were addressed the same day in `paper_gtcrn.py`: one shared
crop start (`_crop_pair`, wrap-padding short pairs), resume-aware run
directories with manifest/`rows_sha256` checks, recorded
`val_count`/`validation_scored_count`, `source.sha256`, per-file audio SHA-256
verification, and enforced config values for STFT/validation/checkpoint/results
root. The smoke suite adds an aligned-crop check and passes **15/15**; the
3-step CPU fixture passes. No GTCRN real run has been launched.
