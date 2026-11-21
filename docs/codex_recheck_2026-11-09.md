# GTCRN paper track 修复后的只读复核（2026-11-09）

范围限定为 `speech_denoising/training/paper_gtcrn.py` 及其直接使用的
manifest、配置和 smoke 代码。没有启动、停止或修改任何训练，也没有访问
正在运行的 LiSenNet 目录；本文件是本次唯一写入。

## Confirmed

1. **B1 裁剪错位已修复。** `_crop_pair()` 只调用一次 `torch.randint()`，
   然后用同一个 `start` 切 noisy 和 clean（`paper_gtcrn.py:419-437`）。
   两个张量长度不一致和空音频都会拒绝。现有 `aligned_crop` smoke 还检查了
   同起点、实际切片内容和不对齐输入的异常路径。

2. **短音频不再零填充。** 当音频短于 segment 时，noisy/clean 各自按同一
   周期重复并截到目标长度（`paper_gtcrn.py:433-437`）。因此没有静音 target
   帧被无 mask 地送入 `HybridLoss`；重复后的 noisy/clean 仍保持配对关系。

3. **resume 目录现在可打开。** `--resume` 会允许已有非空 run 目录，但仍要求
   `run.json` 存在，checkpoint 必须位于该 run 目录内，并校验 checkpoint 的
   `settings_sha256`（`paper_gtcrn.py:641-652, 704-747`）。

4. **数据与记录增强已生效。** 实际 manifest resume 会比较完整 manifest SHA
   和 `rows_sha256`；manifest 行中的 `noisy_sha256`/`clean_sha256` 在首次读
   音频时校验；`source.sha256` 会记录声明的源码文件；CSV 和 `run.json` 都
   记录实际验证数量（`val_count`/`validation_scored_count`）。

5. **已声明的配置护栏确实执行。** 当前实现拒绝不匹配的 FFT、hop、窗口、
   validation batch/segment、checkpoint 策略，并要求 run 目录位于配置的
   results root 下（`paper_gtcrn.py:474-536, 641-652`）。

## Discrepancy

1. **resume 没有真正 pin 住源码和 split provenance。** `_write_source_checksums()`
   每次启动（包括 resume）都会重写 `source.sha256`，但没有读取旧文件并比较；
   `git.dirty` 只写入 `run.json`，没有作为 resume 拒绝条件。与此同时，
   `run.json` 没有保存当前 `plan["split_sha256"]` 或验证 ID 指纹，resume 也不
   比较它们。若 manifest 字节不变但 `training_protocol.yaml` 的 split 参数被
   改动，当前代码可能用新 split 继续旧 checkpoint。相关位置：
   `paper_gtcrn.py:659-666, 704-714, 749-796`。

2. **resume 没有核对 checkpoint、CSV 和 run.json 的进度一致性。** 操作者可以
   指向同一 run 目录中较早的 checkpoint；代码会从该 epoch 继续，却直接向已有
   `learning_curve.csv` 追加，造成重复或乱序 epoch 行。恢复启动时还把
   `best`/`last` 暂时重置为 `None`，并且直到本轮完成才重新写入 history；若中途
   再次中断，run.json 的上一段进度元数据会丢失，尽管 checkpoint 和 CSV 仍在。
   位置：`paper_gtcrn.py:731-747, 749-805, 861-890`。

3. **B7 的配置校验覆盖面仍不是“所有 paper 配置”。** 代码读取但没有强制
   固定 `sample_rate`（实际数据和 STFT 使用模块常量 16 kHz）、`batch_size`、
   optimizer/type、learning rate、scheduler type 及各 loss 权重。修改这些值会
   改变运行行为，只会被记录进 settings fingerprint，而不会被判定为非法 paper
   recipe。当前配置文件本身的默认值是正确的，这属于护栏缺口而非当前默认运行
   的数值错误。

4. **epoch 边界仍缺少正数校验。** `--epochs 0` 会通过参数解析和
   `training_settings()`，随后生成一个没有 checkpoint 的 completed run。它不
   影响正常的正数训练，但会产生不可用的成功记录（`paper_gtcrn.py:474-537,
   895-917`）。

5. **音频 SHA 缓存按路径而不是按“路径 + 期望 hash”缓存。** 同一进程内若同一路径
   被另一个期望 hash 的行再次引用，第二次不会重新计算（`_VERIFIED_AUDIO`，
   `paper_gtcrn.py:366-375`）。规范 manifest 的路径唯一性使正常运行不受影响，
   但这是校验器的边界弱点。

## Unknown

- 论文/官方 GTCRN 代码没有定义短于训练 segment 的音频应如何补齐。wrap-pad
  对当前“无 padding mask”的 hybrid loss 语义是自洽的，也避免了零 target；但
  它会重复波形并在拼接处引入人为边界，不能称为论文明确规定的行为。
- 论文没有给出 VCTK 的 epoch budget，因此本次复核不能判断 CLI 传入的 epoch 数
  是否“论文正确”；代码把它保留为 operator input 是合适的。

## Verification note

仓库已有记录称 15/15 synthetic smoke 和 3-step CPU fixture 通过。本次尝试以
`-B` 只读重跑 smoke 时，环境在导入 `.conda` 的 `_ctypes` 阶段因 Windows DLL
初始化失败而中止，未执行到测试逻辑；没有启动 GPU 或写入训练产物。

## Verdict

这次修复解决了最严重的 noisy/clean 错位，并使 wrap-pad 的损失输入保持配对；
短音频路径本身没有发现新的 loss 语义错误。GTCRN 仍不宜宣称为完全可复现的
paper track，主要剩余阻断点是 resume 时未锁定源码/split provenance，以及可
回退到旧 checkpoint 后对 CSV/run.json 不做一致性检查。
