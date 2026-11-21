# 独立只读审计（2026-11-09）

本审计重点阅读了 `docs/paper_alignment.md`、LiSenNet paper-track 训练脚本、LiSenNet 判别器、`.research/lisennet.txt`、`.research/tfgridnet.txt`、`.research/gtcrn.pdf` 及 `.research/upstream/*.py`，并仅查看了相关配置和现有审计记录作交叉引用。没有启动、停止或重启训练，没有访问 GPU，没有安装依赖，也没有修改正在运行的 `results/paper_runs/lisennet_paper_seed42_100ep`。本次只写入本文件。

## confirmed

### LiSenNet

- 论文文本确认 VoiceBank+DEMAND 有 11,572 条训练语音、872 条测试语音、28/2 位说话人，并从 48 kHz 重采样到 16 kHz（`.research/lisennet.txt:141-149`）。论文确认 Hann 窗、512 点 FFT、256 hop（32/16 ms），DPR 重复 2 次，Griffin–Lim 迭代 2 次（`:159-166`）。
- 论文明确写明 AdamW、100 epochs、L2 gradient clipping 5.0、初始学习率 5e-4、每 epoch 衰减因子 0.98（`.research/lisennet.txt:167-169`）。因此 100 epochs 是论文值，不是统一 benchmark 草案值。
- 论文的损失权重明确为 λ1=0.9、λ2=0.1、λ3=0.05（`.research/lisennet.txt:132-136`）；PESQ 判别器把 PESQ 缩放到 (0,1)，生成器的 PESQ loss 是预测 1 的 MSE（`:113-126`）。
- 官方数据模块默认 `cut_len=16000*2`（2 s），训练样本先 crop/pad，再用 noisy 的标准差同时归一化 noisy/clean（`.research/upstream/lisennet_data_module.py:12-15,52-63`）。本地 `random_crop_pair()` 在 `speech_denoising/training/paper_lisennet.py:75-90` 保持了这个顺序。
- 本地 paper-track 的常量与优化器对应论文/官方实现：2 s、batch 4、100、AdamW 5e-4、betas (0.8,0.99)、StepLR gamma 0.98、clip 5.0、0.1 complex + 0.9 magnitude + 0.05 adversarial（`paper_lisennet.py:43-53,288-293`）。`configs/paper_recipes.yaml:52-66` 也记录了同一配方。论文正文没有直接打印 batch size 或 betas；这两项是配置/实现层证据。
- 官方模型的 GAN 更新顺序、real target=1、fake target 为 `(PESQ-1)/3.5` 后 clip 到 [0,1]、无效 PESQ 时只保留 real 项，与本地 `paper_lisennet.py:128-154` 和 `.research/upstream/lisennet_model.py:33-62,130-186` 一致。

### TF-GridNet

- `.research/tfgridnet.txt` 确认论文默认 STFT 为 32 ms window、8 ms hop、sqrt-Hann；16 kHz 使用 512-point DFT 和 257 个频点（`:705-716`）。
- 论文训练每个 epoch 从每个 mixture 取 4 s segment，以该 segment 的 sample variance 归一化到 1.0，并对 target 使用同一缩放因子；优化器 Adam、初始学习率 0.001、clip L2 1.0，验证损失连续 3 epochs 不改善时学习率减半（`:733-739`）。
- 论文在 loss comparison 中明确默认采用 Wav+Mag+MC（`:907-915`）。
- 论文列出的主要实验数据包括 WSJ0-2mix、SMS-WSJ、WHAMR!、WSJ0CAM-DEREVERB 和 L3DAS22（`:541-548,607-617,625-648`）。论文文本没有 VoiceBank-DEMAND recipe；全文可检索到的 epoch 叙述只有每 epoch 的采样/速度描述，没有总 epoch budget（`:733-739,1161-1163`）。所以仓库对 VoiceBank 的 TF-GridNet 运行不能标成论文原配方，也没有论文给出的 VoiceBank epoch 数可对齐。

## discrepancy

### LiSenNet

- **训练 split 不同。** 论文是 11,572/872 的随机说话人划分；仓库 paper track 使用 speaker-disjoint 的 10,778 train、794 validation、824 test（`docs/paper_alignment.md:62-63,81-83`）。这会影响结果可比性，虽不改变 100-epoch 结论。
- **尾部 partial batch 被丢弃。** 本地循环是 `range(0, len(order) - BATCH_SIZE + 1, BATCH_SIZE)`（`paper_lisennet.py:370`）。按当前 10,778 条训练样本和 batch 4，每 epoch 只处理 10,776 条，丢弃最后 2 条；官方 DataLoader 默认 `drop_last=False`（`.research/upstream/lisennet_data_module.py:98-100`），应保留最后一个 2-item batch。
- **验证归一化路径不同。** 官方非训练 Dataset 在 crop 之外对整条 noisy/clean 按 noisy std 归一化（`.research/upstream/lisennet_data_module.py:32-36,48-50`）。本地 `validate()` 直接把原始 `noisy_audio` 送入 wrapper 并用原始 clean 评分（`paper_lisennet.py:161-192`）；wrapper 会重新缩放输出，但这不是官方验证数据流的逐字复现。
- **验证抽样是仓库选择。** 本地通过 `--validation-limit` 固定抽取最多 200 条（`paper_lisennet.py:165-169`），这是运行策略，论文没有规定该 200-item 选择。
- **判别器 head 与 upstream 不同。** upstream 对两层 Linear 都使用 spectral normalization（`.research/upstream/lisennet_discriminator.py:41-49`）；本地 `speech_denoising/models/lisennet/discriminator.py:41-47` 的两层 `nn.Linear` 没有 spectral normalization。该差异会改变参数化和训练动力学。
- **判别器参数形状/API 也不完全同构。** upstream `LearnableSigmoid.slope` 是 shape `[1]`（`:5-13`），本地是 `[1,1]`（`discriminator.py:13-20`）；upstream 返回 5 个 block 输出，本地把 block 展平为 16 个 layer 输出（upstream `:20-59`、本地 `:28-56`）。末端标量可继续被 `fake[-1]` 使用，但 state_dict 结构不能视为官方逐项兼容。

### TF-GridNet

- 矩阵把论文数据集概括为 WSJ0-2mix、SMS-WSJ、WHAMR!、L3DAS22，但漏列了论文的 WSJ0CAM-DEREVERB speech-dereverberation setup（`.research/tfgridnet.txt:653-658`）。
- 仓库本地 profile 使用 `stride=256`（16 ms）、AdamW、无 scheduler、clip 5.0、negative SI-SNR（`configs/tfgridnet.yaml`、`configs/training_protocol.yaml`），与论文的 8 ms hop、Adam + plateau、clip 1.0、Wav+Mag+MC 不同。该 profile 是 VoiceBank adaptation/unified recipe，不能写成论文复现。

### GTCRN

- 现有 `docs/paper_alignment.md` 矩阵把若干 GTCRN 数字写成论文值，但本次无法从本地论文文件复核这些数字（见 unknown）。在证据恢复前，不应把这些数字当作本地独立确认的审计结论。

## unknown

### GTCRN 论文证据不可用

- `.research/gtcrn.pdf` 不是 PDF：文件开头是 HTML，内容包含 `Error Code: 418`，大小仅 16,794 bytes，且没有 PDF header、page objects 或可提取正文。因此本次无法独立核对矩阵中的 VCTK-DEMAND 训练/测试数量、validation 数量、STFT、batch 4/16、Adam 1e-3、5-epoch plateau、loss 系数、参数量 23.7 K、39.6 MMAC/s 等任何一项。
- 同理，“论文没有 VCTK epoch budget”这一结论无法由该损坏/付费墙 HTML 文件直接证明。仓库的 `configs/paper_recipes.yaml:34-39`、`docs/gtcrn_paper_track_plan_2026-11-09.md:161-165,342-358` 将 `epochs_vctk` 设为 null，并记录官方仓库没有绑定 VCTK trainer；这属于仓库侧证据，不能替代论文原文核验。需要重新取得可读的 GTCRN 论文后再把矩阵数字标为 confirmed。

### LiSenNet 配置来源边界

- 本 checkout 的 `.research/upstream` 没有官方 `config.yaml`；其 DataModule 只接收 `batch_size` 参数，未硬编码 4（`.research/upstream/lisennet_data_module.py:66-99`）。因此 batch 4、betas (0.8,0.99) 的“论文值”不能仅凭 `lisennet.txt` 正文确认，只能凭用户指出的官方 config、仓库 `configs/paper_recipes.yaml` 和 paper-track 常量确认。100 epochs 则有论文正文的直接证据。

## 结论

LiSenNet 的 paper-track 目标确实应是 100 epochs、2 s crop、batch 4、AdamW 5e-4、betas (0.8,0.99)、StepLR 0.98、clip 5.0 和 0.9/0.1/0.05 GAN loss；但当前实现仍有尾 batch、验证归一化、数据 split 和判别器 head 的已确认偏差。TF-GridNet 的论文超参文本可核对，论文没有 VoiceBank epoch budget；GTCRN 的本地论文文件是 418 错误页，矩阵数字与无 epoch budget 结论目前应保留为 unknown。
