# Windows 项目恢复

仓库：https://github.com/simonzhw32-cyber/edge-speech-denoising-benchmark
保持 private。代码、配置、审计和已提交报告由 Git 恢复；checkpoint、
数据集、results 和 Python 环境需要按实际任务重新准备。

只复制代码块里的命令，不要复制终端路径、PASS 输出或 Git 日志。
每条命令成功后再继续。不要因为换电脑而重复运行旧阶段安装器。

## 恢复代码

```bat
git --version
python --version
```

未安装工具时，使用官方 Windows 安装包，Python 安装时启用 PATH：
https://git-scm.com/install/windows
https://www.python.org/downloads/windows/
安装后重新打开 CMD。Python 3.10+；当前集成在现代 PyTorch CPU FP32 检查。

新电脑没有仓库时：

```bat
git clone https://github.com/simonzhw32-cyber/edge-speech-denoising-benchmark.git "%USERPROFILE%\edge-speech-denoising-benchmark"
```

按 Git 的浏览器登录提示认证。已有目录时不要重复 clone；保留未提交修改，
工作区干净时可 `git pull --ff-only`。

```bat
cd /d "%USERPROFILE%\edge-speech-denoising-benchmark"
git log -1 --oneline
git status --short
```

以最新已推送提交为准，不要强行 reset 回旧阶段。Phase 2.9 安装器的已确认
起点为 `0e74ad0`，并检查 Phase 2.8 文档及 TF-GridNet 源码指纹。
首次提交缺身份时，设置当前仓库的 `git config user.name` 和 `user.email`，
使用自己的名字/邮箱或 GitHub noreply 邮箱。无需在其他电脑沿用全局设置。

## Phase 2.9 依赖与 DNS 权重

源码已经从 Git 恢复时，不需要安装器。若正在首次迁移 Phase 2.9，
先将 phase29_install_tfgridnet_pretrained.py 保存到 Downloads，然后执行：

```bat
python "%USERPROFILE%\Downloads\phase29_install_tfgridnet_pretrained.py" --repo "%CD%"
```

源码安装器只写文件和备份；不会联网、安装依赖、加载权重、训练或推送。
随后从仓库根目录执行：

```bat
python -m pip install torch==2.14.1 --index-url https://pypi.org/simple
python -m pip install -r requirements-tfgridnet.txt --index-url https://pypi.org/simple
python -m scripts.fetch_tfgridnet_checkpoint
python -m scripts.smoke_tfgridnet_pretrained
python -m scripts.smoke_tfgridnet
```

fetch 仅下载固定 DNS checkpoint（约 10.3 MB）并核对哈希；已有正确文件会复用。
遇到网络或证书错误，保留错误输出，不关闭 TLS 验证。smoke 使用合成输入
验证真实预训练权重，报告 `results/tfgridnet_dns_pretrained_check.json` 无质量分数。
历史参考会出现 `stft(return_complex=False)` 弃用提示；以 PASS/异常结果为准。

无需仅因换电脑重跑 GTCRN 824 条评测。恢复 GTCRN 权重时使用
`python -m scripts.fetch_gtcrn_checkpoint`，不要运行旧源码迁移安装器。
需要 VoiceBank 时再使用 `scripts.prepare_data` 和 benchmark 依赖；本阶段不需要。
同一 checkpoint 在不同电脑上的质量/RTF 验证应记录环境，RTF 不混用旧机器报告。

## 下次继续时保留

提交并推送源代码后，记录 `git log -1 --oneline`。将这一行和当前阶段发给
助手即可继续。需要保留未提交结果时，单独备份 ignored 的 results/checkpoints/data；
不要误以为 git push 已上传这些资产。

参见 `speech_denoising/models/tfgridnet/PRETRAINED.md` 的配置、参考范围和限制。
