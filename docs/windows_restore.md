# 新电脑恢复与 Phase 2.8 操作（Windows CMD）

上次已推送提交：32c1b59。恢复自己的 benchmark 仓库，不需重新运行
Phase 2.2–2.7 安装器。Phase 2.8 是文档审计安装，只需 Git 和 Python 3.10+；
本步不需要 PyTorch、PESQ、测试集、GTCRN 权重或 Visual Studio Build Tools。

只复制下面代码框内的命令，每次一条。不要复制 C:\...> 提示符或运行输出。

## 1. 检查基础工具

```bat
git --version
```

```bat
python --version
```

如果 Python 命令不存在，也可检查 `py --version`；若 py 可用，将后续
`python` 替换为 `py`。如果两者都没有，安装 Python 后重新打开 CMD。
Python 官方下载：https://www.python.org/downloads/windows/
Git 官方安装：https://git-scm.com/install/windows
安装 Python 时启用 PATH 选项。已有合适版本可直接使用。

## 2. 恢复仓库

新电脑没有项目目录时执行：

```bat
git clone https://github.com/simonzhw32-cyber/edge-speech-denoising-benchmark.git "%USERPROFILE%\edge-speech-denoising-benchmark"
```

Private 仓库需要登录有权限的 GitHub 账号。使用 Git 提供的登录流程，
不需要将仓库改为 public，也不要把账号密码或 token 发到聊天中。
如果已经复制了完整 Git 仓库，跳过 clone；先检查工作区后再拉取。

```bat
cd /d "%USERPROFILE%\edge-speech-denoising-benchmark"
```

```bat
git log -1 --oneline
```

```bat
git status --short
```

预期最新提交是 32c1b59，工作区没有输出。若有更新或未提交修改，保留它们；
安装器会检查已知 README 和 TF-GridNet 源码指纹，不会强行覆盖其他版本。
若项目位于其他目录，cd 到实际仓库根目录即可。

## 3. 安装 Phase 2.8 审计文档

将 phase28_record_tfgridnet_audit.py 保存到下载文件夹，然后执行：

```bat
python "%USERPROFILE%\Downloads\phase28_record_tfgridnet_audit.py" --repo "%CD%"
```

```bat
python "%USERPROFILE%\Downloads\phase28_record_tfgridnet_audit.py" --repo "%CD%" --check
```

--check 是只读检查，不依赖 PyTorch。它核对文档、JSON、基线源码指纹和
候选结构审计的一致性。它不重新下载候选，也不声称验证本机推理质量。

查看审计文档：

```bat
notepad docs\tfgridnet_checkpoint_audit.md
```

通过检查后，提交目标仅有 README 和三个 docs 文件：

```bat
git add README.md docs/tfgridnet_checkpoint_audit.md docs/tfgridnet_checkpoint_audit.json docs/windows_restore.md
```

```bat
git commit -m "Record TF-GridNet pretrained checkpoint provenance audit"
```

```bat
git push origin main
```

新电脑第一次提交可能需要设置 Git 作者信息。请使用自己的姓名和 GitHub
邮箱（或 GitHub 设置中提供的 noreply 邮箱），只设置此仓库的 git config；
不要猜测邮箱，不需发送 token。代码与报告以 GitHub 推送结果为准。

## 后续运行模型时再恢复

再建立本机虚拟环境，参考 benchmark_reports/gtcrn_windows_environment.txt
选择兼容依赖。固定 GTCRN checkpoint、数据集版本与文件哈希已在项目中记录。
没有备份的 data / checkpoints / results 需重新获取；不影响已提交源码和报告。
新机器 RTF 另存一份报告，不能当成原 Windows 测试机器的 RTF。
GTCRN 824 条已有报告不需仅因换电脑就重新评测。

下一项：DNS 候选预训练适配与原生输出一致性验证。训练仍未开始。
