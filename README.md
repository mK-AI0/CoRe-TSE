# CORE-TSE

CORE-TSE（Complementary EEG-Speech Representation Learning for EEG-Guided Target Speaker Extraction）代码仓库。

方法在 Stage 1 中分别以两种负样本策略训练 EEG--speech encoder：View-A 使用同一 trial 内的时间负样本，View-B 使用不受 trial 限制的 in-batch 负样本；在 Stage 2 中冻结两个表示并拼接为 TSE extractor 的条件特征，同时固定等权融合两分支的选择 margin。

## 仓库范围

本仓库包含论文方法及论文中出现的消融实现：

- `CoRe-TSE`：独立的 A/B 双分支与固定等权融合；
- `Single-A`：仅 View-A；
- `Single-B`：仅 View-B；
- `Single-C`：同步 ignored-speech 负样本诊断；
- `Shared-AB`：单编码器上的 A/B 联合目标；
- `Dual-AA`：两个独立 View-A 分支的容量对照。

## 安装

建议使用 Python 3.10 和与本机 CUDA 匹配的 PyTorch。先按 [PyTorch 官方说明](https://pytorch.org/get-started/locally/) 安装 PyTorch，再安装其余依赖：

```bash
pip install -r requirements.txt
python tests/smoke_test.py
```

## 数据集

论文使用 USTC、KUL 和 DTU EEG-auditory-attention 数据集，并沿用其公开协议：USTC 为 subject-adaptive 协议，KUL/DTU 为 strict cross-trial 协议。数据的来源、使用边界和本实现期望的输入字段见 [data/README.md](data/README.md)。

可从原始数据集作者或其官方发布渠道获得数据并完成预处理。运行前必须在命令行显式提供本机 `metadata_path`、`audio_dir`、`eeg_dir` 或 `data_dir`；代码不含任何默认私有路径。

## 运行

KUL/DTU 的 Single-A 两阶段训练：

```bash
bash scripts/run_core_tse.sh KUL \
  metadata_path=/path/to/metadata.csv \
  audio_dir=/path/to/audio \
  eeg_dir=/path/to/eeg \
  log_dir=outputs/kul_single_a
```

Single-B 仅改变 Stage-1 的负采样策略：

```bash
bash scripts/run_ablation.sh single_b DTU \
  metadata_path=/path/to/metadata.csv \
  audio_dir=/path/to/audio \
  eeg_dir=/path/to/eeg \
  log_dir=outputs/dtu_single_b
```

CoRe-TSE 与 Dual-AA 需要同一数据划分、同一 fold 对齐的两个 Stage-1 checkpoint。调用 `scripts/run_ablation.sh core_tse` 或 `dual_aa` 时，通过 `attended_stage1_path=...` 与 `inbatch_stage1_path=...` 显式传入。USTC 的入口为 `core_tse/ustc/train.py`，配置位于 `configs/ustc_*.yaml`。

## 代码结构

```text
core_tse/          KUL/DTU 核心实现与 USTC 独立实现
configs/           数据集、主方法与论文消融配置
scripts/           KUL/DTU 训练与消融入口
tests/             不依赖数据的 smoke test
data/README.md     数据来源与输入接口说明
```

## 许可

本仓库沿用 Apache-2.0 许可；部分 extractor 代码源自 ClearerVoice-Studio，相关保留声明位于源文件和 [LICENSE](LICENSE) 中。
