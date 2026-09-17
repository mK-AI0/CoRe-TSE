# 数据集

本仓库不包含 EEG、音频、metadata、HDF5 文件或任何派生数据。

论文使用以下公开发布的数据及其原始协议：

- USTC：BASE-USTC 数据集及其 subject-adaptive 协议，见论文参考文献 [13]。
- KUL：KUL EEG auditory-attention 数据集，见论文参考文献 [15]。
- DTU：DTU auditory-attention 数据集，见论文参考文献 [16]。

请根据各数据集作者的访问要求自行获取和处理数据。本实现要求调用者在配置中显式提供 `metadata_path`、`audio_dir`、`eeg_dir`（KUL/DTU），或 USTC 已处理样本所在的 `data_dir`。仓库不会提供这些文件、生成脚本或默认机器路径。
