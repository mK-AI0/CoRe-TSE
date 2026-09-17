# CORE-TSE

Code repository for CORE-TSE (Complementary EEG-Speech Representation Learning for EEG-Guided Target Speaker Extraction).

In Stage 1, the method trains EEG-speech encoders with two negative-sampling strategies: View-A uses temporal negatives within the same trial, whereas View-B uses unrestricted in-batch negatives. In Stage 2, both representations are frozen and concatenated as conditioning features for the TSE extractor, while their selection margins are fused with fixed equal weights.

## Repository Scope

This repository includes the method and the ablations reported in the paper:

- `CoRe-TSE`: independently pretrained A/B branches with fixed equal-weight fusion;
- `Single-A`: View-A only;
- `Single-B`: View-B only;
- `Single-C`: a diagnostic using synchronized ignored-speech negatives;
- `Shared-AB`: joint A/B objectives in a single encoder;
- `Dual-AA`: a capacity control with two independently initialized View-A branches.

The repository does not distribute datasets, preprocessing pipelines, pretrained models, checkpoints, training logs, or result files.

## Installation

Python 3.10 and a PyTorch build compatible with the local CUDA environment are recommended. Install PyTorch according to the [official instructions](https://pytorch.org/get-started/locally/), then install the remaining dependencies:

```bash
pip install -r requirements.txt
python tests/smoke_test.py
```

## Datasets

The paper uses the USTC, KUL, and DTU EEG auditory-attention datasets under their published protocols: USTC follows a subject-adaptive protocol, whereas KUL and DTU use strict cross-trial protocols. Dataset sources, use boundaries, and the expected input interface are described in [data/README.md](data/README.md).

Obtain the datasets from their original authors or official release channels and perform preprocessing independently. Before running the code, explicitly provide local `metadata_path`, `audio_dir`, `eeg_dir`, or `data_dir` paths. The repository contains no machine-specific default paths.

## Usage

Two-stage Single-A training on KUL or DTU:

```bash
bash scripts/run_core_tse.sh KUL \
  metadata_path=/path/to/metadata.csv \
  audio_dir=/path/to/audio \
  eeg_dir=/path/to/eeg \
  log_dir=outputs/kul_single_a
```

Single-B changes only the Stage-1 negative-sampling strategy:

```bash
bash scripts/run_ablation.sh single_b DTU \
  metadata_path=/path/to/metadata.csv \
  audio_dir=/path/to/audio \
  eeg_dir=/path/to/eeg \
  log_dir=outputs/dtu_single_b
```

CoRe-TSE and Dual-AA require two fold-aligned Stage-1 checkpoints trained on the same data split. When calling `scripts/run_ablation.sh core_tse` or `dual_aa`, provide `attended_stage1_path=...` and `inbatch_stage1_path=...`. The USTC entry point is `core_tse/ustc/train.py`, with configurations in `configs/ustc_*.yaml`.

## Project Structure

```text
core_tse/          KUL/DTU core implementation and a separate USTC implementation
configs/           Dataset, method, and paper-ablation configurations
scripts/           KUL/DTU training and ablation entry points
tests/             Data-independent smoke test
data/README.md     Dataset sources and expected input interface
```

## License

Original CORE-TSE code and modifications are released under Apache-2.0. Portions adapted from BASE-USTC, TRUST-TSE, and ClearerVoice-Studio retain their respective notices; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
