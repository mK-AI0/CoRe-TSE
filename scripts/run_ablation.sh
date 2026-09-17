#!/usr/bin/env bash
set -euo pipefail

# Stage-1 Single-A / Single-B examples for KUL or DTU.
# Shared-AB and Single-C are USTC controls; Dual-AA/CoRe stage 2 requires
# fold-matched branch checkpoints supplied through Hydra overrides.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VARIANT="${1:?usage: $0 {single_a|single_b|core_tse|dual_aa} {KUL|DTU} [Hydra overrides...]}"
DATASET="${2:?missing dataset}"
shift 2

case "$DATASET" in
  KUL) BASE=config_KUL ;;
  DTU) BASE=config_DTU ;;
  *) echo "dataset must be KUL or DTU" >&2; exit 2 ;;
esac

case "$VARIANT" in
  single_a)
    bash "$ROOT/scripts/run_core_tse.sh" "$DATASET" "stage1.negative_mode=attended_same_trial" "$@"
    ;;
  single_b)
    bash "$ROOT/scripts/run_core_tse.sh" "$DATASET" "stage1.negative_mode=in_batch" "$@"
    ;;
  core_tse|dual_aa)
    CONFIG="${BASE}_ensemble_attended_inbatch"
    if [[ "$VARIANT" == dual_aa ]]; then CONFIG="${BASE}_ensemble_duplicate_capacity"; fi
    cd "$ROOT/core_tse"
    python train_ensemble.py --config-path ../configs --config-name "$CONFIG" "$@"
    ;;
  *) echo "unknown variant: $VARIANT" >&2; exit 2 ;;
esac
