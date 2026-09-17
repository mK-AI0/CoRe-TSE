#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/run_core_tse.sh KUL <Hydra overrides...>
#        bash scripts/run_core_tse.sh DTU <Hydra overrides...>
# The required local overrides are metadata_path=..., audio_dir=..., eeg_dir=....

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${1:?usage: $0 {KUL|DTU} [Hydra overrides...]}"
shift

case "$DATASET" in
  KUL) CONFIG=config_KUL ;;
  DTU) CONFIG=config_DTU ;;
  *) echo "dataset must be KUL or DTU" >&2; exit 2 ;;
esac

cd "$ROOT/core_tse"
python train.py --config-path ../configs --config-name "$CONFIG" "$@"
