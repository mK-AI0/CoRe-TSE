"""No-data smoke test for the core neural modules."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core_tse"))

from models import AudioEncoder, EEGEncoder  # noqa: E402
from loss import loss_nce, loss_nce_in_batch  # noqa: E402


def main() -> None:
    # The two encoders must produce length-aligned sequences for contrastive loss.
    eeg = torch.randn(4, 20, 64)
    audio = torch.randn(4, 1, 80, 20)
    eeg_z = EEGEncoder()(eeg)
    audio_z = AudioEncoder()(audio)
    assert eeg_z.shape[0] == audio_z.shape[0] == 4
    assert torch.isfinite(loss_nce(eeg_z, audio_z, trials_per_batch=1, windows_per_trial=4))
    assert torch.isfinite(loss_nce_in_batch(eeg_z, audio_z))
    print("CORE-TSE smoke test passed")


if __name__ == "__main__":
    main()
