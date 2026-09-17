from __future__ import annotations

import numpy as np
import torch
from pesq import pesq
from pystoi import stoi


def sisdr(reference: torch.Tensor, estimate: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    reference = reference - reference.mean(dim=-1, keepdim=True)
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    scale = (estimate * reference).sum(dim=-1, keepdim=True) / (reference.square().sum(dim=-1, keepdim=True) + eps)
    target = scale * reference
    noise = estimate - target
    return 10 * torch.log10(target.square().sum(dim=-1) / (noise.square().sum(dim=-1) + eps) + eps)


def safe_pesq(sr: int, ref: np.ndarray, est: np.ndarray) -> float | None:
    try:
        return float(pesq(sr, ref, est, "wb" if sr == 16000 else "nb"))
    except Exception:
        return None


def safe_stoi(sr: int, ref: np.ndarray, est: np.ndarray) -> float | None:
    try:
        return float(stoi(ref, est, sr, extended=False))
    except Exception:
        return None


def pcc(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])
