# This file is adapted from ClearerVoice-Studio
# (https://github.com/modelscope/ClearerVoice-Studio),
# licensed under the Apache License 2.0.
# Original work Copyright (c) Alibaba Group.
# CORE-TSE modifications Copyright (c) 2026 mK-AI0.


import torch
import numpy as np
from pesq import pesq
from pystoi import stoi


EPS = 1e-6

def cal_SISDR(source, estimate_source):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        source: torch tensor, [batch size, sequence length]
        estimate_source: torch tensor, [batch size, sequence length]
    Returns:
        SISNR, [batch size]
    """
    assert source.size() == estimate_source.size()
    # Step 1. Zero-mean norm
    source = source - torch.mean(source, axis = -1, keepdim=True)
    estimate_source = estimate_source - torch.mean(estimate_source, axis = -1, keepdim=True)
    # Step 2. SI-SNR
    # s_target = <s', s>s / ||s||^2
    ref_energy = torch.sum(source ** 2, axis = -1, keepdim=True) + EPS
    proj = torch.sum(source * estimate_source, axis = -1, keepdim=True) * source / ref_energy
    # e_noise = s' - s_target
    noise = estimate_source - proj
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    ratio = torch.sum(proj ** 2, axis = -1) / (torch.sum(noise ** 2, axis = -1) + EPS)
    sisnr = 10 * torch.log10(ratio + EPS)
    return sisnr


def cal_pesq(sr: int, pesq_mode: str, ref_1d: np.ndarray, deg_1d: np.ndarray):
    try:
        return float(pesq(sr, ref_1d.astype(np.float32), deg_1d.astype(np.float32), pesq_mode))
    except Exception:
        return None


def cal_stoi(sr: int, ref_1d: np.ndarray, deg_1d: np.ndarray):
    try:
        return float(stoi(ref_1d.astype(np.float32), deg_1d.astype(np.float32), sr, extended=False))
    except Exception:
        return None
