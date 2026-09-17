import torch
import torch.nn.functional as F


def loss_nce(
    eeg_z,
    mel_tgt_z,
    temperature=0.07,
    trials_per_batch=None,
    windows_per_trial=None,
):
    """
    eeg_z, mel_tgt_z: (B, T, D)
    B must be trials_per_batch * windows_per_trial

    batch order must be:
        [trial1 windows] + [trial2 windows] + ... + [trialP windows]
    """
    B, T, _ = eeg_z.shape
    P = trials_per_batch
    K = windows_per_trial

    if P is None or K is None:
        raise ValueError('trials_per_batch and windows_per_trial must be provided.')
    if B != P * K:
        raise ValueError(f'Batch size mismatch: got B={B}, expected {P*K}.')

    eeg_z = F.normalize(eeg_z, p=2, dim=-1).reshape(P, K, -1)
    mel_tgt_z = F.normalize(mel_tgt_z, p=2, dim=-1).reshape(P, K, -1)

    logits = torch.matmul(eeg_z, mel_tgt_z.transpose(1, 2)) / (temperature * T)
    labels = torch.arange(K, device=logits.device).expand(P, K)

    loss_i = F.cross_entropy(logits.reshape(P * K, K), labels.reshape(-1))
    loss_j = F.cross_entropy(logits.transpose(1, 2).reshape(P * K, K), labels.reshape(-1))
    return 0.5 * (loss_i + loss_j)


def loss_nce_in_batch(eeg_z, mel_tgt_z, temperature=0.07):
    """Standard cross-modal in-batch NCE without trial-constrained negatives.

    Every other sample in the minibatch is a negative candidate, irrespective
    of subject or trial.  This is the ablation counterpart to ``loss_nce``.
    """
    batch_size, time_steps, _ = eeg_z.shape
    eeg_z = F.normalize(eeg_z, p=2, dim=-1).reshape(batch_size, -1)
    mel_tgt_z = F.normalize(mel_tgt_z, p=2, dim=-1).reshape(batch_size, -1)
    logits = torch.matmul(eeg_z, mel_tgt_z.transpose(0, 1)) / (temperature * time_steps)
    labels = torch.arange(batch_size, device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.transpose(0, 1), labels))


def loss_nce_synchronous_unattended(eeg_z, mel_att_z, mel_ign_z, temperature=0.07):
    """Binary NCE with the temporally synchronous unattended speaker as negative.

    Each EEG window is contrasted only with its paired attended and unattended
    speech windows.  In particular, this mode deliberately does *not* add
    other minibatch items, trials, or time offsets as negatives, so it isolates
    the effect of replacing attended-speaker negatives with the interferer.
    """
    if eeg_z.shape != mel_att_z.shape or eeg_z.shape != mel_ign_z.shape:
        raise ValueError("EEG, attended audio, and unattended audio embeddings must have the same shape.")
    eeg_z = F.normalize(eeg_z, p=2, dim=-1)
    mel_att_z = F.normalize(mel_att_z, p=2, dim=-1)
    mel_ign_z = F.normalize(mel_ign_z, p=2, dim=-1)
    time_steps = eeg_z.size(1)
    positive = (eeg_z * mel_att_z).sum(dim=(-1, -2)) / (temperature * time_steps)
    negative = (eeg_z * mel_ign_z).sum(dim=(-1, -2)) / (temperature * time_steps)
    logits = torch.stack((positive, negative), dim=1)
    labels = torch.zeros(eeg_z.size(0), dtype=torch.long, device=eeg_z.device)
    return F.cross_entropy(logits, labels)
