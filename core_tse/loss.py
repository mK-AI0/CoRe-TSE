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
    """对称的全局 in-batch NCE。

    这是论文 5.4.1 的 ``In-batch (All speakers)`` 对照：一个 EEG
    窗口的正样本是其配对 attended 语音，当前 minibatch 中其余所有
    attended 语音窗口都是负样本，不限制说话人、trial 或时间位置。
    ``loss_nce`` 则保留原始的同 trial attended-speaker 负样本策略。
    """
    if eeg_z.shape != mel_tgt_z.shape:
        raise ValueError('EEG and attended-audio embeddings must have the same shape.')

    batch_size, time_steps, _ = eeg_z.shape
    eeg_z = F.normalize(eeg_z, p=2, dim=-1).reshape(batch_size, -1)
    mel_tgt_z = F.normalize(mel_tgt_z, p=2, dim=-1).reshape(batch_size, -1)
    logits = torch.matmul(eeg_z, mel_tgt_z.transpose(0, 1)) / (temperature * time_steps)
    labels = torch.arange(batch_size, device=logits.device)
    return 0.5 * (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.transpose(0, 1), labels)
    )
