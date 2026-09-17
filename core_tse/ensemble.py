"""KUL/DTU TRUST 双 Stage-1 编码器等权集成。"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import Extractor, EEGEncoder, AudioEncoder
from trainers import Stage2Trainer
from metrics import cal_SISDR


def _interp(eeg_emb, audio_emb):
    if audio_emb.size(1) != eeg_emb.size(1):
        audio_emb = F.interpolate(audio_emb.transpose(1, 2), size=eeg_emb.size(1),
                                  mode='linear', align_corners=False).transpose(1, 2)
        audio_emb = F.normalize(audio_emb, p=2, dim=-1)
    return audio_emb


def load_stage1_branch(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if not {'eeg_encoder', 'audio_encoder'}.issubset(checkpoint):
        raise ValueError(f'Invalid Stage-1 checkpoint: {path}')
    eeg_encoder, audio_encoder = EEGEncoder(), AudioEncoder()
    eeg_encoder.load_state_dict(checkpoint['eeg_encoder'], strict=True)
    audio_encoder.load_state_dict(checkpoint['audio_encoder'], strict=True)
    for module in (eeg_encoder, audio_encoder):
        module.to(device).eval()
        for parameter in module.parameters():
            parameter.requires_grad = False
    return eeg_encoder, audio_encoder


def validation_reliability_weights(accuracies, epsilon=1e-3):
    """Convert validation selection accuracies into fixed positive weights.

    This is the BASE validation-weight rule: chance-level or worse branches
    retain only epsilon evidence, then the two evidences are normalized.
    """
    values = torch.tensor(accuracies, dtype=torch.float32)
    evidence = (values - 50.0).clamp_min(0.0) + epsilon
    return evidence / evidence.sum()


class EqualWeightEEGExtractor(Extractor):
    """将两个冻结 EEG 分支按固定（等权或验证加权）规则拼接。"""
    def __init__(self, args, eeg_encoders, branch_weights=None):
        if len(eeg_encoders) != 2:
            raise ValueError('This experiment requires exactly two EEG branches.')
        super().__init__(args)
        hidden = {encoder.hidden for encoder in eeg_encoders}
        if len(hidden) != 1:
            raise ValueError(f'Incompatible EEG hidden dimensions: {hidden}')
        del self.eeg_encoder
        self.eeg_encoders = nn.ModuleList(eeg_encoders)
        self.num_branches = 2
        if branch_weights is None:
            branch_weights = torch.full((self.num_branches,), 1.0 / self.num_branches)
        branch_weights = torch.as_tensor(branch_weights, dtype=torch.float32)
        if (branch_weights.numel() != self.num_branches or not torch.isfinite(branch_weights).all()
                or (branch_weights <= 0).any()):
            raise ValueError('Branch weights must be finite, positive, and match the two branches.')
        # Kept non-persistent so fold-specific validation evidence is never
        # overwritten by a generic Stage-2 checkpoint during re-evaluation.
        self.register_buffer('branch_weights', branch_weights / branch_weights.sum(), persistent=False)
        self.eeg_hidden = next(iter(hidden)) * self.num_branches
        self.front_fusion = nn.Conv1d(self.B + self.eeg_hidden, self.B, 1, bias=False)
        self.block_fusion = nn.ModuleList([
            nn.Conv1d(self.B + self.eeg_hidden, self.B, 1, bias=False) for _ in range(self.R)
        ])

    def eeg_features(self, eeg):
        # 2 * 0.5 keeps each branch at its original feature scale, exactly as BASE.
        return torch.cat([self.num_branches * weight * encoder(eeg)
                          for weight, encoder in zip(self.branch_weights, self.eeg_encoders)], dim=-1)

    def _estimate_mask(self, mixture_w, eeg):
        batch_size, _, frames = mixture_w.size()
        y = self.bottleneck_conv1x1(self.layer_norm(mixture_w))
        eeg_feat = F.interpolate(self.eeg_features(eeg).transpose(1, 2), size=frames,
                                 mode='linear', align_corners=False)
        y = self.front_fusion(torch.cat((y, eeg_feat), dim=1))
        y, gap = self._segmentation(y, self.K)
        eeg_seg, _ = self._segmentation(eeg_feat, self.K)
        eeg_flat = eeg_seg.flatten(2)
        for rnn, fusion in zip(self.dual_rnn, self.block_fusion):
            y = rnn(y)
            chunk, segment = y.size(2), y.size(3)
            y = fusion(torch.cat((y.flatten(2), eeg_flat), dim=1)).reshape(
                batch_size, self.B, chunk, segment)
        y = self._over_add(y, gap)
        return F.relu(self.mask_conv1x1(self.prelu(y))).unsqueeze(1)


@torch.no_grad()
def ensemble_margin(extractor, audio_encoders, eeg, mel_att, mel_ign):
    margins = []
    for eeg_encoder, audio_encoder in zip(extractor.eeg_encoders, audio_encoders):
        eeg_z = eeg_encoder(eeg)
        att_z = _interp(eeg_z, audio_encoder(mel_att))
        ign_z = _interp(eeg_z, audio_encoder(mel_ign))
        margins.append((eeg_z * att_z).sum(-1).mean(-1) - (eeg_z * ign_z).sum(-1).mean(-1))
    return (extractor.branch_weights[:, None] * torch.stack(margins, dim=0)).sum(dim=0)


@torch.no_grad()
def branch_selection_accuracy(eeg_encoder, audio_encoder, loader, device):
    """Selection Acc of one frozen Stage-1 branch on the validation split."""
    eeg_encoder.eval(); audio_encoder.eval()
    correct = total = 0
    for eeg, mel_att, mel_ign in loader:
        eeg = eeg.to(device); mel_att = mel_att.to(device); mel_ign = mel_ign.to(device)
        eeg_z = eeg_encoder(eeg)
        att_z = _interp(eeg_z, audio_encoder(mel_att))
        ign_z = _interp(eeg_z, audio_encoder(mel_ign))
        margin = (eeg_z * att_z).sum(-1).mean(-1) - (eeg_z * ign_z).sum(-1).mean(-1)
        correct += (margin > 0).sum().item(); total += eeg.size(0)
    return 100.0 * correct / max(total, 1)


@torch.no_grad()
def ensemble_selection_accuracy(extractor, audio_encoders, loader, device):
    extractor.eval()
    correct = total = 0
    for eeg, mel_att, mel_ign in loader:
        margin = ensemble_margin(extractor, audio_encoders, eeg.to(device),
                                 mel_att.to(device), mel_ign.to(device))
        correct += (margin > 0).sum().item()
        total += eeg.size(0)
    return 100.0 * correct / max(total, 1)


class EnsembleStage2Trainer(Stage2Trainer):
    def __init__(self, args, extractor, audio_encoders, train_loader, val_loader, test_loader):
        super().__init__(args, extractor, audio_encoders[0], train_loader, val_loader, test_loader)
        self.audio_encoders = list(audio_encoders)

    def _set_train(self):
        self.extractor.train()
        for encoder in self.extractor.eeg_encoders:
            encoder.eval()

    def _run_one_epoch(self, data_loader, train=True):
        total_loss = total_correct = total_samples = 0
        for a_mix, a_att, a_ign, eeg, mel_att, mel_ign in data_loader:
            a_mix, a_att, a_ign = a_mix.to(self.args.device), a_att.to(self.args.device), a_ign.to(self.args.device)
            eeg, mel_att, mel_ign = eeg.to(self.args.device), mel_att.to(self.args.device), mel_ign.to(self.args.device)
            a_est = self.extractor(a_mix, eeg)
            margin = ensemble_margin(self.extractor, self.audio_encoders, eeg, mel_att, mel_ign)
            sisdr_att, sisdr_ign = cal_SISDR(a_att, a_est), cal_SISDR(a_ign, a_est)
            confidence = torch.tanh(5.0 * margin.abs())
            loss = -(confidence * torch.where(margin >= 0, sisdr_att, sisdr_ign)).mean()
            if train:
                self.optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in self.extractor.parameters() if p.requires_grad],
                                                self.args.clip_grad_norm)
                self.optimizer.step()
            correct = sisdr_att > sisdr_ign
            batch_size = a_mix.size(0)
            total_loss += loss.item() * batch_size
            total_correct += correct.sum().item()
            total_samples += batch_size
        return total_loss / total_samples, 100.0 * total_correct / total_samples
