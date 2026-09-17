"""Frozen Stage-1 encoder ensembles for TRUST-USTC Stage-2 training."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from models import Extractor, EEGEncoder, AudioEncoder
from trainers import MelFrontend, Stage2Trainer, _interp
from metrics import sisdr


def load_stage1_branch(path: str | Path, device: str):
    """Load one compatible, frozen EEG/audio Stage-1 branch."""
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {"eeg_encoder", "audio_encoder"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"Invalid Stage-1 checkpoint {path}: missing {sorted(missing)}")
    eeg_encoder, audio_encoder = EEGEncoder(), AudioEncoder()
    eeg_encoder.load_state_dict(checkpoint["eeg_encoder"], strict=True)
    audio_encoder.load_state_dict(checkpoint["audio_encoder"], strict=True)
    eeg_encoder.to(device).eval(); audio_encoder.to(device).eval()
    for module in (eeg_encoder, audio_encoder):
        for parameter in module.parameters():
            parameter.requires_grad = False
    return eeg_encoder, audio_encoder


def validation_reliability_weights(accuracies: list[float], epsilon: float = 1e-3) -> torch.Tensor:
    """Turn validation attended-selection accuracies into fixed positive weights."""
    values = torch.tensor(accuracies, dtype=torch.float32)
    evidence = (values - 50.0).clamp_min(0.0) + epsilon
    return evidence / evidence.sum()


class MultiEEGExtractor(Extractor):
    """Extractor whose frozen EEG condition is a feature concatenation ensemble."""

    def __init__(self, cfg, eeg_encoders: list[EEGEncoder], branch_weights: torch.Tensor | None = None):
        if len(eeg_encoders) < 2:
            raise ValueError("MultiEEGExtractor requires at least two Stage-1 EEG encoders.")
        super().__init__(cfg)
        hidden_sizes = {encoder.hidden for encoder in eeg_encoders}
        if len(hidden_sizes) != 1:
            raise ValueError(f"All EEG encoders must share one hidden size, got {hidden_sizes}.")
        # The parent owns a single randomly initialized encoder; never retain it.
        del self.eeg_encoder
        self.eeg_encoders = nn.ModuleList(eeg_encoders)
        self.num_branches = len(eeg_encoders)
        if branch_weights is None:
            branch_weights = torch.full((self.num_branches,), 1.0 / self.num_branches)
        if branch_weights.numel() != self.num_branches or not torch.isfinite(branch_weights).all() or (branch_weights <= 0).any():
            raise ValueError("Branch weights must be finite, positive, and match the branch count.")
        # A general checkpoint must not overwrite fold-specific validation weights.
        self.register_buffer("branch_weights", branch_weights.float() / branch_weights.sum(), persistent=False)
        self.eeg_hidden = next(iter(hidden_sizes)) * self.num_branches
        self.front_fusion = nn.Conv1d(self.B + self.eeg_hidden, self.B, 1, bias=False)
        self.block_fusion = nn.ModuleList([
            nn.Conv1d(self.B + self.eeg_hidden, self.B, 1, bias=False)
            for _ in range(self.R)
        ])
        for encoder in self.eeg_encoders:
            encoder.eval()
            for parameter in encoder.parameters():
                parameter.requires_grad = False

    def eeg_features(self, eeg: torch.Tensor) -> torch.Tensor:
        """Return frozen concatenated features with shape [B, T, branches*hidden]."""
        return torch.cat([self.num_branches * weight * encoder(eeg) for weight, encoder in zip(self.branch_weights, self.eeg_encoders)], dim=-1)

    def _estimate_mask(self, mixture_w, eeg):
        batch_size, _, mixture_frames = mixture_w.size()
        y = self.layer_norm(mixture_w)
        y = self.bottleneck_conv1x1(y)

        eeg_feat = self.eeg_features(eeg).transpose(1, 2)
        eeg_feat = torch.nn.functional.interpolate(eeg_feat, size=mixture_frames, mode="linear", align_corners=False)

        y = self.front_fusion(torch.cat((y, eeg_feat), dim=1))
        y, gap = self._segmentation(y, self.K)
        eeg_seg, _ = self._segmentation(eeg_feat, self.K)
        eeg_flat = eeg_seg.flatten(2)
        for rnn, fusion in zip(self.dual_rnn, self.block_fusion):
            y = rnn(y)
            chunk, segment = y.size(2), y.size(3)
            y = fusion(torch.cat((y.flatten(2), eeg_flat), dim=1)).reshape(batch_size, self.B, chunk, segment)
        y = self._over_add(y, gap)
        y = self.prelu(y)
        return torch.relu(self.mask_conv1x1(y)).unsqueeze(1)


class AlignedGatedEEGExtractor(MultiEEGExtractor):
    """48-D time-varying gate after branch-specific alignment and LayerNorm.

    Frozen A/B encoders stay unchanged.  Unlike 96-D concatenation, this
    controls Stage-2 condition width at 48-D; the only new capacity is a tiny
    branch-alignment projection and shared per-time-step gate.
    """
    def __init__(self, cfg, eeg_encoders, branch_weights=None):
        super().__init__(cfg, eeg_encoders, branch_weights)
        if self.num_branches != 2:
            raise ValueError("AlignedGatedEEGExtractor is defined for A/B only.")
        hidden = next(iter({encoder.hidden for encoder in eeg_encoders}))
        self.eeg_hidden = hidden
        self.align = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(2)])
        self.align_norm = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(2)])
        self.gate = nn.Sequential(nn.Linear(3 * hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
        self.front_fusion = nn.Conv1d(self.B + hidden, self.B, 1, bias=False)
        self.block_fusion = nn.ModuleList([nn.Conv1d(self.B + hidden, self.B, 1, bias=False) for _ in range(self.R)])

    def eeg_features(self, eeg):
        a, b = [norm(proj(encoder(eeg))) for encoder, proj, norm in zip(self.eeg_encoders, self.align, self.align_norm)]
        gate = torch.sigmoid(self.gate(torch.cat((a, b, (a - b).abs()), dim=-1)))
        return gate * a + (1.0 - gate) * b


@torch.no_grad()
def ensemble_margin(extractor: MultiEEGExtractor, audio_encoders: list[AudioEncoder], frontend, eeg, attended, unattended):
    """Fixed-weight mean of branch-specific attended-minus-unattended margins."""
    if len(extractor.eeg_encoders) != len(audio_encoders):
        raise ValueError("Each EEG branch must have its matching Stage-1 audio encoder.")
    margins = []
    for eeg_encoder, audio_encoder in zip(extractor.eeg_encoders, audio_encoders):
        eeg_z = eeg_encoder(eeg)
        attended_z = _interp(eeg_z, audio_encoder(frontend(attended)))
        unattended_z = _interp(eeg_z, audio_encoder(frontend(unattended)))
        margins.append((eeg_z * attended_z).sum(-1).mean(-1) - (eeg_z * unattended_z).sum(-1).mean(-1))
    return (extractor.branch_weights[:, None] * torch.stack(margins, dim=0)).sum(dim=0)



def initialize_residual_from_single_stage2(extractor: MultiEEGExtractor, path: str | Path):
    """Initialize a 96-D ensemble from a 48-D attended-branch Stage-2 model.

    All compatible separation-tail tensors are copied.  The original mixture
    and attended-EEG fusion channels are retained verbatim; new in-batch EEG
    channels are zeroed, so the first forward is exactly the single-branch
    condition before training learns whether the extra branch helps.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    source = checkpoint["extractor"]
    target = extractor.state_dict()
    state = {key: value for key, value in source.items()
             if key in target and target[key].shape == value.shape}
    for prefix in ("front_fusion.weight", *(f"block_fusion.{idx}.weight" for idx in range(extractor.R))):
        source_weight = source.get(prefix)
        target_weight = target.get(prefix)
        if source_weight is None or target_weight is None:
            raise ValueError(f"Missing required fusion weight {prefix} in residual initializer.")
        if source_weight.shape[0] != target_weight.shape[0] or source_weight.shape[1] >= target_weight.shape[1]:
            raise ValueError(f"Incompatible residual fusion shape for {prefix}: {source_weight.shape} -> {target_weight.shape}")
        expanded = torch.zeros_like(target_weight)
        expanded[:, :source_weight.shape[1]] = source_weight
        state[prefix] = expanded
    extractor.load_state_dict(state, strict=False)
    return {"source": str(Path(path).resolve()), "copied_tensors": len(state), "new_branch_channels_zero_initialized": True}


class EnsembleStage2Trainer(Stage2Trainer):
    """Stage-2 trainer with unchanged loss and equal-weight ensemble confidence."""

    def __init__(self, cfg, extractor, audio_encoders, train_loader, val_loader, test_loader, log_dir):
        super().__init__(cfg, extractor, audio_encoders[0], train_loader, val_loader, test_loader, log_dir)
        self.audio_encoders = list(audio_encoders)
        for encoder in self.audio_encoders:
            encoder.eval()
            for parameter in encoder.parameters():
                parameter.requires_grad = False

    def load_init(self, path: str | None, resume: bool = False):
        """Load Stage-2 tails while preserving the current fold's frozen branches.

        General Stage-2 initialization contains its own general Stage-1 encoder
        copies.  They must never replace the subject/fold-matched encoders that
        were supplied to this run.
        """
        if not path:
            return
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = {key: value for key, value in checkpoint["extractor"].items()
                 if not key.startswith("eeg_encoders.")}
        self.extractor.load_state_dict(state, strict=False)
        if resume:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.start_epoch = checkpoint["epoch"] + 1
            self.best_val_loss = checkpoint["best_val_loss"]
            self.no_improve = checkpoint["no_improve"]
            print(f"Resume training from epoch: {self.start_epoch}")

    def _run(self, loader, train: bool):
        self.extractor.train(train)
        for encoder in self.extractor.eeg_encoders:
            encoder.eval()
        for encoder in self.audio_encoders:
            encoder.eval()
        total_loss = correct = total = 0
        for batch in loader:
            mix, att, ign, eeg = (batch[key].to(self.cfg.device, non_blocking=True) for key in ("mixture", "attended", "unattended", "eeg"))
            estimate = self.extractor(mix, eeg)
            margin = ensemble_margin(self.extractor, self.audio_encoders, self.frontend, eeg, att, ign)
            confidence = torch.tanh(self.cfg.confidence_scale * margin.abs())
            att_sisdr, ign_sisdr = sisdr(att, estimate), sisdr(ign, estimate)
            selected = torch.where(margin >= 0, att_sisdr, ign_sisdr)
            loss = -(confidence * selected).mean()
            wins = att_sisdr.detach() > ign_sisdr.detach()
            if train:
                self.optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in self.extractor.parameters() if p.requires_grad], self.cfg.clip_grad_norm)
                self.optimizer.step()
            total_loss += loss.item() * mix.size(0); correct += wins.sum().item(); total += mix.size(0)
        return total_loss / max(total, 1), 100 * correct / max(total, 1)


@torch.no_grad()
def ensemble_selection_accuracy(extractor, audio_encoders, frontend, loader, device):
    extractor.eval()
    for encoder in extractor.eeg_encoders:
        encoder.eval()
    correct = total = 0
    for batch in loader:
        eeg, att, ign = (batch[key].to(device) for key in ("eeg", "clean", "unattended"))
        margin = ensemble_margin(extractor, audio_encoders, frontend, eeg, att, ign)
        correct += (margin > 0).sum().item(); total += eeg.size(0)
    return 100.0 * correct / max(total, 1)


@torch.no_grad()
def branch_selection_accuracy(eeg_encoder, audio_encoder, frontend, loader, device):
    """Stage-1 attended-vs-unattended accuracy for one frozen branch."""
    eeg_encoder.eval(); audio_encoder.eval(); correct = total = 0
    for batch in loader:
        eeg, att, ign = (batch[key].to(device) for key in ("eeg", "clean", "unattended"))
        eeg_z = eeg_encoder(eeg)
        att_z = _interp(eeg_z, audio_encoder(frontend(att)))
        ign_z = _interp(eeg_z, audio_encoder(frontend(ign)))
        margin = (eeg_z * att_z).sum(-1).mean(-1) - (eeg_z * ign_z).sum(-1).mean(-1)
        correct += (margin > 0).sum().item(); total += eeg.size(0)
    return 100.0 * correct / max(total, 1)
