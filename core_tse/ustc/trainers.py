"""Two-stage trainers with TRUST-TSE-compatible checkpoints and console logs."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.tensorboard import SummaryWriter

from loss import loss_nce, loss_nce_in_batch, loss_nce_synchronous_unattended
from metrics import pcc, safe_pesq, safe_stoi, sisdr


class MelFrontend(torch.nn.Module):
    def __init__(self, sample_rate: int = 16000):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=256, win_length=256, hop_length=64,
            n_mels=64, power=2.0,
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return torch.log(self.mel(audio).clamp_min(1e-8)).unsqueeze(1)


def _interp(eeg_emb, audio_emb):
    if audio_emb.size(1) != eeg_emb.size(1):
        audio_emb = F.interpolate(audio_emb.transpose(1, 2), size=eeg_emb.size(1), mode="linear", align_corners=False).transpose(1, 2)
        audio_emb = F.normalize(audio_emb, p=2, dim=-1)
    return audio_emb


class Stage1Trainer:
    def __init__(self, cfg, eeg_encoder, audio_encoder, train_loader, val_loader, test_loader, log_dir: Path):
        self.cfg, self.eeg_encoder, self.audio_encoder = cfg, eeg_encoder, audio_encoder
        self.train_loader, self.val_loader, self.test_loader = train_loader, val_loader, test_loader
        self.log_dir = log_dir / "Stage1"; self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir / "tensorboard")
        self.frontend = MelFrontend(cfg.sample_rate).to(cfg.device)
        self.optimizer = torch.optim.AdamW(list(eeg_encoder.parameters()) + list(audio_encoder.parameters()), lr=cfg.stage1_lr)
        self.start_epoch, self.best_val_acc, self.no_improve = 1, float("-inf"), 0

    def _sampling_mode(self, epoch: int, batch_index: int) -> str:
        """Resolve the loss/sampler mode for one Stage-1 training batch."""
        mode = self.cfg.stage1_negative_mode
        if mode == "b_then_a":
            switch_epoch = int(getattr(self.cfg, "stage1_switch_epoch", self.cfg.stage1_epochs // 2 + 1))
            return "in_batch" if epoch < switch_epoch else "attended_same_trial"
        if mode == "alternate_batches":
            first = getattr(self.cfg, "stage1_alternating_first", "in_batch")
            return first if batch_index % 2 == 0 else ("attended_same_trial" if first == "in_batch" else "in_batch")
        return mode

    @staticmethod
    def _set_sampler_epoch_mode(loader, mode: str):
        sampler = getattr(loader, "batch_sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch_mode"):
            sampler.set_epoch_mode(mode)

    def load_init(self, path: str | None, resume: bool = False):
        if not path:
            return
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.eeg_encoder.load_state_dict(ckpt["eeg_encoder"])
        self.audio_encoder.load_state_dict(ckpt["audio_encoder"])
        if resume:
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.start_epoch = ckpt["epoch"] + 1; self.best_val_acc = ckpt["best_val_acc"]; self.no_improve = ckpt["no_improve"]
            print(f"Resume training from epoch: {self.start_epoch}")

    def _save(self, path: Path, epoch: int):
        torch.save({"eeg_encoder": self.eeg_encoder.state_dict(), "audio_encoder": self.audio_encoder.state_dict(),
                    "optimizer": self.optimizer.state_dict(), "epoch": epoch, "best_val_acc": self.best_val_acc,
                    "no_improve": self.no_improve}, path)

    def _run(self, loader, train: bool, epoch: int = 0):
        self.eeg_encoder.train(train); self.audio_encoder.train(train)
        total_loss = total_correct = total = 0
        active_mode = self.cfg.stage1_negative_mode
        if train and active_mode == "b_then_a":
            active_mode = self._sampling_mode(epoch, 0)
            self._set_sampler_epoch_mode(loader, active_mode)
        for batch_index, batch in enumerate(loader):
            eeg, clean, ign = (batch[k].to(self.cfg.device, non_blocking=True) for k in ("eeg", "clean", "unattended"))
            eeg_z = self.eeg_encoder(eeg); audio_z = _interp(eeg_z, self.audio_encoder(self.frontend(clean)))
            if train:
                active_mode = self._sampling_mode(epoch, batch_index)
                if active_mode == "attended_same_trial":
                    loss = loss_nce(eeg_z, audio_z, trials_per_batch=self.cfg.trials_per_batch, windows_per_trial=self.cfg.windows_per_trial)
                elif active_mode == "in_batch":
                    loss = loss_nce_in_batch(eeg_z, audio_z)
                elif active_mode == "dual_loss_same_batch":
                    # Both objectives see exactly the same embeddings and the
                    # same packed batch.  A is constrained to each 8-window
                    # trial pack; B uses every other current batch item.
                    loss_a = loss_nce(eeg_z, audio_z, trials_per_batch=self.cfg.trials_per_batch, windows_per_trial=self.cfg.windows_per_trial)
                    loss_b = loss_nce_in_batch(eeg_z, audio_z)
                    dual_weight = float(getattr(self.cfg, "stage1_dual_loss_weight", 0.5))
                    if not 0.0 <= dual_weight <= 1.0:
                        raise ValueError(f"stage1_dual_loss_weight must be in [0, 1], got {dual_weight}")
                    loss = dual_weight * loss_a + (1.0 - dual_weight) * loss_b
                elif active_mode == "synchronous_unattended":
                    ign_z = _interp(eeg_z, self.audio_encoder(self.frontend(ign)))
                    loss = loss_nce_synchronous_unattended(eeg_z, audio_z, ign_z)
                else:
                    raise ValueError(f"Unknown Stage-1 negative mode: {active_mode}")
            else:
                loss = None
            if train:
                self.optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.eeg_encoder.parameters()) + list(self.audio_encoder.parameters()), self.cfg.clip_grad_norm)
                self.optimizer.step()
            with torch.no_grad():
                ign_z = _interp(eeg_z, self.audio_encoder(self.frontend(ign)))
                total_correct += ((eeg_z * audio_z).sum(-1).mean(-1) > (eeg_z * ign_z).sum(-1).mean(-1)).sum().item(); total += eeg.size(0)
                if train: total_loss += loss.item() * eeg.size(0)
        return (total_loss / max(total, 1) if train else 0.0), 100.0 * total_correct / max(total, 1)

    @torch.no_grad()
    def _selection_accuracy(self, loader):
        self.eeg_encoder.eval(); self.audio_encoder.eval(); correct = total = 0
        for batch in loader:
            eeg, att, ign = (batch[k].to(self.cfg.device, non_blocking=True) for k in ("eeg", "clean", "unattended"))
            z = self.eeg_encoder(eeg); att_z = _interp(z, self.audio_encoder(self.frontend(att))); ign_z = _interp(z, self.audio_encoder(self.frontend(ign)))
            correct += ((z * att_z).sum(-1).mean(-1) > (z * ign_z).sum(-1).mean(-1)).sum().item(); total += eeg.size(0)
        return 100.0 * correct / max(total, 1)

    def train(self):
        self._save(self.log_dir / "last_checkpoint.pt", self.start_epoch - 1)
        for epoch in range(self.start_epoch, self.cfg.stage1_epochs + 1):
            if self.cfg.stage1_negative_mode == "b_then_a" and epoch == int(self.cfg.stage1_switch_epoch):
                # B is a fixed representation pretraining phase.  Reset model
                # selection at the objective switch so the checkpoint used by
                # Stage-2 is necessarily an A-finetuned encoder, rather than a
                # high-validation checkpoint retained from the B-only phase.
                self.best_val_acc, self.no_improve = float("-inf"), 0
                print(f"Stage-1 sampling switch at epoch {epoch}: in_batch -> attended_same_trial; reset A-phase model selection")
            began = time.time(); tr_loss, tr_acc = self._run(self.train_loader, True, epoch)
            val_acc = self._selection_accuracy(self.val_loader); elapsed = time.time() - began
            if self.cfg.stage1_negative_mode == "b_then_a":
                schedule_label = self._sampling_mode(epoch, 0)
            elif self.cfg.stage1_negative_mode == "alternate_batches":
                schedule_label = f"{getattr(self.cfg, 'stage1_alternating_first', 'in_batch')}↔batch"
            elif self.cfg.stage1_negative_mode == "dual_loss_same_batch":
                schedule_label = f"same-batch A+B (lambda={float(getattr(self.cfg, 'stage1_dual_loss_weight', 0.5)):.2f})"
            else:
                schedule_label = self.cfg.stage1_negative_mode
            print(f"Epoch {epoch} | Train Loss {tr_loss:.3f} | Train Acc {tr_acc:.2f}")
            print(f"Epoch {epoch} | Val Acc {val_acc:.2f} | Time {elapsed:.2f}s | Sampling {schedule_label}")
            if val_acc > self.best_val_acc:
                self.best_val_acc, self.no_improve = val_acc, 0; self._save(self.log_dir / "last_best_checkpoint.pt", epoch); print("Found new best model, dict saved")
            else: self.no_improve += 1
            self.writer.add_scalar("Train_loss", tr_loss, epoch); self.writer.add_scalar("Train_acc", tr_acc, epoch); self.writer.add_scalar("Validation_acc", val_acc, epoch)
            self._save(self.log_dir / "last_checkpoint.pt", epoch)
            allow_early_stop = not (self.cfg.stage1_negative_mode == "b_then_a" and epoch < int(self.cfg.stage1_switch_epoch))
            if allow_early_stop and self.no_improve >= self.cfg.patience: print(f"No improvement for {self.cfg.patience} epochs, early stopping."); break
        self.writer.close()

    def evaluate(self):
        self.load_init(str(self.log_dir / "last_best_checkpoint.pt"))
        value = self._selection_accuracy(self.test_loader)
        metrics = {"Accuracy": value}
        print(f"Accuracy: {value:.2f}")
        (self.log_dir / "evaluation.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


class Stage2Trainer:
    def __init__(self, cfg, extractor, audio_encoder, train_loader, val_loader, test_loader, log_dir: Path):
        self.cfg, self.extractor, self.audio_encoder = cfg, extractor, audio_encoder
        self.train_loader, self.val_loader, self.test_loader = train_loader, val_loader, test_loader
        self.log_dir = log_dir / "Stage2"; self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir / "tensorboard"); self.frontend = MelFrontend(cfg.sample_rate).to(cfg.device)
        self.optimizer = torch.optim.AdamW([p for p in extractor.parameters() if p.requires_grad], lr=cfg.stage2_lr)
        self.start_epoch, self.best_val_loss, self.no_improve = 1, float("inf"), 0

    def load_init(self, path: str | None, resume: bool = False):
        if not path: return
        ckpt = torch.load(path, map_location="cpu", weights_only=False); self.extractor.load_state_dict(ckpt["extractor"], strict=False)
        if resume:
            self.optimizer.load_state_dict(ckpt["optimizer"]); self.start_epoch = ckpt["epoch"] + 1; self.best_val_loss = ckpt["best_val_loss"]; self.no_improve = ckpt["no_improve"]
            print(f"Resume training from epoch: {self.start_epoch}")

    def _save(self, path: Path, epoch: int):
        torch.save({"extractor": self.extractor.state_dict(), "optimizer": self.optimizer.state_dict(), "epoch": epoch,
                    "best_val_loss": self.best_val_loss, "no_improve": self.no_improve}, path)

    def _run(self, loader, train: bool):
        self.extractor.train(train); self.extractor.eeg_encoder.eval(); self.audio_encoder.eval()
        total_loss = correct = total = 0
        for batch in loader:
            mix, att, ign, eeg = (batch[k].to(self.cfg.device, non_blocking=True) for k in ("mixture", "attended", "unattended", "eeg"))
            estimate = self.extractor(mix, eeg)
            with torch.no_grad():
                z = self.extractor.eeg_encoder(eeg); az = _interp(z, self.audio_encoder(self.frontend(att))); iz = _interp(z, self.audio_encoder(self.frontend(ign)))
                margin = (z * az).sum(-1).mean(-1) - (z * iz).sum(-1).mean(-1)
                confidence = torch.tanh(self.cfg.confidence_scale * margin.abs())
            att_sisdr, ign_sisdr = sisdr(att, estimate), sisdr(ign, estimate)
            if getattr(self.cfg, "stage2_loss", "confidence_weighted") == "plain_sisdr":
                # Ablation/version switch: train only against the attended source.
                # The unattended source remains available for unchanged validation
                # and reporting, but contributes no gradient to this loss.
                loss = -att_sisdr.mean()
            else:
                selected = torch.where(margin >= 0, att_sisdr, ign_sisdr)
                loss = -(confidence * selected).mean()
            wins = att_sisdr.detach() > ign_sisdr.detach()
            if train:
                self.optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_([p for p in self.extractor.parameters() if p.requires_grad], self.cfg.clip_grad_norm); self.optimizer.step()
            total_loss += loss.item() * mix.size(0); correct += wins.sum().item(); total += mix.size(0)
        return total_loss / max(total, 1), 100 * correct / max(total, 1)

    def train(self):
        self._save(self.log_dir / "last_checkpoint.pt", self.start_epoch - 1)
        for epoch in range(self.start_epoch, self.cfg.stage2_epochs + 1):
            began = time.time(); tr_loss, tr_acc = self._run(self.train_loader, True)
            with torch.no_grad(): val_loss, val_acc = self._run(self.val_loader, False)
            print(f"Epoch {epoch} | Train Loss {tr_loss:.3f} | Train Acc {tr_acc:.2f}")
            print(f"Epoch {epoch} | Val Loss {val_loss:.3f} | Val Acc {val_acc:.2f} | Time {time.time()-began:.2f}s")
            if val_loss < self.best_val_loss:
                self.best_val_loss, self.no_improve = val_loss, 0; self._save(self.log_dir / "last_best_checkpoint.pt", epoch); print("Found new best model, dict saved")
            else: self.no_improve += 1
            self.writer.add_scalar("Train_loss", tr_loss, epoch); self.writer.add_scalar("Validation_loss", val_loss, epoch); self.writer.add_scalar("Train_acc", tr_acc, epoch); self.writer.add_scalar("Validation_acc", val_acc, epoch)
            self._save(self.log_dir / "last_checkpoint.pt", epoch)
            if self.no_improve >= self.cfg.patience: print(f"No improvement for {self.cfg.patience} epochs, early stopping."); break
        self.writer.close()

    @torch.no_grad()
    def evaluate(self):
        self.load_init(str(self.log_dir / "last_best_checkpoint.pt")); self.extractor.eval()
        values = {key: [] for key in ("sisdr", "sisdr_ign", "sisdr_mix", "pesq", "pesq_correct", "pesq_ign_wrong", "stoi", "stoi_correct", "stoi_ign_wrong", "pcc_target", "pcc_unattended", "pcc_mixture")}
        correct = total = 0
        for batch in self.test_loader:
            mix, att, ign, eeg = (batch[k].to(self.cfg.device, non_blocking=True) for k in ("mixture", "attended", "unattended", "eeg"))
            est = self.extractor(mix, eeg); a = sisdr(att, est); i = sisdr(ign, est); baseline = sisdr(att, mix); wins = a > i
            correct += wins.sum().item(); total += mix.size(0); values["sisdr"].extend(a.cpu().tolist()); values["sisdr_ign"].extend(i.cpu().tolist()); values["sisdr_mix"].extend(baseline.cpu().tolist())
            for n in range(mix.size(0)):
                e, target, ignored, mixture = (x[n].detach().cpu().numpy() for x in (est, att, ign, mix))
                pv, sv = safe_pesq(self.cfg.sample_rate, target, e), safe_stoi(self.cfg.sample_rate, target, e)
                values["pesq"].append(pv); values["stoi"].append(sv)
                if bool(wins[n]): values["pesq_correct"].append(pv); values["stoi_correct"].append(sv)
                else:
                    values["pesq_ign_wrong"].append(safe_pesq(self.cfg.sample_rate, ignored, e)); values["stoi_ign_wrong"].append(safe_stoi(self.cfg.sample_rate, ignored, e))
                values["pcc_target"].append(pcc(e, target)); values["pcc_unattended"].append(pcc(e, ignored)); values["pcc_mixture"].append(pcc(e, mixture))
        mean = lambda xs: float(np.nanmean([x for x in xs if x is not None])) if any(x is not None for x in xs) else float("nan")
        wins = np.asarray(values["sisdr"]) > np.asarray(values["sisdr_ign"]); acc = 100 * correct / max(total, 1)
        metrics = {"SISDR_att-All": mean(values["sisdr"]), "SISDR_att-Correct": mean(np.asarray(values["sisdr"])[wins].tolist()), "SISDR_ign-Wrong": mean(np.asarray(values["sisdr_ign"])[~wins].tolist()),
                   "PESQ_att-all": mean(values["pesq"]), "PESQ_att-Correct": mean(values["pesq_correct"]), "PESQ-ign-Wrong": mean(values["pesq_ign_wrong"]),
                   "STOI_att-all": mean(values["stoi"]), "STOI_att-Correct": mean(values["stoi_correct"]), "STOI-ign-Wrong": mean(values["stoi_ign_wrong"]), "Accuracy": acc,
                   "SISDRi": mean((np.asarray(values["sisdr"]) - np.asarray(values["sisdr_mix"])).tolist()), "target_vs_unattended_margin": mean((np.asarray(values["sisdr"]) - np.asarray(values["sisdr_ign"])).tolist()),
                   "PCC_target": mean(values["pcc_target"]), "PCC_unattended": mean(values["pcc_unattended"]), "PCC_mixture": mean(values["pcc_mixture"])}
        for name, value in metrics.items(): print(f"{name}: {value:.2f}")
        (self.log_dir / "evaluation.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
