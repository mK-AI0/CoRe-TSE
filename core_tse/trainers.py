# This file is adapted from ClearerVoice-Studio
# (https://github.com/modelscope/ClearerVoice-Studio),
# licensed under the Apache License 2.0.
# Original work Copyright (c) Alibaba Group.
# Modifications Copyright (c) 2026 <your name / institution>.


import json
import time
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from loss import loss_nce, loss_nce_in_batch
from metrics import cal_SISDR, cal_pesq, cal_stoi


class Stage1Trainer(object):
    def __init__(self, args, eeg_encoder, audio_encoder, train_loader, val_loader, test_loader):
        self.args = args
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.log_dir = args.log_dir / 'Stage1'
        self.writer = SummaryWriter(self.log_dir / 'tensorboard')

        self.eeg_encoder = eeg_encoder
        self.audio_encoder = audio_encoder
        self.optimizer = torch.optim.AdamW(
            list(eeg_encoder.parameters()) + list(audio_encoder.parameters()),
            lr=args.stage1.init_lr,
        )
        self.start_epoch = 1
        self.epoch = 0
        self.best_val_acc = float('-inf')
        self.val_no_impv = 0

        if self.args.train_from_last_checkpoint:
            self._load_model(
                self.log_dir / 'last_checkpoint.pt',
                load_training_states=True,
            )

        self._save_model(self.log_dir / 'last_checkpoint.pt')

    def train(self):
        for epoch in range(self.start_epoch, self.args.stage1.max_epoch + 1):
            epoch_start = time.time()
            self.epoch = epoch
            self._set_train()
            tr_loss, tr_acc = self._run_one_epoch(self.train_loader, train=True)
            print(f'Epoch {epoch} | Train Loss {tr_loss:.3f} | Train Acc {tr_acc:.2f}')

            self._set_eval()

            with torch.no_grad():
                val_acc = self._run_selection_eval(self.val_loader)

            epoch_time = time.time() - epoch_start
            print(f'Epoch {epoch} | Val Acc {val_acc:.2f} | Time {epoch_time:.2f}s')

            if val_acc <= self.best_val_acc:
                self.val_no_impv += 1

                if self.val_no_impv >= 10:
                    print('No improvement for 10 epochs, early stopping.')
                    break
            else:
                self.val_no_impv = 0
                self.best_val_acc = val_acc
                self._save_model(self.log_dir / 'last_best_checkpoint.pt')
                print('Found new best model, dict saved')

            # Tensorboard logging
            self.writer.add_scalar('Train_loss', tr_loss, self.epoch)
            self.writer.add_scalar('Train_acc', tr_acc, self.epoch)
            self.writer.add_scalar('Validation_acc', val_acc, self.epoch)
            self._save_model(self.log_dir / 'last_checkpoint.pt')

        self.writer.close()

    def _run_selection_eval(self, data_loader):
        total_correct = 0
        total_samples = 0

        self._set_eval()

        for batch in tqdm(data_loader, desc='Batch', leave=False, disable=True):
            eeg, mel_att, mel_ign = batch

            eeg = eeg.to(self.args.device)
            mel_att = mel_att.to(self.args.device)
            mel_ign = mel_ign.to(self.args.device)

            eeg_emb = self.eeg_encoder(eeg)

            audio_att_emb = self.audio_encoder(mel_att)
            audio_att_emb = self._temporal_interpolate(eeg_emb, audio_att_emb)

            audio_ign_emb = self.audio_encoder(mel_ign)
            audio_ign_emb = self._temporal_interpolate(eeg_emb, audio_ign_emb)

            sim_att = (eeg_emb * audio_att_emb).sum(dim=-1).mean(dim=-1)
            sim_ign = (eeg_emb * audio_ign_emb).sum(dim=-1).mean(dim=-1)

            correct = (sim_att > sim_ign).float()
            total_correct += correct.sum().item()
            total_samples += eeg.shape[0]

        return total_correct / total_samples * 100

    def _set_train(self):
        self.eeg_encoder.train()
        self.audio_encoder.train()

    def _set_eval(self):
        self.eeg_encoder.eval()
        self.audio_encoder.eval()

    def _save_model(self, path):
        ckpt = {
            'eeg_encoder': self.eeg_encoder.state_dict(),
            'audio_encoder': self.audio_encoder.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'best_val_acc': self.best_val_acc,
            'val_no_impv': self.val_no_impv,
        }
        torch.save(ckpt, path)

    def _load_model(self, path, load_training_states=False):
        ckpt = torch.load(path, map_location='cpu')
        self.eeg_encoder.load_state_dict(ckpt['eeg_encoder'], strict=True)
        self.audio_encoder.load_state_dict(ckpt['audio_encoder'], strict=True)

        if load_training_states:
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.val_no_impv = ckpt['val_no_impv']
            self.start_epoch = ckpt['epoch'] + 1
            self.epoch = self.start_epoch - 1
            print(f'Resume training from epoch: {self.start_epoch}')

    def _run_one_epoch(self, data_loader, train: bool = True):
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for batch in tqdm(data_loader, desc='Batch', leave=False, disable=True):
            eeg, mel_att, mel_ign = batch

            eeg = eeg.to(self.args.device)
            mel_att = mel_att.to(self.args.device)
            mel_ign = mel_ign.to(self.args.device)

            eeg_emb = self.eeg_encoder(eeg)
            audio_att_emb = self.audio_encoder(mel_att)
            audio_att_emb = self._temporal_interpolate(eeg_emb, audio_att_emb)

            with torch.no_grad():
                audio_ign_emb = self.audio_encoder(mel_ign)
                audio_ign_emb = self._temporal_interpolate(eeg_emb, audio_ign_emb)

            if self.args.stage1.negative_mode == 'attended_same_trial':
                loss = loss_nce(
                    eeg_emb, audio_att_emb,
                    trials_per_batch=self.args.stage1.trials_per_batch,
                    windows_per_trial=self.args.stage1.windows_per_trial,
                )
            elif self.args.stage1.negative_mode == 'in_batch':
                loss = loss_nce_in_batch(eeg_emb, audio_att_emb)
            else:
                raise ValueError(
                    f'Unknown Stage-1 negative mode: {self.args.stage1.negative_mode}. '
                    'Expected attended_same_trial or in_batch.'
                )

            with torch.no_grad():
                sim_att = (eeg_emb * audio_att_emb).sum(dim=-1).mean(dim=-1)
                sim_ign = (eeg_emb * audio_ign_emb).sum(dim=-1).mean(dim=-1)

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.eeg_encoder.parameters()) + list(self.audio_encoder.parameters()),
                    self.args.clip_grad_norm,
                )
                self.optimizer.step()

            correct = (sim_att > sim_ign).float()
            total_correct += correct.sum().item()
            batch_size = eeg.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        epoch_loss = total_loss / total_samples
        epoch_acc = total_correct / total_samples * 100
        return epoch_loss, epoch_acc

    def _temporal_interpolate(self, eeg_emb, audio_emb):
        if audio_emb.size(1) != eeg_emb.size(1):
            audio_emb = audio_emb.transpose(1, 2)
            audio_emb = F.interpolate(
                audio_emb,
                size=eeg_emb.size(1),
                mode='linear',
                align_corners=False,
            )
            audio_emb = audio_emb.transpose(1, 2)
            audio_emb = F.normalize(audio_emb, p=2, dim=-1)

        return audio_emb

    def evaluate(self):
        total_correct = 0
        total_samples = 0

        self._load_model(self.log_dir / 'last_best_checkpoint.pt')
        self._set_eval()

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='Batch', leave=False, disable=True):
                eeg, mel_att, mel_ign = batch

                eeg = eeg.to(self.args.device)
                mel_att = mel_att.to(self.args.device)
                mel_ign = mel_ign.to(self.args.device)

                eeg_emb = self.eeg_encoder(eeg)
                audio_att_emb = self.audio_encoder(mel_att)
                audio_att_emb = self._temporal_interpolate(eeg_emb, audio_att_emb)
                audio_ign_emb = self.audio_encoder(mel_ign)
                audio_ign_emb = self._temporal_interpolate(eeg_emb, audio_ign_emb)

                sim_att = (eeg_emb * audio_att_emb).sum(dim=-1).mean(dim=-1)
                sim_ign = (eeg_emb * audio_ign_emb).sum(dim=-1).mean(dim=-1)

                correct = (sim_att > sim_ign).float()
                total_correct += correct.sum().item()
                batch_size = eeg.shape[0]
                total_samples += batch_size

        metrics = {
            'Accuracy': total_correct / total_samples * 100,
        }

        for name, value in metrics.items():
            print(f'{name}: {value:.2f}')

        result_path = self.log_dir / 'evaluation.json'

        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)


class Stage2Trainer(object):
    def __init__(self, args, extractor, audio_encoder, train_loader, val_loader, test_loader):
        self.args = args
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.log_dir = args.log_dir / 'Stage2'
        self.writer = SummaryWriter(self.log_dir / 'tensorboard')

        self.extractor = extractor
        self.audio_encoder = audio_encoder
        self.audio_encoder.eval()

        self.optimizer = torch.optim.AdamW(
            [p for p in self.extractor.parameters() if p.requires_grad],
            lr=args.stage2.init_lr,
        )
        self.start_epoch = 1
        self.step_num = 1
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.val_no_impv = 0

        if self.args.train_from_last_checkpoint:
            self._load_model(
                self.log_dir / 'last_checkpoint.pt',
                load_training_states=True,
            )

        self._save_model(self.log_dir / 'last_checkpoint.pt')

    def train(self):
        for epoch in range(self.start_epoch, self.args.stage2.max_epoch + 1):
            epoch_start = time.time()
            self.epoch = epoch
            self._set_train()
            tr_loss, tr_acc = self._run_one_epoch(self.train_loader, train=True)
            print(f'Epoch {epoch} | Train Loss {tr_loss:.3f} | Train Acc {tr_acc:.2f}')

            self._set_eval()

            with torch.no_grad():
                val_loss, val_acc = self._run_one_epoch(self.val_loader, train=False)

            epoch_time = time.time() - epoch_start
            print(f'Epoch {epoch} | Val Loss {val_loss:.3f} | Val Acc {val_acc:.2f} | Time {epoch_time:.2f}s')

            if val_loss >= self.best_val_loss:
                self.val_no_impv += 1

                if self.val_no_impv >= 10:
                    print('No improvement for 10 epochs, early stopping.')
                    break
            else:
                self.val_no_impv = 0
                self.best_val_loss = val_loss
                self._save_model(self.log_dir / 'last_best_checkpoint.pt')
                print('Found new best model, dict saved')

            # Tensorboard logging
            self.writer.add_scalar('Train_loss', tr_loss, self.epoch)
            self.writer.add_scalar('Validation_loss', val_loss, self.epoch)
            self.writer.add_scalar('Train_acc', tr_acc, self.epoch)
            self.writer.add_scalar('Validation_acc', val_acc, self.epoch)
            self._save_model(self.log_dir / 'last_checkpoint.pt')

        self.writer.close()

    def _set_train(self):
        self.extractor.train()
        self.extractor.eeg_encoder.eval()

    def _set_eval(self):
        self.extractor.eval()

    def _save_model(self, path):
        ckpt = {
            'extractor': self.extractor.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step_num': self.step_num,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'val_no_impv': self.val_no_impv,
        }
        torch.save(ckpt, path)

    def _load_model(self, path, load_training_states=False):
        ckpt = torch.load(path, map_location='cpu')
        self.extractor.load_state_dict(ckpt['extractor'], strict=True)

        if load_training_states:
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.best_val_loss = ckpt['best_val_loss']
            self.val_no_impv = ckpt['val_no_impv']
            self.step_num = ckpt['step_num']
            self.start_epoch = ckpt['epoch'] + 1
            self.epoch = self.start_epoch - 1
            print(f'Resume training from epoch: {self.start_epoch}')

    def _run_one_epoch(self, data_loader, train: bool = True):
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for batch in tqdm(data_loader, desc='Batch', leave=False, disable=True):
            a_mix, a_att, a_ign, eeg, mel_att, mel_ign = batch
            a_mix = a_mix.to(self.args.device)
            a_att = a_att.to(self.args.device)
            a_ign = a_ign.to(self.args.device)
            eeg = eeg.to(self.args.device)
            mel_att = mel_att.to(self.args.device)
            mel_ign = mel_ign.to(self.args.device)

            a_est = self.extractor(a_mix, eeg)

            with torch.no_grad():
                eeg_emb = self.extractor.eeg_encoder(eeg)
                audio_att_emb = self.audio_encoder(mel_att)
                audio_att_emb = self._temporal_interpolate(eeg_emb, audio_att_emb)
                audio_ign_emb = self.audio_encoder(mel_ign)
                audio_ign_emb = self._temporal_interpolate(eeg_emb, audio_ign_emb)

                sim_att = (eeg_emb * audio_att_emb).sum(dim=-1).mean(dim=-1)
                sim_ign = (eeg_emb * audio_ign_emb).sum(dim=-1).mean(dim=-1)

            margin = sim_att - sim_ign
            sisdr_att = cal_SISDR(a_att, a_est)
            sisdr_ign = cal_SISDR(a_ign, a_est)
            sisdr_selected = torch.where(margin >= 0, sisdr_att, sisdr_ign)
            conf_scale = 5.0
            confidence = torch.tanh(conf_scale * margin.abs())
            loss = -(confidence * sisdr_selected).mean()

            with torch.no_grad():
                correct = (sisdr_att > sisdr_ign).float()
                total_correct += correct.sum().item()
                batch_size = eeg.shape[0]
                total_loss += loss.item() * batch_size
                total_samples += batch_size

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.extractor.parameters() if p.requires_grad],
                    self.args.clip_grad_norm,
                )
                self.optimizer.step()

        epoch_loss = total_loss / total_samples
        epoch_acc = total_correct / total_samples * 100
        return epoch_loss, epoch_acc

    def _temporal_interpolate(self, eeg_emb, audio_emb):
        if audio_emb.size(1) != eeg_emb.size(1):
            audio_emb = audio_emb.transpose(1, 2)
            audio_emb = F.interpolate(
                audio_emb,
                size=eeg_emb.size(1),
                mode='linear',
                align_corners=False,
            )
            audio_emb = audio_emb.transpose(1, 2)
            audio_emb = F.normalize(audio_emb, p=2, dim=-1)

        return audio_emb

    def evaluate(self):
        sisdr_att_all = 0
        sisdr_att_correct = 0
        sisdr_ign_wrong = 0

        pesq_att_all_sum = 0.0
        pesq_att_correct_sum = 0.0
        pesq_ign_wrong_sum = 0.0
        pesq_att_all_cnt = 0
        pesq_att_correct_cnt = 0
        pesq_ign_wrong_cnt = 0

        stoi_att_all_sum = 0.0
        stoi_att_correct_sum = 0.0
        stoi_ign_wrong_sum = 0.0
        stoi_att_all_cnt = 0
        stoi_att_correct_cnt = 0
        stoi_ign_wrong_cnt = 0

        att_wins_cnt = 0
        sample_cnt = 0

        self._load_model(self.log_dir / 'last_best_checkpoint.pt')
        self._set_eval()
        sr = int(self.args.audio_sr)
        pesq_mode = 'wb' if sr == 16000 else 'nb'

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='Batch', leave=False, disable=True):
                a_mix, a_att, a_ign, eeg, mel_att, mel_ign = batch
                a_mix = a_mix.to(self.args.device)
                a_att = a_att.to(self.args.device)
                a_ign = a_ign.to(self.args.device)
                eeg = eeg.to(self.args.device)
                mel_att = mel_att.to(self.args.device)
                mel_ign = mel_ign.to(self.args.device)

                a_est = self.extractor(a_mix, eeg)
                sisdr_att = cal_SISDR(a_att, a_est)
                sisdr_ign = cal_SISDR(a_ign, a_est)

                sisdr_att_all += sisdr_att.sum().item()
                att_wins = sisdr_att > sisdr_ign
                sisdr_att_correct += (sisdr_att * att_wins).sum().item()
                sisdr_ign_wrong += (sisdr_ign * ~att_wins).sum().item()
                att_wins_cnt += int(att_wins.sum().item())
                B = a_mix.shape[0]
                sample_cnt += B

                for b in range(B):
                    est = a_est[b].cpu().squeeze().numpy()
                    att = a_att[b].cpu().squeeze().numpy()
                    ign = a_ign[b].cpu().squeeze().numpy()

                    s = max(
                        float(np.max(np.abs(est))),
                        float(np.max(np.abs(att))),
                        float(np.max(np.abs(ign))),
                        1e-8,
                    )
                    est = np.clip(est / s, -1, 1)
                    att = np.clip(att / s, -1, 1)
                    ign = np.clip(ign / s, -1, 1)

                    # All:
                    pv = cal_pesq(sr, pesq_mode, att, est)
                    sv = cal_stoi(sr, att, est)

                    if pv is not None:
                        pesq_att_all_sum += pv
                        pesq_att_all_cnt += 1

                    if sv is not None:
                        stoi_att_all_sum += sv
                        stoi_att_all_cnt += 1

                    # Correct
                    if bool(att_wins[b].item()):
                        if pv is not None:
                            pesq_att_correct_sum += pv
                            pesq_att_correct_cnt += 1

                        if sv is not None:
                            stoi_att_correct_sum += sv
                            stoi_att_correct_cnt += 1
                    # Wrong
                    else:
                        pv = cal_pesq(sr, pesq_mode, ign, est)
                        if pv is not None:
                            pesq_ign_wrong_sum += pv
                            pesq_ign_wrong_cnt += 1

                        sv = cal_stoi(sr, ign, est)
                        if sv is not None:
                            stoi_ign_wrong_sum += sv
                            stoi_ign_wrong_cnt += 1


        metrics = {
            'SISDR_att-All': sisdr_att_all / sample_cnt,
            'SISDR_att-Correct': sisdr_att_correct / att_wins_cnt,
            'SISDR_ign-Wrong': sisdr_ign_wrong / (sample_cnt - att_wins_cnt),
            'PESQ_att-all': pesq_att_all_sum / pesq_att_all_cnt,
            'PESQ_att-Correct': pesq_att_correct_sum / pesq_att_correct_cnt,
            'PESQ-ign-Wrong': pesq_ign_wrong_sum / pesq_ign_wrong_cnt,
            'STOI_att-all': stoi_att_all_sum / stoi_att_all_cnt,
            'STOI_att-Correct': stoi_att_correct_sum / stoi_att_correct_cnt,
            'STOI-ign-Wrong': stoi_ign_wrong_sum / stoi_ign_wrong_cnt,
            'Accuracy': att_wins_cnt / sample_cnt * 100.0,
        }

        for name, value in metrics.items():
            print(f'{name}: {value:.2f}')

        result_path = self.log_dir / 'evaluation.json'

        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
