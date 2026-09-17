import math
import os
import h5py
import random
import torch
import torch.utils.data as data
from torch.utils.data import Sampler
from collections import defaultdict
import numpy as np
import soundfile as sf


import random
from torch.utils.data import Sampler


class TrialAwareBatchSampler(Sampler):
    def __init__(
        self,
        trial_keys,
        samples_by_trial,
        starts_by_trial,
        trials_per_batch,
        windows_per_trial,
        min_gap_sec,
        drop_last=True,
    ):
        self.trial_keys = trial_keys
        self.samples_by_trial = samples_by_trial
        self.starts_by_trial = starts_by_trial
        self.trials_per_batch = trials_per_batch
        self.windows_per_trial = windows_per_trial
        self.min_gap_sec = min_gap_sec
        self.drop_last = drop_last
        self._epoch_units = None

    def _make_packs_for_trial(self, trial_key):
        indices = self.samples_by_trial[trial_key]
        starts = self.starts_by_trial[trial_key]

        remaining = list(range(len(indices)))
        random.shuffle(remaining)

        packs = []

        while True:
            chosen = []

            for pos in remaining:
                t = starts[pos]
                if all(abs(t - starts[p]) >= self.min_gap_sec for p in chosen):
                    chosen.append(pos)

                    if len(chosen) == self.windows_per_trial:
                        break

            if len(chosen) < self.windows_per_trial:
                break

            used = set(chosen)
            packs.append([indices[p] for p in chosen])
            remaining = [p for p in remaining if p not in used]
            random.shuffle(remaining)

        return packs

    def _refresh_epoch_units(self):
        units = []
        for tk in self.trial_keys:
            packs = self._make_packs_for_trial(tk)
            units.extend(packs)

        random.shuffle(units)
        self._epoch_units = units

    def _ensure_epoch_units(self):
        if self._epoch_units is None:
            self._refresh_epoch_units()

    def __iter__(self):
        self._ensure_epoch_units()
        units = self._epoch_units
        n_units = len(units)

        if self.drop_last:
            n_batch = n_units // self.trials_per_batch
        else:
            n_batch = math.ceil(n_units / self.trials_per_batch)

        try:
            for b in range(n_batch):
                start = b * self.trials_per_batch
                end = (b + 1) * self.trials_per_batch
                chosen_units = units[start:end]

                if len(chosen_units) < self.trials_per_batch and self.drop_last:
                    break

                batch = []
                for pack in chosen_units:
                    batch.extend(pack)

                yield batch
        finally:
            self._epoch_units = None

    def __len__(self):
        self._ensure_epoch_units()
        n_units = len(self._epoch_units)

        if self.drop_last:
            return n_units // self.trials_per_batch
        else:
            return math.ceil(n_units / self.trials_per_batch)


def build_dataloader(args, stage, partition):
    # HDF5 handles are opened lazily in each worker, so worker processes never
    # inherit a live h5py handle from the parent.  Persistent workers are used
    # only for training: validation/test loaders are iterated once per epoch
    # and would otherwise retain idle worker pools alongside the train pool.
    worker_kwargs = {}
    if args.num_workers > 0:
        worker_kwargs['prefetch_factor'] = getattr(args, 'dataloader_prefetch_factor', 2)
        if partition == 'train':
            worker_kwargs['persistent_workers'] = True

    if stage == 1:
        dataset = Stage1Dataset(args, partition)

        if partition == 'train':
            batch_sampler = TrialAwareBatchSampler(
                trial_keys=dataset.trial_keys,
                samples_by_trial=dataset.samples_by_trial,
                starts_by_trial=dataset.starts_by_trial,
                trials_per_batch=args.stage1.trials_per_batch,
                windows_per_trial=args.stage1.windows_per_trial,
                min_gap_sec=args.stage1.min_gap_sec,
            )
            dataloader = data.DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
                **worker_kwargs,
            )
        else:
            dataloader = data.DataLoader(
                dataset,
                batch_size=args.stage1.trials_per_batch * args.stage1.windows_per_trial,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                **worker_kwargs,
            )

    elif stage == 2:
        dataset = Stage2Dataset(args, partition)
        dataloader = data.DataLoader(
            dataset,
            batch_size=args.stage2.batch_size,
            shuffle=(partition == 'train'),
            num_workers=args.num_workers,
            pin_memory=True,
            **worker_kwargs,
        )

    return dataloader


class Stage1Dataset(data.Dataset):
    def __init__(self, args, partition):
        self.args = args
        self.partition = partition
        self.audio_sr = args.audio_sr
        self.eeg_sr = args.eeg_sr

        lines = open(args.metadata_path).read().splitlines()
        lines = [line for line in lines if line.split(',')[0] == partition]

        if args.subject_ids is not None:
            subject_ids = set(int(s) for s in args.subject_ids)
            lines = [
                line for line in lines
                if int(line.split(',')[1]) in subject_ids
            ]

        self.items = []
        tmp_by_trial = defaultdict(list)

        for idx, line in enumerate(lines):
            m = line.split(',')
            subject = m[1]
            trial = int(m[2])
            att_name = m[3]
            att_start_sec = float(m[4])
            ign_name = m[6]
            ign_start_sec = float(m[7])
            length_sec = float(m[-1])
            trial_key = f'S{subject}_T{trial}'

            self.items.append((
                subject,
                trial,
                att_name,
                att_start_sec,
                ign_name,
                ign_start_sec,
                length_sec,
            ))
            tmp_by_trial[trial_key].append((att_start_sec, idx))

        self.trial_keys = list(tmp_by_trial.keys())
        self.samples_by_trial = {}
        self.starts_by_trial = {}

        for trial_key in self.trial_keys:
            pairs = sorted(tmp_by_trial[trial_key], key=lambda x: x[0])
            self.starts_by_trial[trial_key] = [s for s, _ in pairs]
            self.samples_by_trial[trial_key] = [i for _, i in pairs]

        self.eeg_path = args.eeg_dir / 'eeg.h5'
        self.mel_path = args.audio_dir / 'mel.h5'
        self.eeg = None
        self.mel = None
        self._h5_pid = None
        print(f'The {partition} dataloader has prepared ({len(self.items)} samples).')

    def _ensure_h5(self):
        """Open HDF5 files in the process that executes ``__getitem__``."""
        pid = os.getpid()
        if self._h5_pid == pid and self.eeg is not None and self.mel is not None:
            return
        self._close_h5()
        self.eeg = h5py.File(self.eeg_path, 'r')
        self.mel = h5py.File(self.mel_path, 'r')
        self._h5_pid = pid

    def _close_h5(self):
        for handle in (self.eeg, self.mel):
            if handle is not None:
                handle.close()
        self.eeg = None
        self.mel = None
        self._h5_pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        # Required for spawn-based workers and defensive for future callers:
        # a live h5py handle is neither picklable nor safe to share.
        state.update(eeg=None, mel=None, _h5_pid=None)
        return state

    def __del__(self):
        self._close_h5()

    def __getitem__(self, index):
        self._ensure_h5()
        subject, trial, att_name, att_start_sec, ign_name, ign_start_sec, length_sec = self.items[index]

        # EEG
        eeg_start = int(att_start_sec * self.eeg_sr)
        eeg_len = math.floor(length_sec * self.eeg_sr)
        eeg_end = eeg_start + eeg_len
        eeg_att = self.eeg[f'S{subject}/Tra{trial}'][eeg_start:eeg_end, :]

        # attended mel
        mel_grp_att = self.mel[att_name]
        mel_att_full = mel_grp_att['mel']
        mel_sr = mel_grp_att.attrs['sr']
        mel_hop = mel_grp_att.attrs['hop_length']

        frame_start = int(att_start_sec * mel_sr / mel_hop)
        num_frames = int(length_sec * mel_sr / mel_hop)
        frame_end = frame_start + num_frames

        if frame_end > mel_att_full.shape[1]:
            frame_end = mel_att_full.shape[1]

        mel_att = mel_att_full[:, frame_start:frame_end]
        mel_att = pad_2d_time_last(mel_att, num_frames)

        # load ign mel
        mel_grp_ign = self.mel[ign_name]
        mel_ign_full = mel_grp_ign['mel']
        frame_start_ign = int(ign_start_sec * mel_sr / mel_hop)
        frame_end_ign = frame_start_ign + num_frames

        if frame_end_ign > mel_ign_full.shape[1]:
            frame_end_ign = mel_ign_full.shape[1]

        mel_ign = mel_ign_full[:, frame_start_ign:frame_end_ign]
        mel_ign = pad_2d_time_last(mel_ign, num_frames)

        return (
            torch.tensor(eeg_att, dtype=torch.float32),
            torch.tensor(mel_att, dtype=torch.float32).unsqueeze(0),
            torch.tensor(mel_ign, dtype=torch.float32).unsqueeze(0),
        )

    def __len__(self):
        return len(self.items)


class Stage2Dataset(data.Dataset):
    def __init__(self, args, partition):
        self.args = args
        self.partition = partition
        self.audio_sr = args.audio_sr
        self.eeg_sr = args.eeg_sr
        self.audio_dir = args.audio_dir
        self.ign_aug_prob = args.stage2.ign_aug_prob

        self.metadata = open(args.metadata_path).read().splitlines()
        self.metadata = list(filter(lambda x: x.split(',')[0] == partition, self.metadata))

        if args.subject_ids is not None:
            subject_ids = set(int(s) for s in args.subject_ids)
            self.metadata = [
                line for line in self.metadata
                if int(line.split(',')[1]) in subject_ids
            ]

        self.eeg_path = args.eeg_dir / 'eeg.h5'
        self.mel_path = args.audio_dir / 'mel.h5'
        self.eeg = None
        self.mel = None
        self._h5_pid = None

        print(f'The {partition} dataloader has prepared ({len(self.metadata)} samples).')

    def _ensure_h5(self):
        pid = os.getpid()
        if self._h5_pid == pid and self.eeg is not None and self.mel is not None:
            return
        self._close_h5()
        self.eeg = h5py.File(self.eeg_path, 'r')
        self.mel = h5py.File(self.mel_path, 'r')
        self._h5_pid = pid

    def _close_h5(self):
        for handle in (self.eeg, self.mel):
            if handle is not None:
                handle.close()
        self.eeg = None
        self.mel = None
        self._h5_pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(eeg=None, mel=None, _h5_pid=None)
        return state

    def __del__(self):
        self._close_h5()

    def __getitem__(self, index):
        self._ensure_h5()
        metadata_att = self.metadata[index].split(',')
        length_second = float(metadata_att[-1])
        length_eeg = math.floor(length_second * self.eeg_sr)
        length_audio = math.floor(length_second * self.audio_sr)

        # load target eeg
        subject, trial = metadata_att[1], int(metadata_att[2])
        start_sec_att = float(metadata_att[4])
        eeg_start = int(start_sec_att * self.eeg_sr)
        eeg_end = eeg_start + length_eeg
        eeg_att = self.eeg[f'S{subject}/Tra{trial}'][eeg_start:eeg_end, :]

        # load att audio
        a_att_name = metadata_att[3]
        a_att_path = self.audio_dir / a_att_name
        start_att = int(start_sec_att * self.audio_sr)
        end_att = int(start_att + length_audio)
        a_att, _ = sf.read(a_att_path, start=start_att, stop=end_att, dtype='float32')

        # load att mel
        mel_grp_att = self.mel[metadata_att[3]]
        mel_full_att = mel_grp_att['mel']
        mel_sr = mel_grp_att.attrs['sr']
        mel_hop = mel_grp_att.attrs['hop_length']

        frame_start_att = int(start_sec_att * mel_sr / mel_hop)
        num_frames = int(length_second * mel_sr / mel_hop)
        frame_end_att = frame_start_att + num_frames
        T_total_att = mel_full_att.shape[1]

        if frame_end_att > T_total_att:
            frame_end_att = T_total_att

        mel_att = mel_full_att[:, frame_start_att:frame_end_att]
        mel_att = pad_2d_time_last(mel_att, num_frames)

        # ignored audio augmentation
        metadata_ign = metadata_att
        use_other_ign = (np.random.rand() < self.ign_aug_prob)

        if use_other_ign and self.partition == 'train':
            rand_idx = np.random.randint(len(self.metadata))
            metadata_ign = self.metadata[rand_idx].split(',')
            a_ign_name = metadata_ign[6]
            start_sec_ign = float(metadata_ign[7])

            if a_ign_name == a_att_name and start_sec_ign == start_sec_att:
                metadata_ign = metadata_att

        # load ign audio
        a_ign_path = self.audio_dir / metadata_ign[6]
        start_sec_ign = float(metadata_ign[7])
        start_ign = int(start_sec_ign * self.audio_sr)
        end_ign = int(start_ign + length_audio)
        a_ign, _ = sf.read(a_ign_path, start=start_ign, stop=end_ign, dtype='float32')

        # load ign mel
        mel_grp_ign = self.mel[metadata_ign[6]]
        mel_full_ign = mel_grp_ign['mel']
        frame_start_ign = int(start_sec_ign * mel_sr / mel_hop)
        num_frames_ign = int(length_second * mel_sr / mel_hop)
        frame_end_ign = frame_start_ign + num_frames_ign
        T_total_ign = mel_full_ign.shape[1]

        if frame_end_ign > T_total_ign:
                frame_end_ign = T_total_ign

        mel_ign = mel_full_ign[:, frame_start_ign:frame_end_ign]
        mel_ign = pad_2d_time_last(mel_ign, num_frames)

        # training snr augmentation
        if self.partition == 'train':
            snr = np.random.uniform(-5, 5)
        else:
            snr = float(metadata_att[8])

        if snr != 0:
            att_power = np.linalg.norm(a_att, 2) ** 2 / a_att.size
            ign_power = np.linalg.norm(a_ign, 2) ** 2 / a_ign.size
            a_ign *= np.sqrt(att_power / ign_power)
            snr_1 = (10 ** (snr/20))

            max_snr = max(1, snr_1)
            a_att /= max_snr
            a_ign /= max_snr
            a_att = a_att * snr_1

        a_mix = a_att + a_ign

        # audio normalization
        max_val = np.max(np.abs(a_mix))

        if max_val > 1:
            a_mix /= max_val
            a_att /= max_val
            a_ign /= max_val

        return (
            torch.tensor(a_mix, dtype=torch.float32),
            torch.tensor(a_att, dtype=torch.float32),
            torch.tensor(a_ign, dtype=torch.float32),
            torch.tensor(eeg_att, dtype=torch.float32),
            torch.tensor(mel_att, dtype=torch.float32).unsqueeze(0),
            torch.tensor(mel_ign, dtype=torch.float32).unsqueeze(0),
        )

    def __len__(self):
        return len(self.metadata)


def pad_2d_time_last(x, target_len):
    if x.shape[1] >= target_len:
        return x[:, :target_len]

    out = np.zeros((x.shape[0], target_len), dtype=x.dtype)
    out[:, :x.shape[1]] = x
    return out
