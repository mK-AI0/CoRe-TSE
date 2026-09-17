"""USTC HDF5 adapter for the two-stage TRUST-TSE training pipeline."""
from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


FIELDS = ("eegs", "clean", "noisy", "unattended", "subjects", "trial")


class USTCH5Split(Dataset):
    """Read one existing USTC split without materialising its waveform arrays."""

    def __init__(self, root: str | Path, partition: str, stage1: bool = False):
        self.root = Path(root)
        self.partition = partition
        self.stage1 = stage1
        self.paths = {name: self.root / f"{name}_{partition}.h5" for name in FIELDS}
        missing = [str(path) for path in self.paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing USTC HDF5 split files: " + ", ".join(missing))
        self._files: dict[str, h5py.File] = {}
        self._keys: dict[str, str] = {}
        with h5py.File(self.paths["clean"], "r") as f:
            self.length = len(f[next(iter(f))])
        self.subjects = self._read_labels("subjects")
        self.trials = self._read_labels("trial")
        self.indices = list(range(self.length))
        self.group_keys = [(int(s), int(t)) for s, t in zip(self.subjects, self.trials)]
        if stage1 and partition == "train":
            self.indices = self._canonical_clean_indices()
        self.samples_by_trial: dict[tuple[int, int], list[int]] = defaultdict(list)
        # Samplers operate on Dataset positions; self.indices stores HDF5 rows.
        for position, index in enumerate(self.indices):
            self.samples_by_trial[self.group_keys[index]].append(position)
        self.trial_keys = sorted(self.samples_by_trial)
        print(f"The {partition} dataloader has prepared ({len(self.indices)} samples).")

    def _read_labels(self, name: str) -> np.ndarray:
        with h5py.File(self.paths[name], "r") as f:
            return np.asarray(f[next(iter(f))][:], dtype=np.int64)

    def _canonical_clean_indices(self) -> list[int]:
        """Keep one row per (subject, trial, exact clean waveform).

        The original USTC train split contains audio/EEG augmentation copies.
        TRUST positives and negatives must be distinct speech windows, so those
        copies cannot be treated as separate Stage-1 candidates.
        """
        selected: dict[tuple[int, int, bytes], int] = {}
        with h5py.File(self.paths["clean"], "r") as f:
            clean = f[next(iter(f))]
            for idx in range(self.length):
                waveform = np.asarray(clean[idx], dtype=np.float32)
                digest = hashlib.blake2b(waveform.tobytes(), digest_size=12).digest()
                key = (*self.group_keys[idx], digest)
                selected.setdefault(key, idx)
        return sorted(selected.values())

    def _open(self, name: str):
        if name not in self._files:
            self._files[name] = h5py.File(self.paths[name], "r")
            self._keys[name] = next(iter(self._files[name]))
        return self._files[name][self._keys[name]]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int):
        index = self.indices[item]
        eeg = np.asarray(self._open("eegs")[index], dtype=np.float32).T  # [T, C]
        clean = np.asarray(self._open("clean")[index], dtype=np.float32)
        if self.stage1:
            unattended = np.asarray(self._open("unattended")[index], dtype=np.float32)
            return {"eeg": torch.from_numpy(eeg), "clean": torch.from_numpy(clean),
                    "unattended": torch.from_numpy(unattended), "index": index}
        noisy = np.asarray(self._open("noisy")[index], dtype=np.float32)
        unattended = np.asarray(self._open("unattended")[index], dtype=np.float32)
        return {
            "mixture": torch.from_numpy(noisy), "attended": torch.from_numpy(clean),
            "unattended": torch.from_numpy(unattended), "eeg": torch.from_numpy(eeg),
            "subject": int(self.subjects[index]), "trial": int(self.trials[index]), "index": index,
        }

    def _close_files(self):
        for f in getattr(self, "_files", {}).values():
            try:
                f.close()
            except Exception:
                pass
        self._files = {}
        self._keys = {}

    def __getstate__(self):
        # Each DataLoader worker must open independent HDF5 handles; never fork a live handle.
        self._close_files()
        return self.__dict__.copy()

    def __del__(self):
        self._close_files()


class TrialAwareBatchSampler(Sampler[list[int]]):
    """Produce [trial windows] packs required by TRUST's within-trial NCE."""

    def __init__(self, dataset: USTCH5Split, trials_per_batch: int, windows_per_trial: int):
        self.dataset = dataset
        self.trials_per_batch = trials_per_batch
        self.windows_per_trial = windows_per_trial

    def _packs(self):
        packs = []
        for trial_key in self.dataset.trial_keys:
            indices = list(self.dataset.samples_by_trial[trial_key])
            random.shuffle(indices)
            for start in range(0, len(indices) - self.windows_per_trial + 1, self.windows_per_trial):
                packs.append(indices[start:start + self.windows_per_trial])
        random.shuffle(packs)
        return packs

    def __iter__(self):
        packs = self._packs()
        for start in range(0, len(packs) - self.trials_per_batch + 1, self.trials_per_batch):
            batch = []
            for pack in packs[start:start + self.trials_per_batch]:
                batch.extend(pack)
            yield batch

    def __len__(self):
        total = sum(len(v) // self.windows_per_trial for v in self.dataset.samples_by_trial.values())
        return total // self.trials_per_batch


class ScheduledStage1BatchSampler(Sampler[list[int]]):
    """Stage-1 sampler for the two same-encoder sampling schedules.

    ``mode='b_then_a'`` exposes one mode for the whole epoch.  The trainer
    changes it with :meth:`set_epoch_mode`, so the first part of training can
    use in-batch negatives and the second part can use TRUST's within-trial
    attended negatives.

    ``mode='alternate_batches'`` creates batches in the fixed order B, A, B,
    A, ... .  Both modes use the same batch size
    ``trials_per_batch * windows_per_trial``; A batches are generated from
    ``TrialAwareBatchSampler`` and B batches are ordinary shuffled batches.
    Keeping the order in this sampler (rather than inferring it from HDF5
    rows) makes the schedule explicit and reproducible in the run log.
    """

    VALID_MODES = {"in_batch", "attended_same_trial", "b_then_a", "alternate_batches"}

    def __init__(self, dataset: USTCH5Split, trials_per_batch: int, windows_per_trial: int,
                 mode: str = "alternate_batches", first_mode: str = "in_batch"):
        if mode not in {"b_then_a", "alternate_batches"}:
            raise ValueError(f"ScheduledStage1BatchSampler requires b_then_a or alternate_batches, got {mode}")
        if first_mode not in {"in_batch", "attended_same_trial"}:
            raise ValueError(f"first_mode must be in_batch or attended_same_trial, got {first_mode}")
        self.dataset = dataset
        self.trials_per_batch = int(trials_per_batch)
        self.windows_per_trial = int(windows_per_trial)
        self.batch_size = self.trials_per_batch * self.windows_per_trial
        self.schedule_mode = mode
        self.first_mode = first_mode
        self.epoch_mode = "in_batch"

    def set_epoch_mode(self, mode: str):
        if mode not in {"in_batch", "attended_same_trial"}:
            raise ValueError(f"epoch mode must be in_batch or attended_same_trial, got {mode}")
        self.epoch_mode = mode

    def _in_batch(self):
        positions = list(range(len(self.dataset)))
        random.shuffle(positions)
        return [positions[start:start + self.batch_size]
                for start in range(0, len(positions), self.batch_size)]

    def _attended(self):
        packs = []
        for trial_key in self.dataset.trial_keys:
            indices = list(self.dataset.samples_by_trial[trial_key])
            random.shuffle(indices)
            for start in range(0, len(indices) - self.windows_per_trial + 1, self.windows_per_trial):
                packs.append(indices[start:start + self.windows_per_trial])
        random.shuffle(packs)
        batches = []
        for start in range(0, len(packs) - self.trials_per_batch + 1, self.trials_per_batch):
            batch = []
            for pack in packs[start:start + self.trials_per_batch]:
                batch.extend(pack)
            batches.append(batch)
        return batches

    def __iter__(self):
        if self.schedule_mode == "b_then_a":
            batches = self._in_batch() if self.epoch_mode == "in_batch" else self._attended()
            yield from batches
            return
        # Build independent pools so that an A batch always has the trial
        # structure required by loss_nce, while B remains unrestricted.
        b_batches, a_batches = self._in_batch(), self._attended()
        n = max(len(b_batches), len(a_batches))
        b_index = a_index = 0
        for index in range(n):
            mode = self.first_mode if index % 2 == 0 else ("attended_same_trial" if self.first_mode == "in_batch" else "in_batch")
            if mode == "in_batch":
                # Separate cursors are essential: using the global alternating
                # index would skip half of each pool and can desynchronise the
                # sampler mode from the trainer loss near the shorter tail.
                yield b_batches[b_index]
                b_index += 1
            else:
                yield a_batches[a_index]
                a_index += 1

    def __len__(self):
        if self.schedule_mode == "b_then_a":
            if self.epoch_mode == "attended_same_trial":
                total = sum(len(v) // self.windows_per_trial for v in self.dataset.samples_by_trial.values())
                return total // self.trials_per_batch
            return math.ceil(len(self.dataset) / self.batch_size)
        return max(math.ceil(len(self.dataset) / self.batch_size),
                   sum(len(v) // self.windows_per_trial for v in self.dataset.samples_by_trial.values()) // self.trials_per_batch)


def build_dataloader(data_dir: str | Path, partition: str, stage: int, batch_size: int,
                     trials_per_batch: int = 4, windows_per_trial: int = 8,
                     num_workers: int = 0, stage1_negative_mode: str = "attended_same_trial",
                     prefetch_factor: int = 2, stage1_first_mode: str = "in_batch") -> DataLoader:
    dataset = USTCH5Split(data_dir, partition, stage1=(stage == 1))
    common = dict(num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    if num_workers > 0:
        common["prefetch_factor"] = prefetch_factor
    if stage == 1 and partition == "train" and stage1_negative_mode in {"attended_same_trial", "dual_loss_same_batch"}:
        # dual_loss_same_batch evaluates both NCE objectives over this same
        # trial-packed batch.  A needs this grouping; B then treats all other
        # items in the 32-sample batch as unrestricted in-batch candidates.
        return DataLoader(dataset, batch_sampler=TrialAwareBatchSampler(dataset, trials_per_batch, windows_per_trial), **common)
    if stage == 1 and partition == "train" and stage1_negative_mode in {"b_then_a", "alternate_batches"}:
        sampler = ScheduledStage1BatchSampler(dataset, trials_per_batch, windows_per_trial,
                                               mode=stage1_negative_mode, first_mode=stage1_first_mode)
        return DataLoader(dataset, batch_sampler=sampler, **common)
    if stage == 1 and partition == "train" and stage1_negative_mode in {"in_batch", "synchronous_unattended"}:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, **common)
    if stage == 1 and stage1_negative_mode not in {"attended_same_trial", "in_batch", "synchronous_unattended", "b_then_a", "alternate_batches", "dual_loss_same_batch"}:
        raise ValueError(f"Unknown Stage-1 negative mode: {stage1_negative_mode}")
    return DataLoader(dataset, batch_size=batch_size, shuffle=(partition == "train"), **common)
