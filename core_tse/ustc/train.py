#!/usr/bin/env python3
"""Standalone TRUST-TSE runner for an existing USTC HDF5 fold."""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from datasets import build_dataloader
from models import AudioEncoder, EEGEncoder, Extractor
from trainers import Stage1Trainer, Stage2Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Run two-stage TRUST-TSE on one USTC HDF5 fold.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True, help=".../general/foldN/tid0 or .../adaptive/subN/tidN")
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--stage1-init", default=None, help="Stage1 last_best_checkpoint.pt from the general run")
    parser.add_argument("--stage2-init", default=None, help="Stage2 last_best_checkpoint.pt from the general run")
    parser.add_argument("--resume", action="store_true", help="Resume both stages from this run's last checkpoints")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override config seed for an isolated reproducibility run.")
    parser.add_argument("--stage1-eeg-init-seed", type=int, default=None,
                        help="Override only the initial EEG-encoder weights, then restore --seed before training. "
                             "Used by the homogeneous A1/A2 CKA control to hold data order and all training RNG fixed.")
    parser.add_argument("--stage1-only", action="store_true")
    parser.add_argument("--skip-stage1", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Smoke-test cap; normal runs leave unset")
    return parser.parse_args()


def namespace(mapping):
    return argparse.Namespace(**mapping)


def main():
    args = parse_args()
    cfg_dict = yaml.safe_load(Path(args.config).read_text())
    cfg_dict.update({"data_dir": str(Path(args.data_dir).resolve()), "log_dir": str(Path(args.log_dir).resolve())})
    if args.seed is not None:
        cfg_dict["seed"] = int(args.seed)
    if args.stage1_eeg_init_seed is not None:
        cfg_dict["stage1_eeg_init_seed"] = int(args.stage1_eeg_init_seed)
    cfg = namespace(cfg_dict)
    cfg.extractor = namespace(cfg_dict["extractor"])
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    # Limit intra-op/interop pools before workers are created.  Without this, each USTC
    # job spawns roughly one thread per host core and starves HDF5 loading under concurrency.
    torch.set_num_threads(int(getattr(cfg, "torch_num_threads", 1)))
    torch.set_num_interop_threads(int(getattr(cfg, "torch_num_interop_threads", 1)))
    print(f"Runtime threads: torch={torch.get_num_threads()}, interop={getattr(cfg, 'torch_num_interop_threads', 1)}, workers={cfg.num_workers}")
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.log_dir, "train_config.yaml").write_text(yaml.safe_dump(cfg_dict, sort_keys=False), encoding="utf-8")
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed); torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    if args.max_train_batches is not None:
        # This option is deliberately rejected for full training rather than silently changing results.
        raise NotImplementedError("Use tests/smoke.py for bounded checks; production runs must not truncate HDF5 loaders.")

    stage1_mode = getattr(cfg, "stage1_negative_mode", "attended_same_trial")
    stage1_first_mode = getattr(cfg, "stage1_alternating_first", "in_batch")
    # Keep the default sampler choice available to Stage1Trainer as well.
    # Older configs omit this optional field, but the trainer logs/branches on it.
    cfg.stage1_negative_mode = stage1_mode
    loader_prefetch = int(getattr(cfg, "prefetch_factor", 2))
    s1_train = build_dataloader(cfg.data_dir, "train", 1, cfg.stage1_eval_batch_size, cfg.trials_per_batch, cfg.windows_per_trial, cfg.num_workers, stage1_mode, loader_prefetch, stage1_first_mode)
    s1_val = build_dataloader(cfg.data_dir, "val", 1, cfg.stage1_eval_batch_size, cfg.trials_per_batch, cfg.windows_per_trial, cfg.num_workers, stage1_mode, loader_prefetch, stage1_first_mode)
    s1_test = build_dataloader(cfg.data_dir, "test", 1, cfg.stage1_eval_batch_size, cfg.trials_per_batch, cfg.windows_per_trial, cfg.num_workers, stage1_mode, loader_prefetch, stage1_first_mode)
    # Stage-1 的点积相似度要求 EEG/audio 嵌入维度相同；96-D 容量消融仅扩大
    # 两者的最终嵌入投影，保留单分支、数据、损失和其余架构不变。
    embedding_dim = int(getattr(cfg, "eeg_hidden", 48))
    # The homogeneous CKA control must vary only the EEG initialization, not
    # the sampler/dropout/random augmentation stream.  Build EEG with its
    # dedicated seed, initialise audio with the common run seed, then reset
    # torch's RNG before the first training batch is drawn.
    if args.stage1_eeg_init_seed is not None:
        torch.manual_seed(args.stage1_eeg_init_seed); torch.cuda.manual_seed_all(args.stage1_eeg_init_seed)
    eeg_encoder = EEGEncoder(hidden=embedding_dim).to(cfg.device)
    if args.stage1_eeg_init_seed is not None:
        torch.manual_seed(cfg.seed); torch.cuda.manual_seed_all(cfg.seed)
    audio_encoder = AudioEncoder(d_model=embedding_dim).to(cfg.device)
    if args.stage1_eeg_init_seed is not None:
        torch.manual_seed(cfg.seed); torch.cuda.manual_seed_all(cfg.seed)
        print(f"Stage-1 homogeneous CKA control: EEG init seed={args.stage1_eeg_init_seed}; training RNG reset to seed={cfg.seed}")
    stage1 = Stage1Trainer(cfg, eeg_encoder, audio_encoder, s1_train, s1_val, s1_test, Path(cfg.log_dir))
    if args.resume:
        stage1.load_init(str(Path(cfg.log_dir) / "Stage1" / "last_checkpoint.pt"), resume=True)
    else:
        stage1.load_init(args.stage1_init)
    if not args.skip_stage1:
        stage1.train(); stage1.evaluate()
    if args.stage1_only:
        return

    best_stage1 = Path(cfg.log_dir) / "Stage1" / "last_best_checkpoint.pt"
    if not best_stage1.exists():
        raise FileNotFoundError(f"Stage1 best checkpoint missing: {best_stage1}")
    s2_train = build_dataloader(cfg.data_dir, "train", 2, cfg.stage2_batch_size, num_workers=cfg.num_workers, prefetch_factor=loader_prefetch)
    s2_val = build_dataloader(cfg.data_dir, "val", 2, cfg.stage2_batch_size, num_workers=cfg.num_workers, prefetch_factor=loader_prefetch)
    s2_test = build_dataloader(cfg.data_dir, "test", 2, cfg.stage2_batch_size, num_workers=cfg.num_workers, prefetch_factor=loader_prefetch)
    extractor = Extractor(cfg).to(cfg.device)
    stage1_ckpt = torch.load(best_stage1, map_location="cpu", weights_only=False)
    extractor.eeg_encoder.load_state_dict(stage1_ckpt["eeg_encoder"])
    for param in extractor.eeg_encoder.parameters(): param.requires_grad = False
    stage2 = Stage2Trainer(cfg, extractor, audio_encoder, s2_train, s2_val, s2_test, Path(cfg.log_dir))
    if args.resume:
        stage2.load_init(str(Path(cfg.log_dir) / "Stage2" / "last_checkpoint.pt"), resume=True)
    elif args.stage2_init:
        stage2.load_init(args.stage2_init)
        # General extractor checkpoint must never overwrite fold-adapted frozen EEG features.
        extractor.eeg_encoder.load_state_dict(stage1_ckpt["eeg_encoder"])
    stage2.train(); stage2.evaluate()


if __name__ == "__main__":
    main()
