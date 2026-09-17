#!/usr/bin/env python3
"""Train Stage-2 TRUST-USTC with frozen, fold-matched Stage-1 encoder branches."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from datasets import build_dataloader
from ensemble import (AlignedGatedEEGExtractor, EnsembleStage2Trainer, MultiEEGExtractor, branch_selection_accuracy,
                      ensemble_selection_accuracy, initialize_residual_from_single_stage2,
                      load_stage1_branch, validation_reliability_weights)
from trainers import MelFrontend


def parse_args():
    parser = argparse.ArgumentParser(description="Run frozen multi-Stage-1 ensemble Stage-2 on one USTC fold.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--branch-stage1", required=True, nargs="+", help="Fold-matched Stage1 last_best_checkpoint.pt paths.")
    parser.add_argument("--reliability-mode", choices=("uniform", "validation_accuracy"), default="uniform")
    parser.add_argument("--allow-duplicate-branches", action="store_true")
    parser.add_argument("--stage2-init", default=None, help="Ensemble general Stage2 last_best_checkpoint.pt")
    parser.add_argument("--residual-stage2-init", default=None,
                        help="Single-branch attended Stage2 checkpoint for zero-extra-channel residual initialization.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fusion-mode", choices=("concat", "aligned_gated"), default="concat")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override config seed for an isolated reproducibility run.")
    return parser.parse_args()


def namespace(mapping):
    return argparse.Namespace(**mapping)


def main():
    args = parse_args()
    if len(args.branch_stage1) != 2:
        raise ValueError("This experiment is fixed to the two branches: attended_same_trial and in_batch.")
    cfg_dict = yaml.safe_load(Path(args.config).read_text())
    cfg_dict.update({"data_dir": str(Path(args.data_dir).resolve()), "log_dir": str(Path(args.log_dir).resolve())})
    if args.seed is not None:
        cfg_dict["seed"] = int(args.seed)
    cfg = namespace(cfg_dict); cfg.extractor = namespace(cfg_dict["extractor"])
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    # Limit intra-op/interop pools before workers are created.  Without this, each USTC
    # job spawns roughly one thread per host core and starves HDF5 loading under concurrency.
    torch.set_num_threads(int(getattr(cfg, "torch_num_threads", 1)))
    torch.set_num_interop_threads(int(getattr(cfg, "torch_num_interop_threads", 1)))
    print(f"Runtime threads: torch={torch.get_num_threads()}, interop={getattr(cfg, 'torch_num_interop_threads', 1)}, workers={cfg.num_workers}")
    log_dir = Path(cfg.log_dir); log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "train_config.yaml").write_text(yaml.safe_dump(cfg_dict, sort_keys=False), encoding="utf-8")
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed); torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

    branch_paths = [Path(path).resolve() for path in args.branch_stage1]
    if len(set(branch_paths)) != len(branch_paths) and not args.allow_duplicate_branches:
        raise ValueError("Ensemble branches must be distinct checkpoints unless --allow-duplicate-branches is set.")
    branches = [load_stage1_branch(path, cfg.device) for path in branch_paths]
    eeg_encoders, audio_encoders = zip(*branches)

    loader_prefetch = int(getattr(cfg, "prefetch_factor", 2))
    s1_val = build_dataloader(cfg.data_dir, "val", 1, cfg.stage1_eval_batch_size,
                              cfg.trials_per_batch, cfg.windows_per_trial, cfg.num_workers, "in_batch", loader_prefetch)
    s1_test = build_dataloader(cfg.data_dir, "test", 1, cfg.stage1_eval_batch_size,
                               cfg.trials_per_batch, cfg.windows_per_trial, cfg.num_workers,
                               "in_batch", loader_prefetch)
    frontend = MelFrontend(cfg.sample_rate).to(cfg.device)
    validation_accuracy = [branch_selection_accuracy(eeg_encoder, audio_encoder, frontend, s1_val, cfg.device)
                           for eeg_encoder, audio_encoder in zip(eeg_encoders, audio_encoders)]
    weights = (validation_reliability_weights(validation_accuracy) if args.reliability_mode == "validation_accuracy"
               else torch.full((2,), 0.5))
    extractor_cls = AlignedGatedEEGExtractor if args.fusion_mode == "aligned_gated" else MultiEEGExtractor
    extractor = extractor_cls(cfg, list(eeg_encoders), weights).to(cfg.device)
    for parameter in extractor.eeg_encoders.parameters():
        parameter.requires_grad = False
    stage1_dir = log_dir / "Stage1"; stage1_dir.mkdir(exist_ok=True)
    stage1_acc = ensemble_selection_accuracy(extractor, list(audio_encoders), frontend, s1_test, cfg.device)
    branch_meta = {"branches": [str(path) for path in branch_paths], "reliability_mode": args.reliability_mode,
                   "validation_accuracy": validation_accuracy, "fixed_weights": weights.tolist(), "fusion": args.fusion_mode}
    (stage1_dir / "branch_checkpoints.json").write_text(json.dumps(branch_meta, indent=2), encoding="utf-8")
    (stage1_dir / "evaluation.json").write_text(json.dumps({"Accuracy": stage1_acc, "num_branches": 2,
                                                              "reliability_mode": args.reliability_mode,
                                                              "fixed_weights": weights.tolist()}, indent=2), encoding="utf-8")
    print(f"Validation branch accuracy: {[round(value, 2) for value in validation_accuracy]}")
    print(f"Fixed branch weights: {[round(float(value), 4) for value in weights]}")
    print(f"Ensemble Stage1 Accuracy: {stage1_acc:.2f}")

    s2_train = build_dataloader(cfg.data_dir, "train", 2, cfg.stage2_batch_size, num_workers=cfg.num_workers, prefetch_factor=loader_prefetch)
    s2_val = build_dataloader(cfg.data_dir, "val", 2, cfg.stage2_batch_size, num_workers=cfg.num_workers, prefetch_factor=loader_prefetch)
    s2_test = build_dataloader(cfg.data_dir, "test", 2, cfg.stage2_batch_size, num_workers=cfg.num_workers, prefetch_factor=loader_prefetch)
    trainer = EnsembleStage2Trainer(cfg, extractor, list(audio_encoders), s2_train, s2_val, s2_test, log_dir)
    if args.resume:
        trainer.load_init(str(log_dir / "Stage2" / "last_checkpoint.pt"), resume=True)
    elif args.residual_stage2_init:
        init_meta = initialize_residual_from_single_stage2(extractor, args.residual_stage2_init)
        (log_dir / "Stage2" / "residual_initialization.json").write_text(json.dumps(init_meta, indent=2), encoding="utf-8")
        print(f"Residual Stage2 initialization: {init_meta['source']}")
    elif args.stage2_init:
        trainer.load_init(args.stage2_init)
    trainer.train(); trainer.evaluate()


if __name__ == "__main__":
    main()
