#!/usr/bin/env python3
"""训练 KUL/DTU fold 对齐的 TRUST attended+in-batch 等权双编码器集成。"""
import json
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from datasets import build_dataloader
from ensemble import (EqualWeightEEGExtractor, EnsembleStage2Trainer, branch_selection_accuracy,
                      ensemble_selection_accuracy, load_stage1_branch, validation_reliability_weights)
from train import set_seed, setup_config


@hydra.main(version_base=None, config_path='config', config_name='config_KUL_ensemble_attended_inbatch')
def main(args: DictConfig):
    args = setup_config(args)
    set_seed(args.seed)
    branch_paths = [Path(args.attended_stage1_path), Path(args.inbatch_stage1_path)]
    allow_duplicate = bool(getattr(args, 'allow_duplicate_branches', False))
    if ((branch_paths[0] == branch_paths[1] and not allow_duplicate)
            or not all(path.is_file() for path in branch_paths)):
        raise FileNotFoundError('Two existing fold-matched Stage-1 checkpoints are required; '
                            'duplicate paths require allow_duplicate_branches=true.')
    branches = [load_stage1_branch(path, args.device) for path in branch_paths]
    eeg_encoders, audio_encoders = zip(*branches)

    reliability_mode = getattr(args, 'reliability_mode', 'uniform')
    if reliability_mode not in ('uniform', 'validation_accuracy'):
        raise ValueError(f'Unknown reliability_mode: {reliability_mode}')
    s1_val = build_dataloader(args, stage=1, partition='val')
    validation_accuracy = [branch_selection_accuracy(eeg_encoder, audio_encoder, s1_val, args.device)
                           for eeg_encoder, audio_encoder in zip(eeg_encoders, audio_encoders)]
    weights = (validation_reliability_weights(validation_accuracy)
               if reliability_mode == 'validation_accuracy' else torch.full((2,), 0.5))
    s1_test = build_dataloader(args, stage=1, partition='test')
    extractor = EqualWeightEEGExtractor(args, list(eeg_encoders), weights).to(args.device)
    stage1_acc = ensemble_selection_accuracy(extractor, list(audio_encoders), s1_test, args.device)
    stage1_dir = args.log_dir / 'Stage1'; stage1_dir.mkdir(exist_ok=True)
    (stage1_dir / 'branch_checkpoints.json').write_text(json.dumps({
        'branches': [str(path.resolve()) for path in branch_paths], 'allow_duplicate_branches': allow_duplicate,
        'reliability_mode': reliability_mode,
        'validation_accuracy': validation_accuracy, 'fixed_weights': weights.tolist(),
        'fusion': 'feature_concat_and_weighted_margin',
    }, indent=2), encoding='utf-8')
    (stage1_dir / 'evaluation.json').write_text(json.dumps({'Accuracy': stage1_acc, 'num_branches': 2,
                                                              'reliability_mode': reliability_mode,
                                                              'validation_accuracy': validation_accuracy,
                                                              'fixed_weights': weights.tolist()}, indent=2), encoding='utf-8')
    print(f'Validation branch accuracy: {[round(value, 2) for value in validation_accuracy]}')
    print(f'Fixed branch weights: {[round(float(value), 4) for value in weights]}')
    print(f'Ensemble Stage1 Accuracy: {stage1_acc:.2f}')

    train_loader = build_dataloader(args, stage=2, partition='train')
    val_loader = build_dataloader(args, stage=2, partition='val')
    test_loader = build_dataloader(args, stage=2, partition='test')
    trainer = EnsembleStage2Trainer(args, extractor, list(audio_encoders), train_loader, val_loader, test_loader)
    trainer.train(); trainer.evaluate()


if __name__ == '__main__':
    main()
