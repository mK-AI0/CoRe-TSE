import os
import torch
import hydra
import random
import numpy as np
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from models import EEGEncoder, AudioEncoder, Extractor
from datasets import build_dataloader
from trainers import Stage1Trainer, Stage2Trainer


def setup_config(args: DictConfig) -> DictConfig:
    OmegaConf.set_struct(args, False)
    # Optional per-experiment cap.  This is needed when several folds share a
    # host: PyTorch otherwise initializes one CPU pool per process and can
    # starve both HDF5 loading and CUDA submission.
    if args.get('torch_num_threads') is not None:
        threads = int(args.torch_num_threads)
        if threads < 1:
            raise ValueError(f'torch_num_threads must be >= 1, got {threads}')
        torch.set_num_threads(threads)
        try:
            torch.set_num_interop_threads(threads)
        except RuntimeError:
            # This only occurs if another library initialized the inter-op pool
            # before config setup; the intra-op cap still applies.
            pass
    os.makedirs(args.log_dir, exist_ok=True)
    config_path = os.path.join(args.log_dir, 'train_config.yaml')
    OmegaConf.save(config=args, f=config_path)

    path_keys = [
        'log_dir',
        'metadata_path',
        'audio_dir',
        'eeg_dir',
    ]

    for key in path_keys:
        val = args.get(key)
        args[key] = Path(val)

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return args


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_stage1(args):
    # Stage-1 contrastive dot products require matching EEG/audio dimensions.
    # Default remains 48, while the capacity-control config alone selects 96.
    embedding_dim = int(getattr(args, 'eeg_hidden', 48))
    eeg_encoder = EEGEncoder(hidden=embedding_dim).to(args.device)
    audio_encoder = AudioEncoder(d_model=embedding_dim).to(args.device)

    train_loader = build_dataloader(args, stage=1, partition='train')
    val_loader = build_dataloader(args, stage=1, partition='val')
    test_loader = build_dataloader(args, stage=1, partition='test')

    s1_trainer = Stage1Trainer(
        args,
        eeg_encoder,
        audio_encoder,
        train_loader,
        val_loader,
        test_loader,
    )

    s1_trainer.train()
    s1_trainer.evaluate()


def run_stage2(args):
    # A full two-stage run can be interrupted while Stage-1 is still running.
    # In that case Stage-1 should resume, but Stage-2 must start freshly once
    # Stage-1 finishes because no Stage-2 optimizer/model state exists yet.
    stage2_last = args.log_dir / 'Stage2' / 'last_checkpoint.pt'
    if args.train_from_last_checkpoint and not stage2_last.is_file():
        args.train_from_last_checkpoint = False

    extractor = Extractor(args).to(args.device)
    audio_encoder = AudioEncoder(d_model=int(getattr(args, 'eeg_hidden', 48))).to(args.device)
    # A Stage-2-only continuation may consume a completed Stage-1 checkpoint
    # from an isolated source tree without modifying that source experiment.
    pretrained_path = Path(args.stage1_checkpoint) if args.get('stage1_checkpoint') else args.log_dir / 'Stage1' / 'last_best_checkpoint.pt'
    ckpt = torch.load(pretrained_path, map_location='cpu')
    extractor.eeg_encoder.load_state_dict(ckpt['eeg_encoder'])
    audio_encoder.load_state_dict(ckpt['audio_encoder'])

    for p in extractor.eeg_encoder.parameters():
        p.requires_grad = False

    train_loader = build_dataloader(args, stage=2, partition='train')
    val_loader = build_dataloader(args, stage=2, partition='val')
    test_loader = build_dataloader(args, stage=2, partition='test')

    s2_trainer = Stage2Trainer(
        args,
        extractor,
        audio_encoder,
        train_loader,
        val_loader,
        test_loader,
    )

    s2_trainer.train()
    s2_trainer.evaluate()


@hydra.main(version_base=None, config_path='config', config_name='config_KUL')
def main(args: DictConfig):
    args = setup_config(args)
    set_seed(args.seed)
    run_stage1(args)
    if args.run_stage2:
        run_stage2(args)


if __name__ == '__main__':
    main()
