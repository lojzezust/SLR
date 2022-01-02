import argparse
import os
import math
import json
import torch
from torch.utils.data import DataLoader, ConcatDataset, BatchSampler, DistributedSampler
import torchvision.transforms as T
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import slr.models as M
from slr.trainer import LitModel
from slr.utils import Option, bool_arg
from slr.datasets import MaSTr1325Dataset
from slr.datasets.transforms import get_augmentation_transform, PytorchHubNormalization, LabelSmoothing, GaussLabelSmoothing

DEVICE_BATCH_SIZE = 3
TRAIN_FILE = 'data/mastr1325/train.yaml'
VAL_FILE = 'data/mastr1325/val.yaml'
NUM_CLASSES = 3
PATIENCE = 5
LOG_STEPS = 20
NUM_WORKERS = 1
NUM_GPUS = -1 # All visible GPUs
NUM_NODES = 1 # Single node training
RANDOM_SEED = None
OUTPUT_DIR = 'output'
PRETRAINED_DEEPLAB = True
PRECISION = 32
ARCHITECTURE = 'wasr_resnet101_imu'
MONITOR_VAR = 'val/iou/obstacle'
MONITOR_VAR_MODE = 'max'


def get_arguments(input_args=None):
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch-size", type=int, default=DEVICE_BATCH_SIZE,
                        help="Minibatch size (number of samples) used on each device.")
    parser.add_argument("--train-file", type=str, default=TRAIN_FILE,
                        help="Path to the file containing the MaSTr training dataset mapping.")
    parser.add_argument("--val-file", type=str, default=VAL_FILE,
                        help="Path to the file containing the MaSTr val dataset mapping.")
    parser.add_argument("--mask-dir", type=str, default=None,
                        help="Override the original mask dir. Relative path from the dataset root.")
    parser.add_argument("--validation", action="store_true",
                        help="Report performance on validation set and use early stopping.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--patience", type=Option(int), default=PATIENCE,
                        help="Patience for early stopping (how many epochs to wait without increase).")
    parser.add_argument("--log-steps", type=int, default=LOG_STEPS,
                        help="Number of steps between logging variables.")
    parser.add_argument("--num_nodes", type=int, default=NUM_NODES,
                        help="Number of nodes used for training.")
    parser.add_argument("--gpus", default=NUM_GPUS,
                        help="Number of gpus (or GPU ids) used for training.")
    parser.add_argument("--preload-data", action='store_true',
                        help="Preload the data into memory.")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS,
                        help="Number of workers used for data loading.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--pretrained", type=bool, default=PRETRAINED_DEEPLAB,
                        help="Use pretrained DeepLab weights.")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help="Directory where the output will be stored (models and logs)")
    parser.add_argument("--model-name", type=str, required=True,
                        help="Name of the model. Used to create model and log directories inside the output directory.")
    parser.add_argument("--pretrained-weights", type=str, default=None,
                        help="Path to the pretrained weights to be used.")
    parser.add_argument("--architecture", type=str, choices=M.models, default=ARCHITECTURE,
                        help="Which architecture to use.")
    parser.add_argument("--monitor-metric", type=str, default=MONITOR_VAR,
                        help="Validation metric to monitor for early stopping and best model saving.")
    parser.add_argument("--monitor-metric-mode", type=str, default=MONITOR_VAR_MODE, choices=['min', 'max'],
                        help="Maximize or minimize the monitored metric.")
    parser.add_argument("--no-augmentation", action="store_true",
                        help="Disable on-the-fly image augmentation of the dataset.")
    parser.add_argument("--precision", default=PRECISION, type=int, choices=[16,32],
                        help="Floating point precision.")
    parser.add_argument("--ls-alpha", default=None, type=float,
                        help="Label smoothing alpha in range [0,1]. No label smoothing if not set.")
    parser.add_argument("--svls-sigma", default=None, type=float,
                        help="Gaussian label smoothing sigma. No gaussian label smoothing if not set.")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Resume training from specified checkpoint.")

    parser = LitModel.add_argparse_args(parser)

    args = parser.parse_args(input_args)

    return args

class DataModule(pl.LightningDataModule):
    def __init__(self, args, normalize_t):
        super().__init__()
        self.args = args
        self.normalize_t = normalize_t

    def train_dataloader(self):
        transforms = []

        # Label smoothing transform(s)
        if self.args.ls_alpha is not None:
            transforms.append(LabelSmoothing(self.args.ls_alpha))

        if self.args.svls_sigma is not None:
            transforms.append(GaussLabelSmoothing(sigma=self.args.svls_sigma))

        # Finally, data augmentation transform
        if not self.args.no_augmentation:
            transforms.append(get_augmentation_transform())

        transform = None
        if len(transforms) > 0:
            transform = T.Compose(transforms)

        # If using mask filling, use alternative mask subdir
        alternative_mask_subdir = None
        if self.args.mask_dir is not None:
            alternative_mask_subdir = self.args.mask_dir

        train_ds = MaSTr1325Dataset(self.args.train_file, transform=transform,
                                    normalize_t=self.normalize_t, masks_subdir=alternative_mask_subdir, preload=self.args.preload_data)

        train_dl = DataLoader(train_ds, batch_size=self.args.batch_size, shuffle=True,
                                num_workers=self.args.workers, drop_last=True)
        return train_dl

    def val_dataloader(self):
        val_dl = None
        if self.args.validation:
            val_ds = MaSTr1325Dataset(self.args.val_file, normalize_t=self.normalize_t, include_original=True, preload=self.args.preload_data)
            val_dl = DataLoader(val_ds, batch_size=self.args.batch_size, num_workers=self.args.workers)

        return val_dl


def train_wasr(args):
    # Use or create random seed
    args.random_seed = pl.seed_everything(args.random_seed)

    normalize_t = PytorchHubNormalization()
    data = DataModule(args, normalize_t)

    model = M.get_model(args.architecture, num_classes=args.num_classes, pretrained=args.pretrained)

    if args.pretrained_weights is not None:
        print(f"Loading weights from: {args.pretrained_weights}")
        state_dict = torch.load(args.pretrained_weights, map_location='cpu')
        if 'model' in state_dict:
            # Loading weights from checkpoint
            model.load_state_dict(state_dict['model'])
        else:
            model.load_state_dict(state_dict)

    model = LitModel(model, args.num_classes, args)

    logs_path = os.path.join(args.output_dir, 'logs')
    logger = pl_loggers.TensorBoardLogger(logs_path, args.model_name)
    logger.log_hyperparams(args)

    callbacks = []
    if args.validation:
        # Val: Early stopping and best model saving
        if args.patience is not None:
            callbacks.append(EarlyStopping(monitor=args.monitor_metric, patience=args.patience, mode=args.monitor_metric_mode))
        callbacks.append(ModelCheckpoint(save_last=True, save_top_k=1, monitor=args.monitor_metric, mode=args.monitor_metric_mode))

    trainer = pl.Trainer(logger=logger,
                         gpus=args.gpus,
                         num_nodes=args.num_nodes,
                         max_epochs=args.epochs,
                         accelerator='ddp',
                         resume_from_checkpoint=args.resume_from,
                         callbacks=callbacks,
                         sync_batchnorm=True,
                         log_every_n_steps=args.log_steps,
                         precision=args.precision)
    trainer.fit(model, data)


def main():
    args = get_arguments()
    print(args)

    train_wasr(args)


if __name__ == '__main__':
    main()
