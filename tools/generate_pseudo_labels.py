"""Fills weak masks using feature clustering on backbone features."""

import os
import torch
import numpy as np
import argparse
from PIL import Image
from functools import partial

import torchvision.transforms.functional as TF
import pytorch_lightning as pl

from slr.utils import bool_arg, mkdir_safe, load_weights
from slr.predictor import LitPredictor
from slr.datasets.transforms import PytorchHubNormalization
from slr.datasets.mastr import MaSTr1325Dataset
from slr.mask_refine import mask_refine
import slr.models as M

# Constants
SHUFFLE = False
PER_INSTANCE = True # Per instance or per batch prototype computation

# Defaults
ARCHITECTURE = 'wasr_resnet101_imu'
MASTR_FILE = os.path.expanduser('data/mastr1325/all_weak.yaml')
BATCH_SIZE = 4
WORKERS = 1
FILL_WEIGHT = 0.5
ENTROPY_LIMIT = 1.0
OUTPUT_DIR = os.path.expanduser('data/mastr1325/pseudo_labels/wasr_slr_warmup')
CONSTRAINED = True

def get_arguments(input_args=None):
    """Parse all the arguments provided from the CLI."""

    parser = argparse.ArgumentParser(description="Fill weak masks.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--architecture", type=str, choices=M.models, default=ARCHITECTURE,
                        help="Which architecture to use.")
    parser.add_argument("--weights_file", type=str, required=True,
                        help="Path to the weights of the model.")
    parser.add_argument("--mastr_file", type=str, default=MASTR_FILE,
                        help="Dataset to use for mask prediction.")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                        help="Mask output directory.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Batch size.")
    parser.add_argument("--workers", type=int, default=WORKERS,
                        help="Number of dataloader workers.")
    parser.add_argument("--fill_weight", type=float, default=FILL_WEIGHT,
                        help="Fill weight.")
    parser.add_argument("--entropy_limit", type=float, default=ENTROPY_LIMIT,
                        help="Normalized entropy limit. Areas with larger entropy are ignored in training.")
    parser.add_argument("--constrained", type=bool_arg, default=CONSTRAINED,
                        help="Do not use domain and annotation constraints.")
    parser.add_argument("--fp16", action='store_true',
                        help="Use half precision for inference.")
    parser.add_argument("--gpus", default=-1,
                    help="Number of gpus (or GPU ids) used for training.")

    args = parser.parse_args(input_args)

    return args


def _process_batch(outputs, batch, args=None):
    feat, lbl = batch
    mask_filenames = lbl['mask_filename']
    mask = lbl['segmentation']
    imu = feat['imu_mask']
    objects = lbl['objects'].float()

    logits = outputs['out'].detach()
    logits = TF.resize(logits, (mask.size(2), mask.size(3)), interpolation=Image.BILINEAR)
    probs = logits.softmax(1)
    features = outputs['aux'].detach()

    mask_filled = mask_refine(mask, probs, features, imu, objects, per_instance=PER_INSTANCE,
                            fill_weight=args.fill_weight, entropy_limit=args.entropy_limit, unconstrained=not args.constrained)

    # Export mask
    mask_filled_rgb = (mask_filled * 255).permute(0,2,3,1).cpu().numpy().astype(np.uint8)

    for i, mask in enumerate(mask_filled_rgb):
        mask_img = Image.fromarray(mask)
        img_path = os.path.join(args.output_dir, mask_filenames[i])
        mask_img.save(img_path)

def fill_weak_masks(args):
    mkdir_safe(args.output_dir)

    model = M.get_model(args.architecture)
    state_dict = load_weights(args.weights_file)
    model.load_state_dict(state_dict)

    ds = MaSTr1325Dataset(args.mastr_file, normalize_t=PytorchHubNormalization(), include_original=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=SHUFFLE, num_workers=args.workers)

    export_fn = partial(_process_batch, args=args)
    predictor = LitPredictor(model, export_fn, raw=True)

    precision = 16 if args.fp16 else 32
    trainer = pl.Trainer(gpus=args.gpus,
                         accelerator='ddp',
                         precision=precision,
                         logger=False)

    trainer.predict(predictor, dl)


def main():
    args = get_arguments()
    fill_weak_masks(args)


if __name__ == '__main__':
    main()
