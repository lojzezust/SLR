import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from functools import partial
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from slr.datasets import FolderDataset
from slr.datasets.transforms import PytorchHubNormalization
from slr.predictor import LitPredictor
import slr.models as M
from slr.utils import load_weights


ARCHITECTURE = 'wasr_resnet101_imu'
# Colors corresponding to each segmentation class
SEGMENTATION_COLORS = np.array([
    [247, 195, 37],
    [41, 167, 224],
    [90, 75, 164]
], np.uint8)

BATCH_SIZE = 4
OUTPUT_DIR = 'output/predictions/general'


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="SLR models general inference")
    parser.add_argument("--architecture", type=str, choices=M.models, default=ARCHITECTURE,
                        help="Which architecture to use.")
    parser.add_argument("--weights-file", type=str, required=True,
                        help="Path to the weights of the model.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Minibatch size (number of samples) used on each device.")
    parser.add_argument("--image-dir", type=str, required=True,
                        help="Path to the directory containing input images.")
    parser.add_argument("--imu-dir", type=str, default=None,
                        help="(optional) Path to the directory containing input IMU masks.")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help="Root directory for output prediction saving. Predictions are saved inside model subdir.")
    parser.add_argument("--fp16", action='store_true',
                        help="Use half precision for inference.")
    parser.add_argument("--gpus", default=-1,
                        help="Number of gpus (or GPU ids) used for training.")
    return parser.parse_args()

def export_predictions(probs, batch, output_dir=OUTPUT_DIR):
    features, metadata = batch

    # Class prediction
    out_class = probs.argmax(1).astype(np.uint8)

    for i, pred_mask in enumerate(out_class):
        pred_mask = SEGMENTATION_COLORS[pred_mask]
        mask_img = Image.fromarray(pred_mask)

        out_path = output_dir / Path(metadata['image_path'][i]).with_suffix('.png')
        if not out_path.parent.exists():
            out_path.parent.mkdir(parents=True, exist_ok=True)

        mask_img.save(str(out_path))

def predict_folder(args):
    dataset = FolderDataset(args.image_dir, args.imu_dir, normalize_t=PytorchHubNormalization())
    dl = DataLoader(dataset, batch_size=args.batch_size, num_workers=1)

    model = M.get_model(args.architecture)
    weights = load_weights(args.weights_file)
    model.load_state_dict(weights)

    output_dir = Path(args.output_dir)
    export_fn = partial(export_predictions, output_dir=output_dir)
    predictor = LitPredictor(model, export_fn)

    precision = 16 if args.fp16 else 32
    trainer = pl.Trainer(gpus=args.gpus,
                         accelerator='ddp',
                         precision=precision,
                         logger=False)

    trainer.predict(predictor, dl)

def main():
    args = get_arguments()
    print(args)

    predict_folder(args)


if __name__ == '__main__':
    main()
