import os
from tqdm.auto import tqdm
from PIL import Image
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
import pytorch_lightning as pl


class LitPredictor(pl.LightningModule):
    """Predicts masks and exports them. Supports multi-gpu inference."""
    def __init__(self, model, export_fn, raw=False):
        super().__init__()
        self.model = model
        self.export_fn = export_fn
        self.raw = raw

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        features, metadata = batch
        outputs = self.model(features)
        if self.raw:
            # Keep raw input and device (e.g. for mask filling)
            self.export_fn(outputs, batch)
            return

        out = outputs['out'].cpu().detach()

        # Upscale
        size = (features['image'].size(2), features['image'].size(3))
        out = TF.resize(out, size, interpolation=Image.BILINEAR)
        out = out.numpy()

        self.export_fn(out, batch)
