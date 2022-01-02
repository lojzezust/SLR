from collections import OrderedDict

import torch
from torch import nn
from torchvision.models import segmentation

from .utils import IntermediateLayerGetter

def deeplabv3_resnet101(num_classes=3, pretrained=True):
    """Builds DeepLabV3 with ResNet101 backbone"""
    model = segmentation.deeplabv3_resnet101(pretrained=True, aux_loss=False)
    model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))

    backbone = model.backbone
    decoder = model.classifier

    return_layers = {
            'layer4': 'out',
            'layer3': 'aux'
    }
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = SegmentationNet(backbone, decoder)

    return model

class SegmentationNet(nn.Module):
    """Segmentation net wrapper for SOTA models."""
    def __init__(self, backbone, decoder):
        super(SegmentationNet, self).__init__()

        self.backbone = backbone
        self.decoder = decoder

    def forward(self, x):
        features = self.backbone(x['image'])

        x = self.decoder(features['out'])

        # Return segmentation map and aux feature map
        output = OrderedDict([
            ('out', x),
            ('aux', features['aux'])
        ])

        return output
