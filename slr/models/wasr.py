import torch
from torch import nn
from torchvision.models.resnet import resnet101, resnet50
from torch.hub import load_state_dict_from_url
import torchvision.transforms.functional as TF

from collections import OrderedDict
from PIL import Image

from .utils import IntermediateLayerGetter

model_urls = {
    'deeplabv3_resnet101_coco': 'https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth'
}

def wasr_resnet101(num_classes=3, pretrained=False, imu=False, norelu_aux=False):
    # Pretrained ResNet101 backbone
    backbone = resnet101(pretrained=True, replace_stride_with_dilation=[False, True, True])
    return_layers = {
        'layer4': 'out',
        'layer1': 'skip1',
        'layer2': 'skip2',
        'layer3': 'aux'
    }
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    if imu:
        decoder = IMUDecoder()
    else:
        decoder = NoIMUDecoder()

    model = WaSR(backbone, decoder, imu=imu)

    # Load pretrained DeeplabV3 weights (COCO)
    if pretrained:
        model_url = model_urls['deeplabv3_resnet101_coco']
        state_dict = load_state_dict_from_url(model_url, progress=True)

        # Only load backbone weights, since decoder is entirely different
        keys_to_remove = [key for key in state_dict.keys() if not key.startswith('backbone.')]
        for key in keys_to_remove: del state_dict[key]

        model.load_state_dict(state_dict, strict=False)

    return model


def wasr_resnet50(num_classes=3, pretrained=False, imu=False, norelu_aux=False):
    # Pretrained ResNet101 backbone
    backbone = resnet50(pretrained=True, replace_stride_with_dilation=[False, True, True])

    return_layers = {
        'layer4': 'out',
        'layer1': 'skip1',
        'layer2': 'skip2',
        'layer3': 'aux'
    }
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    if imu:
        decoder = IMUDecoder()
    else:
        decoder = NoIMUDecoder()

    model = WaSR(backbone, decoder, imu=imu)

    return model


class WaSR(nn.Module):
    """
    Implements WaSR model from
    `"A water-obstacle separation and refinement network for unmanned surface vehicles"
    <https://arxiv.org/abs/2001.01921>`_.
    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the following keys:
            - "out": last feature map of the backbone (2048 features)
            - "aux": feature map used for the auxiliary separation loss (1024 features)
            - "skip1": high-resolution feature map (skip connection) used in FFM (256 features)
            - "skip2": low-resolution feature map (skip connection) used in ARM2 (512 features)
        decoder (nn.Module): a WaSR decoder module. Takes the backbone outputs (with skip connections)
            and returns a dense segmentation prediction for the classes
        classifier_input_features (int, optional): number of input features required by classifier
    """
    def __init__(self, backbone, decoder, imu=False):
        super(WaSR, self).__init__()

        self.imu = imu

        self.backbone = backbone
        self.decoder = decoder

    def forward(self, x):
        features = self.backbone(x['image'])

        if self.imu:
            features['imu_mask'] = x['imu_mask']

        x = self.decoder(features)

        # Return segmentation map and aux feature map
        output = OrderedDict([
            ('out', x),
            ('aux', features['aux'])
        ])

        return output

class AttentionRefinementModule(nn.Module):
    def __init__(self, in_channels, last_arm=False):
        super(AttentionRefinementModule, self).__init__()

        self.last_arm = last_arm

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, in_channels, 1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x

        x = self.global_pool(x)
        x = self.conv1(x)
        x = self.bn1(x)
        weights = self.sigmoid(x)

        out = weights * input

        if self.last_arm:
            weights = self.global_pool(out)
            out = weights * out

        return out


class FeatureFusionModule(nn.Module):
    def __init__(self, bg_channels, sm_channels, num_features):
        super(FeatureFusionModule, self).__init__()

        self.upsampling = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv1 = nn.Conv2d(bg_channels + sm_channels, num_features, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features)
        self.relu = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv2 = nn.Conv2d(num_features, num_features, 1)
        self.conv3 = nn.Conv2d(num_features, num_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_big, x_small):
        if x_big.size(2) > x_small.size(2):
            x_small = self.upsampling(x_small)

        x = torch.cat((x_big, x_small), 1)
        x = self.conv1(x)
        x = self.bn1(x)
        conv1_out = self.relu(x)

        x = self.global_pool(conv1_out)
        x = self.conv2(x)
        x = self.conv3(x)
        weights = self.sigmoid(x)

        mul = weights * conv1_out
        out = conv1_out + mul

        return out

class ASPPv2Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation, bias=False, bn=False, relu=False):
        modules = []
        modules.append(nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=bias))

        if bn:
            modules.append(nn.BatchNorm2d(out_channels))

        if relu:
            modules.append(nn.ReLU())

        super(ASPPv2Conv, self).__init__(*modules)

class ASPPv2(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256, relu=False, biased=True):
        super(ASPPv2, self).__init__()
        modules = []

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPv2Conv(in_channels, out_channels, rate, bias=True))

        self.convs = nn.ModuleList(modules)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))

        # Sum convolution results
        res = torch.stack(res).sum(0)
        return res

class BottleneckLogits(nn.Module):
    """Bottleneck module wrapper that enables logit output.
    Wraps a `torchvision.models.resnet.Bottleneck` module.

    Args:
        bottleneck: the bottleneck module to wrap
    """

    def __init__(self, bottleneck):
        super(BottleneckLogits, self).__init__()

        self.conv1 = bottleneck.conv1
        self.bn1 = bottleneck.bn1
        self.conv2 = bottleneck.conv2
        self.bn2 = bottleneck.bn2
        self.conv3 = bottleneck.conv3
        self.bn3 = bottleneck.bn3
        self.relu = bottleneck.relu
        self.downsample = bottleneck.downsample
        self.stride = bottleneck.stride

        # New logits module
        self.relu_n = nn.ReLU()
        self.logits = nn.Identity()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        # Pass through logits layer
        out = self.logits(out)
        out = self.relu_n(out)

        return out


class NoIMUDecoder(nn.Module):
    def __init__(self):
        super(NoIMUDecoder, self).__init__()

        self.arm1 = AttentionRefinementModule(2048)
        self.arm2 = nn.Sequential(
            AttentionRefinementModule(512, last_arm=True),
            nn.Conv2d(512, 2048, 1) # Equalize number of features with ARM1
        )

        self.ffm = FeatureFusionModule(256, 2048, 1024)
        self.aspp = ASPPv2(1024, [6, 12, 18, 24], 3)

    def forward(self, x):
        features = x

        arm1 = self.arm1(features['out'])
        arm2 = self.arm2(features['skip2'])
        arm_combined = arm1 + arm2

        x = self.ffm(features['skip1'], arm_combined)

        output = self.aspp(x)

        return output

class IMUDecoder(nn.Module):
    def __init__(self):
        super(IMUDecoder, self).__init__()

        self.arm1 = AttentionRefinementModule(2048 + 1)
        self.aspp1 = ASPPv2(2048, [6, 12, 18], 32)
        self.ffm1 = FeatureFusionModule(2048 + 1, 32, 1024)

        self.arm2 = nn.Sequential(
            AttentionRefinementModule(512 + 1, last_arm=True),
            nn.Conv2d(512 + 1, 1024, 1, bias=False) # Equalize number of features with FFM1
        )

        self.ffm = FeatureFusionModule(256 + 1, 1024, 1024)
        self.aspp = ASPPv2(1024, [6, 12, 18, 24], 3)

    def forward(self, x):
        features = x

        # Resize IMU mask to two required scales
        out = features['out']
        skip1 = features['skip1']
        imu_mask = features['imu_mask'].float().unsqueeze(1)
        imu_mask_s1 = TF.resize(imu_mask, (out.size(2), out.size(3)), Image.NEAREST)
        imu_mask_s0 = TF.resize(imu_mask, (skip1.size(2), skip1.size(3)), Image.NEAREST)

        # Concat backbone output and IMU
        out_imu = torch.cat([out, imu_mask_s1], dim=1)
        arm1 = self.arm1(out_imu)

        aspp1 = self.aspp1(out)

        # Fuse ARM1 and ASPP1
        ffm1 = self.ffm1(arm1, aspp1)

        # Concat Skip 2 and IMU
        skip2_imu = torch.cat([features['skip2'], imu_mask_s1], dim=1)
        arm2 = self.arm2(skip2_imu)

        arm_combined = ffm1 + arm2

        # Concat Skip 1 and IMU
        skip1_imu = torch.cat([features['skip1'], imu_mask_s0], dim=1)
        x = self.ffm(skip1_imu, arm_combined)

        output = self.aspp(x)

        return output
