import albumentations as A
import torchvision.transforms as T
import numpy as np
import cv2


def get_augmentation_transform():
    color_transform = A.Compose([
        A.ColorJitter(p=0.7, hue=0.05),
        A.RandomGamma(p=1, gamma_limit=(70,120))], p=0.5)

    noise_transform = A.Compose([
        A.GaussNoise(p=0.5),
        A.ISONoise(p=0.5)], p=0.3)

    transform = A.Compose([
        A.HorizontalFlip(),
        A.ShiftScaleRotate(scale_limit=[0,0.3], rotate_limit=15, border_mode=0, p=0.7),
        color_transform,
        noise_transform
    ])

    return AlbumentationsTransform(transform)

class AlbumentationsTransform(object):
    def __init__(self, transform, image_feature='image', mask_features=['segmentation', 'imu_mask', 'objects', 'pa_similarity', 'instance_seg']):
        self.transform = transform
        self.image_feature = image_feature
        self.mask_features = mask_features

    def __call__(self, x):
        valid_mask_features = [feat for feat in self.mask_features if feat in x]
        masks = [x[feat] for feat in valid_mask_features]

        res = self.transform(image=x[self.image_feature], masks=masks)

        output = {}
        output[self.image_feature] = res['image']
        for feat, mask in zip(valid_mask_features, res['masks']):
            output[feat] = mask
        for feat in x:
            if feat not in output:
                output[feat] = x[feat]

        return output


def PytorchHubNormalization():
    """Transform that normalizes the image to pytorch hub models (DeepLab, ResNet,...) expected range.
    See: https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/"""

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    transform = T.Compose([
        T.ToTensor(), # CHW order, divide by 255
        T.Normalize(mean, std)
    ])

    return transform

class LabelSmoothing(object):
    """Applies label smoothing to segmentation labels."""

    def __init__(self, alpha=0.1, n_cls=3):
        self.alpha = alpha
        self.n_cls = n_cls

    def __call__(self, x):
        mask = x['segmentation']
        mask_ls = mask * (1-self.alpha) + self.alpha / self.n_cls

        res = {k:x[k] for k in x}
        res['segmentation'] = mask_ls
        return res

class GaussLabelSmoothing(object):
    """Applies spatially varying label smoothing (gauss blur) to segmentation labels."""

    def __init__(self, sigma=1):
        self.sigma = sigma

    def __call__(self, x):
        mask = x['segmentation']
        mask_ls = cv2.GaussianBlur(mask, (0,0), self.sigma)

        res = {k:x[k] for k in x}
        res['segmentation'] = mask_ls
        return res
