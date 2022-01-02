from .wasr import wasr_resnet101, wasr_resnet50
from .other import deeplabv3_resnet101

models = ['wasr_resnet101_imu', 'wasr_resnet101', 'wasr_resnet50_imu', 'wasr_resnet50', 'deeplabv3_resnet101']

def get_model(architecture, num_classes=3, pretrained=True):
    if architecture not in models:
        raise ValueError('Unknown model architecture: %s' % architecture)

    imu = False
    if architecture.endswith('imu'):
        imu = True

    model = None
    if architecture.startswith('wasr_resnet101'):
        model = wasr_resnet101(num_classes=num_classes, pretrained=pretrained, imu=imu)
    elif architecture.startswith('wasr_resnet50'):
        model = wasr_resnet50(num_classes=num_classes, pretrained=pretrained, imu=imu)
    elif architecture == 'deeplabv3_resnet101':
        model = deeplabv3_resnet101(num_classes=num_classes, pretrained=pretrained)

    if model is None:
        raise ValueError('Error initializing model: %s' % architecture)

    return model
