import torch
import argparse
from pathlib import Path
from pytorch_lightning.utilities.distributed import rank_zero_only

# Argparse type that can parse None
def Option(type):
    def _type(value):
        if value.lower() == 'none':
            return None

        return type(value)
    return _type

def bool_arg(v):
    """Generalized bool argument for argparse."""
    if isinstance(v, bool):
        return v
    if str(v).lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif str(v).lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


@rank_zero_only
def mkdir_safe(dir):
    dir = Path(dir)
    if not dir.exists():
        dir.mkdir(parents=True)


def load_weights(weight_path):
    """Loads weights from weight file or from training checkpoint."""

    state_dict = torch.load(weight_path, map_location='cpu')
    if 'model' in state_dict:
        return state_dict['model']
    else:
        return state_dict
