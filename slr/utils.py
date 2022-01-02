import argparse

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

