import numpy as np
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import PIL

__all__ = ["pairwise_affinity_loss", "object_pairwise_affinity_loss"]

def get_neighbor_kernel():
    """Constructs a convolutional kernel that extracts 8 neighboring pixels into 8 channels."""
    kernels = []
    for i in range(9):
        if i == 4:
            continue
        k = np.zeros(9)
        k[i] = 1
        k = k.reshape(3,3)
        kernels.append(k)

    kernels = np.stack(kernels).astype(np.int64)
    kernels_t = torch.from_numpy(kernels).unsqueeze(1)

    return kernels_t

def get_neighbors(input, dilation=2):
    """Get the 8 neighbor maps for the input map. """

    # Construct neighbor kernel
    kernels_t = get_neighbor_kernel()
    kernels_t = kernels_t.type(input.type())

    bs,c,h,w = input.shape
    out_t = torch.nn.functional.conv2d(input.view(bs*c, 1, h, w), kernels_t, dilation=dilation, padding=dilation)
    out_t = out_t.view(bs, c, 8, h, w)

    return out_t

def pairwise_affinity_loss(logits, labels, pa_similarity, tau=0.1, dilation=2, target_scale='labels', ignore_mask=None):

    eps = 1.e-9

    if target_scale == 'logits':
        # Resize one-hot labels (and pa sim) to match the logits scale
        logits_size = (logits.size(2), logits.size(3))
        labels = F.interpolate(labels, size=logits_size, mode='area')
        pa_similarity = F.interpolate(pa_similarity, size=logits_size, mode='area')
    elif target_scale == 'labels':
        # Resize network output to match the label (and pa sim) size
        labels_size = (labels.size(2), labels.size(3))
        logits = TF.resize(logits, labels_size, interpolation=PIL.Image.BILINEAR)
    else:
        raise ValueError('Invalid value for target_scale: %s' % target_scale)

    logits_sm = torch.softmax(logits, 1)
    logits_sm_neighbors = get_neighbors(logits_sm, dilation=dilation)
    logits_sm = logits_sm.unsqueeze(2)
    pa_masks = (pa_similarity > tau).float()
    if ignore_mask is not None:
        # Ignore edges within ignore mask (i.e. objects)
        pa_masks = pa_masks * (1. - ignore_mask).unsqueeze(1)

    # Predicted prob that neighbors are in the same class (eq. 4)
    p_y = (logits_sm_neighbors * logits_sm).sum(1)

    # Cross entropy for positive edges only (eq. 8)
    pa_loss = -(pa_masks * torch.log(p_y + eps)).sum() / pa_masks.sum()

    return pa_loss

def object_pairwise_affinity_loss(logits, object_masks, n_objects, pa_similarity, tau=0.1, dilation=2, normalize=False, reduce='mean', target_scale='labels'):
    eps = 1.e-9

    object_masks = object_masks.float()
    if target_scale == 'logits':
        # Resize object masks to match the logits scale
        logits_size = (logits.size(2), logits.size(3))
        object_masks = F.interpolate(object_masks, size=logits_size, mode='area')
        pa_similarity = F.interpolate(pa_similarity, size=logits_size, mode='area')
    elif target_scale == 'labels':
        # Resize network output to match the object masks size
        labels_size = (object_masks.size(2), object_masks.size(3))
        logits = TF.resize(logits, labels_size, interpolation=PIL.Image.BILINEAR)
    else:
        raise ValueError('Invalid value for target_scale: %s' % target_scale)


    logits_sm = torch.softmax(logits, 1)
    logits_sm_neighbors = get_neighbors(logits_sm, dilation=dilation)
    logits_sm = logits_sm.unsqueeze(2)
    pa_masks = (pa_similarity > tau).float()

    # Predicted prob that neighbors are in the same class (eq. 4)
    p_y = (logits_sm_neighbors * logits_sm).sum(1)

    # Cross entropy for positive edges only (eq. 8), average over 8 neighbors
    pa_ce = -(pa_masks * torch.log(p_y + eps)).sum(1, keepdim=True)

    # Normalize by number of positive edges
    n_pos = object_masks * pa_masks.sum(1, keepdims=True)
    pa_loss_obj = (pa_ce * object_masks).sum((2,3)) / (n_pos.sum((2,3)) + eps)

    # Normalize by object size
    if normalize:
        pa_loss_obj = pa_loss_obj * object_masks.mean((2,3))

    # Sum or average over all obstacles
    object_mask_1d = (object_masks.sum((2,3)) > 0).float()
    if reduce == 'sum':
        loss = (object_mask_1d * pa_loss_obj).sum()
    elif reduce == 'mean':
        loss = (object_mask_1d * pa_loss_obj).sum() / (n_objects.sum() + eps)
    else:
        raise ValueError("Invalide reduce option.")

    return loss
