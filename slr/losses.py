import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import PIL

def water_obstacle_separation_loss(features, gt_mask, include_sky=False):
    """Computes the water-obstacle separation loss from intermediate features.

    Args:
        features (torch.tensor): Features tensor
        gt_mask (torch.tensor): Ground truth tensor
        include_sky (bool): Include sky into separation loss
    """
    epsilon_watercost = 0.01
    min_samples = 5

    # Resize gt mask to match the extracted features shape (x,y)
    feature_size = (features.size(2), features.size(3))
    gt_mask = F.interpolate(gt_mask, size=feature_size, mode='area')

    # Create water and obstacles masks.
    # The masks should be of type float so we can multiply it later in order to mask the elements
    # (1 = water, 2 = sky, 0 = obstacles)
    if include_sky:
        mask_water = (gt_mask[:,1] + gt_mask[:,2]).unsqueeze(1)
    else:
        mask_water = gt_mask[:,1].unsqueeze(1)

    mask_obstacles = gt_mask[:,0].unsqueeze(1)

    # Count number of water and obstacle pixels, clamp to at least 1 (for numerical stability)
    elements_water = mask_water.sum((0,2,3), keepdim=True).clamp(min=1.)
    elements_obstacles = mask_obstacles.sum((0,2,3), keepdim=True)

    # Zero loss if number of samples for any class is smaller than min_samples
    if elements_obstacles.squeeze() < min_samples or elements_water.squeeze() < min_samples:
        return torch.tensor(0.)

    # Only keep water and obstacle pixels. Set the rest to 0.
    water_pixels = mask_water * features
    obstacle_pixels = mask_obstacles * features

    # Mean value of water pixels per feature (batch average)
    mean_water = water_pixels.sum((0,2,3), keepdim=True) / elements_water

    # Mean water value matrices for water and obstacle pixels
    mean_water_wat = mean_water * mask_water
    mean_water_obs = mean_water * mask_obstacles

    # Variance of water pixels (per channel, batch average)
    var_water = (water_pixels - mean_water_wat).pow(2).sum((0,2,3), keepdim=True) / elements_water

    # Average quare difference of obstacle pixels and mean water values (per channel)
    difference_obs_wat = (obstacle_pixels - mean_water_obs).pow(2).sum((0,2,3), keepdim=True)

    # Compute the separation
    loss_c = elements_obstacles * var_water / (difference_obs_wat + epsilon_watercost)

    return loss_c.mean()

def focal_loss(logits, labels, gamma=2.0, alpha=4.0, target_scale='labels'):
    """Focal loss of the segmentation output `logits` and ground truth `labels`."""

    epsilon = 1.e-9

    if target_scale == 'logits':
        # Resize one-hot labels to match the logits scale
        logits_size = (logits.size(2), logits.size(3))
        labels = F.interpolate(labels, size=logits_size, mode='area')
    elif target_scale == 'labels':
        # Resize network output to match the label size
        labels_size = (labels.size(2), labels.size(3))
        logits = TF.resize(logits, labels_size, interpolation=PIL.Image.BILINEAR)
    else:
        raise ValueError('Invalid value for target_scale: %s' % target_scale)

    logits_sm = torch.softmax(logits, 1)

    # Focal loss
    fl = -labels * torch.log(logits_sm + epsilon) * (1. - logits_sm) ** gamma
    fl = fl.sum(1) # Sum focal loss along channel dimension

    # Return mean of the focal loss along spatial and batch dimensions
    return fl.mean()

def object_focal_loss(logits, object_masks, n_objects, instance_masks, gamma=2.0, normalize=False, reduce='mean', target_scale='labels'):
    eps = 1.e-9

    object_masks = object_masks.float()
    instance_masks = instance_masks.float()
    if target_scale == 'logits':
        # Resize one-hot labels to match the logits scale
        logits_size = (logits.size(2), logits.size(3))
        object_masks = F.interpolate(object_masks, size=logits_size, mode='area')
        instance_masks = F.interpolate(instance_masks, size=logits_size, mode='area')
    elif target_scale == 'labels':
        # Resize network output to match the label size
        labels_size = (object_masks.size(2), object_masks.size(3))
        logits = TF.resize(logits, labels_size, interpolation=PIL.Image.BILINEAR)
    else:
        raise ValueError('Invalid value for target_scale: %s' % target_scale)

    logits_sm = torch.softmax(logits, 1).unsqueeze(1)

    # Focal loss
    fl = -instance_masks * torch.log(logits_sm + eps) * (1. - logits_sm) ** gamma
    fl = fl.sum(2) # Sum focal loss along channel dimension
    fl = (fl * object_masks).sum((2,3)) / (object_masks.sum((2,3)) + eps) # Average loss over object positions

    # Normalize by object size
    if normalize:
        fl = fl * object_masks.mean((2,3))

    # Sum or average over all obstacles in the batch
    object_mask_1d = (object_masks.sum((2,3)) > 0).float()
    if reduce == 'sum':
        loss = (object_mask_1d * fl).sum()
    elif reduce == 'mean':
        loss = (object_mask_1d * fl).sum() / (n_objects.sum() + eps)
    else:
        raise ValueError("Invalide reduce option.")

    return loss

def object_projection_loss(logits, object_masks, n_objects, target_scale='labels', normalize=False, reduce='mean'):
    epsilon = 1.e-9

    object_masks = object_masks.float()
    if target_scale == 'logits':
        # Resize one-hot labels to match the logits scale
        logits_size = (logits.size(2), logits.size(3))
        object_masks = F.interpolate(object_masks, size=logits_size, mode='area')
    elif target_scale == 'labels':
        # Resize network output to match the label size
        labels_size = (object_masks.size(2), object_masks.size(3))
        logits = TF.resize(logits, labels_size, interpolation=PIL.Image.BILINEAR)
    else:
        raise ValueError('Invalid value for target_scale: %s' % target_scale)

    logits_sm = torch.softmax(logits, 1)
    p_obst = logits_sm[:,0].unsqueeze(1)


    p_obst_inst = p_obst * object_masks
    alpha = 8.0
    pproj_y = (p_obst_inst.multiply(alpha).softmax(2) * p_obst_inst).sum(2)
    pproj_x = (p_obst_inst.multiply(alpha).softmax(3) * p_obst_inst).sum(3)

    lproj_y = object_masks.max(2).values
    lproj_x = object_masks.max(3).values


    # Dice loss for each axis
    loss_x = 1 - 2 * (pproj_x * lproj_x).sum(2) / (pproj_x.pow(2).sum(2) + lproj_x.pow(2).sum(2) + epsilon)
    loss_y = 1 - 2 * (pproj_y * lproj_y).sum(2) / (pproj_y.pow(2).sum(2) + lproj_y.pow(2).sum(2) + epsilon)

    obj_loss = loss_x + loss_y

    if normalize:
        obj_loss = obj_loss * object_masks.mean((2,3))

    # Sum or average over all obstacles
    object_mask_1d = (object_masks.sum((2,3)) > 0).float()
    if reduce == 'sum':
        loss = (object_mask_1d * obj_loss).sum()
    elif reduce == 'mean':
        loss = (object_mask_1d * obj_loss).sum() / (n_objects.sum() + epsilon)
    else:
        raise ValueError("Invalide reduce option.")

    return loss
