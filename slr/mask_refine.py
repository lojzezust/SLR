import torch
import torch.nn.functional as F
import numpy as np

def mask_refine(weak_mask, predictions, features, imu, objects, per_instance=True, fill_weight=1.0, iterations=1, unconstrained=False, entropy_limit=1.0):
        """Refines partial masks by filling in empty regions. Feature prototypes are computed from logits, then
        pixels in the mask are reasigned values according to the distances to the feature prototypes.

        Args:
            weak_mask (torch.Tensor): weakly annotated masks to be filled (batch)
            predictions (torch.Tensor): predictions (probabilities) used to compute the prototypes
            features (torch.Tensor): feature tensor used in prototypes
            imu (torch.Tensor): IMU horizon mask
            objects (torch.Tensor): object masks
            per_instance (bool, optional): Whether to perform the prototype computation per-instance.
                                           Otherwise per-batch is used. Defaults to True.
            fill_weight (float, optional): The weight of filled labels (0-1). Defaults to 1.0.
            iterations (int, optional): Number of iterations
            unconstrained (bool, optional): Ignore IMU and weak_mask data in computation (unconstrained).
            entropy_limit (float, optional): Maximum normalized entropy of filled masks. Uncertain regions are ignored.

        Returns:
            torch.Tensor: filled masks
        """
        neg_inf = torch.tensor(-np.inf, device=predictions.device)
        eps = 1e-9

        mask_w = weak_mask.sum(1, keepdim=True)
        objects_m = objects.max(1,keepdim=True).values * (1.0 - mask_w)

        # Resize
        mask_sm = F.interpolate(weak_mask, size=(features.size(2), features.size(3)), mode='area')
        objects_sm = F.interpolate(objects.float(), size=(features.size(2), features.size(3)), mode='area')
        imu_sm = F.interpolate(imu.unsqueeze(1).float(), size=(features.size(2), features.size(3)), mode='area').squeeze(1)
        object_mask = objects_sm.max(1).values

        # Upper and lower imu masks (with slight overlap)
        imu_sm_up = 1 - F.pad(imu_sm, (0,0,1,0))[:,:-1]
        imu_sm_down = 1 - F.pad(1-imu_sm, (0,0,0,1))[:,1:]

        for i in range(iterations):
            # Constraint: Only use predictions in parts that are uncertain
            if not unconstrained:
                predictions = (1.0-mask_w) * predictions + weak_mask

            # Resize segmentation predictions
            predictions_sm = F.interpolate(predictions, size=(features.size(2), features.size(3)), mode='area')

            # Class masks
            classes = predictions_sm.argmax(1)
            obstacle_m = predictions_sm[:,0].unsqueeze(1)
            land_m = obstacle_m * (1-object_mask.unsqueeze(1))
            water_m = predictions_sm[:,1].unsqueeze(1)
            sky_m = predictions_sm[:,2].unsqueeze(1)

            # Constraint: Use IMU to clean predictions for water and sky
            if not unconstrained:
                water_m = water_m * imu_sm_down.unsqueeze(1)
                sky_m = sky_m * imu_sm_up.unsqueeze(1)

            # Compute class feature prototypes
            sumdim = (2,3) if per_instance else (1,2,3)
            water_mean = (features * water_m).sum(sumdim, keepdim=True) / water_m.sum(sumdim, keepdim=True)
            land_mean = (features * land_m).sum(sumdim, keepdim=True) / land_m.sum(sumdim, keepdim=True)
            sky_mean = (features * sky_m).sum(sumdim, keepdim=True) / sky_m.sum(sumdim, keepdim=True)

            # Cosine distance to prototypes
            water_logits = (features * water_mean).sum(1) / (features.pow(2).sum(1).sqrt() * water_mean.pow(2).sum(1).sqrt())
            land_logits = (features * land_mean).sum(1) / (features.pow(2).sum(1).sqrt() * land_mean.pow(2).sum(1).sqrt())
            sky_logits = (features * sky_mean).sum(1) / (features.pow(2).sum(1).sqrt() * sky_mean.pow(2).sum(1).sqrt())

            # Constraint: Use IMU to clean logits for water and sky
            if not unconstrained:
                water_logits = torch.where(imu_sm_down > 0, water_logits, neg_inf)
                sky_logits = torch.where(imu_sm_up > 0, sky_logits, neg_inf)

            # Compute per object prototypes
            features_obj = features.unsqueeze(2)
            pos_objects_sm = objects_sm * obstacle_m
            object_f = features_obj * pos_objects_sm.unsqueeze(1)
            object_means = object_f.sum((3,4), keepdim=True) / (pos_objects_sm.unsqueeze(1).sum((3,4), keepdim=True) + 1e-8)

            # Cosine distance to object prototypes
            object_logits = (features_obj * object_means).sum(1) / (features_obj.pow(2).sum(1).sqrt() * object_means.pow(2).sum(1).sqrt() + 1e-8)
            object_logits_m = object_logits.max(1).values # If objects overlap, use most probable

            # Combined obstacle logits (land + objects)
            obstacle_logits_c = land_logits * (1-object_mask) + object_logits_m * object_mask
            generated_mask_sm = torch.stack([obstacle_logits_c, water_logits, sky_logits], dim=1).multiply(20).softmax(1)
            generated_mask = F.interpolate(generated_mask_sm, size=(imu.size(1), imu.size(2)), mode='bilinear')

            # Limit uncertainty
            entropy = -(generated_mask * generated_mask.add(eps).log() / np.log(3)).sum(1, keepdim=True)
            generated_mask = generated_mask * (entropy < entropy_limit)

            # Fill missing data with generated masks
            mask_filled = mask_w * weak_mask + (1 - mask_w) * generated_mask * fill_weight

            # New predictions are filled masks
            predictions = mask_filled

        return mask_filled
