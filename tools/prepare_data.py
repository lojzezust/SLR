import os
import cv2
import numpy as np
import json
from PIL import Image
import torch
from tqdm.auto import tqdm
import yaml
from pathlib import Path
from multiprocessing import Pool

from slr.datasets.mastr import read_image_list
from slr.pairwise_affinity import get_neighbors
from slr.datasets.utils import save_pa_sim

MASTR_ROOT = 'data/mastr1325'
MASTR_FILE = os.path.join(MASTR_ROOT, 'all.yaml')
MASTR_ANNOTATIONS = os.path.join(MASTR_ROOT, 'weak_annotations.json')
OUTPUT_FILE = os.path.join(MASTR_ROOT, 'all_weak.yaml')

# Object masks vars
MANUAL_OBJECT_MASKS_DIR = os.path.join(MASTR_ROOT, 'manual_objects')
OBJECT_MASK_OUTPUT_DIR = os.path.join(MASTR_ROOT, 'objects')

# PA similarity vars
PA_SIM_COLORSPACE = cv2.COLOR_RGB2LAB # Colorspace used to compute PA similarity maps
PA_SIM_THETA = 2
PA_SIM_WORKERS = 32
PA_SIM_OUTPUT_DIR = os.path.join(MASTR_ROOT, 'pa_similarity')

# Partial masks vars
PARTIAL_MASKS_OUTPUT_DIR = os.path.join(MASTR_ROOT, 'masks_weak')
PARTIAL_MASKS_WE_BETA = 0.5 # TODO

# Prerequisites: weak-annotations, Mastr dataset

def main():
    with open(MASTR_FILE, 'r') as file:
        mastr_paths = yaml.safe_load(file)
        img_list_path = (Path(MASTR_FILE).parent / mastr_paths['image_list']).resolve()
        images = read_image_list(img_list_path)

    # 1. Generate object masks
    generate_object_masks(images)

    # 2. Compute PA similarity maps
    compute_pa_similarity_maps(images)

    # 3. Generate partial masks
    generate_partial_masks(images, mastr_paths)


def generate_object_masks(images):

    with open(MASTR_ANNOTATIONS, 'r') as file:
        annotations = json.load(file)

    if not os.path.exists(OBJECT_MASK_OUTPUT_DIR):
        os.makedirs(OBJECT_MASK_OUTPUT_DIR)

    max_objects = 0
    for img_name in tqdm(images, desc="Generating object masks"):
        img_filename = img_name + '.jpg'
        img = np.array(Image.open(os.path.join(MASTR_ROOT, data['image_dir'], f'{img_name}.jpg')))
        ann = annotations[img_filename]

        # Load manual object masks
        obj_mask_path = os.path.join(MANUAL_OBJECT_MASKS_DIR, img_name)
        if os.path.exists(obj_mask_path):
            obj_mask = np.array(Image.open(obj_mask_path))
            # Convert RGB channel values to int
            obj_mask = np.sum(obj_mask * np.array([1, 256, 256**2]), 2)
        else:
            obj_mask = np.zeros((img.shape[0],img.shape[1]), dtype=np.int64)

        H,W,_ = img.shape

        masks = []
        # Bounding boxes
        for i,obj in enumerate(ann['objects']):
            mask = np.zeros((H,W), np.uint8)
            x,y,w,h = obj
            mask[y:y+h, x:x+w] = 1
            masks.append(mask)

        # Manually labeled object masks
        object_vals = np.unique(obj_mask)
        object_vals = object_vals[object_vals>0]
        for object_val in object_vals:
            mask = (obj_mask == object_val).astype(np.uint8)
            masks.append(mask)

        max_objects = max(len(masks), max_objects)
        if len(masks) == 0:
            masks.append(np.zeros((H,W), np.uint8))
        masks = np.stack(masks)


        filename = img_name + '.npz'
        out_path = os.path.join(OBJECT_MASK_OUTPUT_DIR, filename)
        np.savez_compressed(out_path, masks)

    print(max_objects)

def _compute_pa_sim(img_p):
    img = np.array(Image.open(os.path.join(MASTR_ROOT, img_p)))
    img_t = torch.from_numpy(img.transpose(2,0,1).astype(np.int64)).unsqueeze(0)

    out_t = get_neighbors(img_t).squeeze(0)
    out = out_t.permute(1,2,3,0).numpy().astype(np.uint8)

    # Transform into target colorspace
    img_l = cv2.cvtColor(img, PA_SIM_COLORSPACE)[np.newaxis]
    out_l = np.stack([cv2.cvtColor(out_i, PA_SIM_COLORSPACE) for out_i in out])

    # Compute similarity
    sim = np.exp(-np.sqrt((out_l - img_l)**2).sum(axis=-1)/PA_SIM_THETA).astype(np.float)

    filename = os.path.splitext(os.path.basename(img_p))[0] + '.npz'
    save_pa_sim(sim, os.path.join(PA_SIM_OUTPUT_DIR, filename))

def compute_pa_similarity_maps(images):
    """ Generates neighbor similarity maps for pixelwise affinity loss. """

    if not os.path.exists(PA_SIM_OUTPUT_DIR):
        os.makedirs(PA_SIM_OUTPUT_DIR)

    with Pool(PA_SIM_WORKERS) as pool:
        list(tqdm(pool.imap(_compute_pa_sim, images), total=len(images), desc="Computing pairwise similarity maps"))

# 3. Generate partial masks
def _horizon_to_vertical(points, ny, imu):
    delta = 50
    h,w = imu.shape
    height = imu.argmax(0)

    results = []
    for px in points:
        if px==0 or px==w-1:
            results.append([px, ny])
            continue

        py = height[px]
        x0, x1 = max(px-delta, 2), min(px+delta, w-3)
        kx = x1 - x0
        ky = height[x1] - height[x0]
        dx = -ky
        dy = kx
        nx = int(round(px + (ny - py)/dy * dx))

        results.append([nx, ny])

    return results

def generate_mask(imu, annotations, obj_mask):
    """Generates partial mask for a single image."""

    # Prepare masks
    water_p = imu.astype(np.float32)
    water_p = cv2.GaussianBlur(water_p, (11,51), 0)
    sky_p = 1 - water_p
    land_p = np.zeros_like(water_p)
    weights_m = np.ones_like(water_p)

    # Water edges
    for we in annotations['water_edges']:
        we_slack_u = 2
        we_slack_d = 1
        we_l = np.array(we)
        # Find endpoints
        x0u = we_l[0][0]
        x1u = we_l[-1][0]

        imin = we_l[:,0].argmin()
        imax = we_l[:,0].argmax()

        x0d = we_l[imin][0]
        x1d = we_l[imax][0]

        # Compute vertical points from horizon
        (p0u,p1u) = _horizon_to_vertical([x0u, x1u], 0, imu)
        we_u = np.array([p0u] + we + [p1u])
        (p0d,p1d) = _horizon_to_vertical([x0d, x1d], imu.shape[0], imu)
        we_d = np.array([p0d] + we[imin:imax+1] + [p1d])

        # Generate upper and lower and water_edge masks
        u_mask = np.zeros_like(imu)
        u_mask = cv2.fillPoly(u_mask, [we_u], 1)
        u_mask = cv2.polylines(u_mask, [we_u], 1, 1, we_slack_u)

        d_mask = np.zeros_like(imu)
        d_mask = cv2.fillPoly(d_mask, [we_d], 1)
        d_mask = cv2.polylines(d_mask, [we_d], 1, 1, we_slack_d)

        c_mask = u_mask | d_mask

        we_mask = np.zeros_like(imu)
        we_mask = cv2.polylines(we_mask, [we_l], 0, 1)

        # Distance from we
        we_dist = cv2.distanceTransform(1-we_mask, cv2.DIST_L2, cv2.DIST_MASK_3)

        # Update masks
        land_p = u_mask + (1 - u_mask) * land_p
        sky_p = (1 - c_mask) * sky_p
        water_p = d_mask + (1 - c_mask) * water_p
        sure_land = u_mask & imu

        # Water edge falloff
        weights_m = u_mask * np.exp(-PARTIAL_MASKS_WE_BETA * we_dist) + (1 - u_mask) * weights_m
        # Sure land
        weights_m = (1-sure_land) * weights_m + sure_land
        # Water edge uncertainty
        weights_m = (1 - (u_mask & d_mask)) * weights_m

    # Bounding box objects
    for obj in annotations['objects']:
        x,y,w,h = obj

        mask = np.zeros_like(land_p, np.uint8)
        mask[y:y+h, x:x+w] = 1

        # Only ignore water or sky parts of the mask
        mask = mask * np.logical_or(water_p, sky_p)
        mask = mask.astype(np.bool)

        # Add objects to masks
        land_p[mask] = 1
        sky_p[mask] = 0
        water_p[mask] = 0
        weights_m[mask] = 0

    # Manually labeled object masks
    object_vals = np.unique(obj_mask)
    object_vals = object_vals[object_vals>0]
    for object_val in object_vals:
        mask = (obj_mask == object_val).astype(np.uint8)

        # Only ignore water or sky parts of the mask
        mask = mask * np.logical_or(water_p, sky_p)
        mask = mask.astype(np.bool)

        # Add objects to masks
        land_p[mask] = 1
        sky_p[mask] = 0
        water_p[mask] = 0
        weights_m[mask] = 0

    # Combine into an RGB image
    comb_m = np.stack([land_p, water_p, sky_p], axis=-1)
    comb_m = comb_m / comb_m.sum(axis=-1, keepdims=True) # Normalize masks so they sum to 1
    comb_w = comb_m * weights_m[..., np.newaxis]

    return comb_w


def generate_partial_masks(images, mastr_paths):
    with open(MASTR_ANNOTATIONS, 'r') as file:
        annotations = json.load(file)

    if not os.path.exists(PARTIAL_MASKS_OUTPUT_DIR):
        os.makedirs(PARTIAL_MASKS_OUTPUT_DIR)

    for img_name in tqdm(images):
        img_filename = '%s.jpg' % img_name
        imu_filename = '%s.png' % img_name
        imu = np.array(Image.open(os.path.join(MASTR_ROOT, mastr_paths['imu_dir'], imu_filename)))
        ann = annotations[img_filename]

        obj_mask_path = os.path.join(MANUAL_OBJECT_MASKS_DIR, imu_filename)
        if os.path.exists(obj_mask_path):
            obj_mask = np.array(Image.open(obj_mask_path))
            # Convert RGB channel values to int
            obj_mask = np.sum(obj_mask * np.array([1, 256, 256**2]), 2)
        else:
            obj_mask = np.zeros_like(imu, dtype=np.int64)

        mask = generate_mask(imu, ann, obj_mask)
        # Convert to image
        mask = (mask*255).astype(np.uint8)

        out_filename = '%sm.png' % img_name
        out_path = os.path.join(PARTIAL_MASKS_OUTPUT_DIR, out_filename)

        Image.fromarray(mask).save(out_path)


if __name__=='__main__':
    main()
