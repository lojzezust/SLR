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

from slr.datasets.mastr import read_image_list, read_mask
from slr.pairwise_affinity import get_neighbors
from slr.datasets.utils import save_pa_sim
from mask_helper import MaskReader

MASTR_ROOT = 'data/mastr1325'
MASTR_FILE = os.path.join(MASTR_ROOT, 'all.yaml')
MASTR_ANNOTATIONS = os.path.join(MASTR_ROOT, 'weak_annotations.json')
OUTPUT_FILE = os.path.join(MASTR_ROOT, 'all_weak.yaml')
IMAGE_SIZE = (512, 384)

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

# Prior obstacle masks vars
PRIOR_MASKS_FILE = os.path.join(MASTR_ROOT, 'dextr_masks.json')
PRIOR_MASKS_OUTPUT_DIR = os.path.join(MASTR_ROOT, 'prior_instance_masks')

# Prerequisites: weak-annotations, Mastr dataset

def main():
    with open(MASTR_FILE, 'r') as file:
        mastr_paths = yaml.safe_load(file)
        img_list_path = (Path(MASTR_FILE).parent / mastr_paths['image_list']).resolve()
        images = read_image_list(img_list_path)

    # 1. Generate object masks
    generate_object_masks(images, mastr_paths)

    # 2. Compute PA similarity maps
    compute_pa_similarity_maps(images, mastr_paths)

    # 3. Generate partial masks
    generate_partial_masks(images, mastr_paths)

    # 4. Prepare prior dynamic obstacle masks
    generate_prior_masks(images, mastr_paths)

    # 5. Save file
    data_file = {
        'image_dir': mastr_paths['image_dir'],
        'image_list': mastr_paths['image_list'],
        'imu_dir': mastr_paths['imu_dir'],
        'mask_dir': os.path.relpath(PARTIAL_MASKS_OUTPUT_DIR, MASTR_ROOT),
        'object_masks_dir': os.path.relpath(OBJECT_MASK_OUTPUT_DIR, MASTR_ROOT),
        'pa_sim_dir': os.path.relpath(PA_SIM_OUTPUT_DIR, MASTR_ROOT),
        'instance_masks_dir': os.path.relpath(PRIOR_MASKS_OUTPUT_DIR, MASTR_ROOT)
    }
    with open(OUTPUT_FILE, 'w') as file:
        yaml.safe_dump(data_file, file)


def generate_object_masks(images, mastr_paths):

    with open(MASTR_ANNOTATIONS, 'r') as file:
        annotations = json.load(file)

    if not os.path.exists(OBJECT_MASK_OUTPUT_DIR):
        os.makedirs(OBJECT_MASK_OUTPUT_DIR)

    max_objects = 0
    for img_name in tqdm(images, desc="1. Generating object masks"):
        img_filename = img_name + '.jpg'
        img = np.array(Image.open(os.path.join(MASTR_ROOT, mastr_paths['image_dir'], f'{img_name}.jpg')))
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

def _compute_pa_sim(args):
    img_p, image_dir = args
    img = np.array(Image.open(os.path.join(MASTR_ROOT, image_dir, img_p)))
    img_t = torch.from_numpy(img.transpose(2,0,1).astype(np.int64)).unsqueeze(0)

    out_t = get_neighbors(img_t).squeeze(0)
    out = out_t.permute(1,2,3,0).numpy().astype(np.uint8)

    # Transform into target colorspace
    img_l = cv2.cvtColor(img, PA_SIM_COLORSPACE)[np.newaxis]
    out_l = np.stack([cv2.cvtColor(out_i, PA_SIM_COLORSPACE) for out_i in out])

    # Compute similarity
    sim = np.exp(-np.sqrt((out_l - img_l)**2).sum(axis=-1)/PA_SIM_THETA).astype(np.float32)

    filename = os.path.splitext(os.path.basename(img_p))[0] + '.npz'
    save_pa_sim(sim, os.path.join(PA_SIM_OUTPUT_DIR, filename))

def compute_pa_similarity_maps(images, mastr_paths):
    """ Generates neighbor similarity maps for pixelwise affinity loss. """

    if not os.path.exists(PA_SIM_OUTPUT_DIR):
        os.makedirs(PA_SIM_OUTPUT_DIR)

    image_dir = mastr_paths['image_dir']
    args = [('%s.jpg' % img, image_dir) for img in images]
    with Pool(PA_SIM_WORKERS) as pool:
        list(tqdm(pool.imap(_compute_pa_sim, args), total=len(images), desc="2. Computing pairwise similarity maps"))

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
        mask = mask.astype(bool)

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
        mask = mask.astype(bool)

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

    for img_name in tqdm(images, desc='3. Generating partial masks'):
        img_filename = '%s.jpg' % img_name
        imu_filename = '%s.png' % img_name
        # Update imu names according to new version of dataset
        if imu_filename.startswith('old'):
            imu_filename = imu_filename[0:3] + "_imu" + imu_filename[3:]
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


# 4. Prepare prior obstacle masks (L_aux)
def generate_prior_masks(images, mastr_paths):
    if not os.path.exists(PRIOR_MASKS_OUTPUT_DIR):
        os.makedirs(PRIOR_MASKS_OUTPUT_DIR)

    mr = MaskReader(PRIOR_MASKS_FILE)

    images = sorted(os.path.splitext(p)[0] for p in os.listdir(OBJECT_MASK_OUTPUT_DIR))
    for img in tqdm(images, desc='4. Preparing prior obstacle masks'):
        # Load object masks for the image
        obj_masks = np.load(os.path.join(OBJECT_MASK_OUTPUT_DIR, "%s.npz" % img))['arr_0']
        obj_masks = np.expand_dims(obj_masks, -1)

        # Read predicted masks for the image
        pred_masks = []
        for i in range(obj_masks.shape[0]):
            try:
                mask = mr.get_object_mask(img, i, IMAGE_SIZE[0], IMAGE_SIZE[1])
            except Exception:
                mask = None

            if mask is None:
                mask = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0]), np.uint8)
            pred_masks.append(mask)
        pred_masks = np.stack(pred_masks)
        pred_masks = np.stack([pred_masks, np.zeros_like(pred_masks), np.zeros_like(pred_masks)], -1)

        # Read partial labels
        partial_m = read_mask(os.path.join(PARTIAL_MASKS_OUTPUT_DIR, "%sm.png" % img))
        partial_m = np.expand_dims(partial_m, 0)

        # Read IMUs

        # Updated as mastrs filenames (imu) were changed
        imu_filename = img
        if img.startswith('old'):
            imu_filename = img[0:3] + "_imu" + img[3:]
        
        imu = np.array(Image.open(os.path.join(MASTR_ROOT, mastr_paths['imu_dir'], "%s.png" % imu_filename)))
        imu = np.expand_dims(imu, 0)
        imu = np.stack([np.zeros_like(imu), imu, 1-imu], -1)

        # Add water and sky segmentation (IMU) to partial labels
        partial_m_a = partial_m.sum(-1, keepdims=True)
        partial_m = partial_m + (1-partial_m_a) * imu

        # Add predicted masks to partial labels
        pred_masks_a = pred_masks.sum(-1, keepdims=True)
        instance_seg = partial_m * (1-pred_masks_a) + pred_masks
        instance_seg = obj_masks * instance_seg
        instance_seg = instance_seg.astype(np.uint8)

        out_file = os.path.join(PRIOR_MASKS_OUTPUT_DIR, '%s.npz' % img)
        np.savez_compressed(out_file, instance_seg)


if __name__=='__main__':
    main()