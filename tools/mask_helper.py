"""Helper for reading predicted prior masks encoded in RLE."""

import json
import numpy as np

def mask_to_rle(m):
    """
    # Input: 2-D numpy array
    # Output: list of numbers (1st number = #0s, 2nd number = #1s, 3rd number = #0s, ...)
    """
    # reshape mask to vector
    v = m.reshape((m.shape[0] * m.shape[1]))

    if v.size == 0:
        return [0]

    # output is empty at the beginning
    rle = []
    # index of the last different element
    last_idx = 0
    # check if first element is 1, so first element in RLE (number of zeros) must be set to 0
    if v[0] > 0:
        rle.append(0)

    # go over all elements and check if two consecutive are the same
    for i in range(1, v.size):
        if v[i] != v[i - 1]:
            rle.append(i - last_idx)
            last_idx = i

    if v.size > 0:
        # handle last element of rle
        if last_idx < v.size - 1:
            # last element is the same as one element before it - add number of these last elements
            rle.append(v.size - last_idx)
        else:
            # last element is different than one element before - add 1
            rle.append(1)

    return rle

def rle_to_mask(rle, width, height):
    """
    rle: input rle mask encoding
    each evenly-indexed element represents number of consecutive 0s
    each oddly indexed element represents number of consecutive 1s
    width and height are dimensions of the mask
    output: 2-D binary mask
    """
    # allocate list of zeros
    v = [0] * (width * height)

    # set id of the last different element to the beginning of the vector
    idx_ = 0
    for i in range(len(rle)):
        if i % 2 != 0:
            # write as many 1s as RLE says (zeros are already in the vector)
            for j in range(rle[i]):
                v[idx_ + j] = 1
        idx_ += rle[i]

    # reshape vector into 2-D mask
    # return np.reshape(np.array(v, dtype=np.uint8), (height, width)) # numba bug / not supporting np.reshape
    return np.array(v, dtype=np.uint8).reshape((height, width))

def mask_bounds(mask: np.ndarray):
    """
    mask: 2-D array with a binary mask
    output: coordinates of the top-left and bottom-right corners of the minimal axis-aligned region containing all positive pixels
    """
    ii32 = np.iinfo(np.int32)
    top = ii32.max
    bottom = ii32.min
    left = ii32.max
    right = ii32.min

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] != 0:
                top = min(top, i)
                bottom = max(bottom, i)
                left = min(left, j)
                right = max(right, j)

    return (left, top, right, bottom)


def encode_mask(mask):
    """
    mask: input binary mask, type: uint8
    output: full RLE encoding in the format: (x0, y0, w, h), RLE
    first get minimal axis-aligned region which contains all positive pixels
    extract this region from mask and calculate mask RLE within the region
    output position and size of the region, dimensions of the full mask and RLE encoding
    """
    # calculate coordinates of the top-left corner and region width and height (minimal region containing all 1s)
    x_min, y_min, x_max, y_max = mask_bounds(mask)

    # handle the case when the mask empty
    if x_min is None:
        return (0, 0, 0, 0), [0]
    else:
        tl_x = x_min
        tl_y = y_min
        region_w = x_max - x_min + 1
        region_h = y_max - y_min + 1

        # extract target region from the full mask and calculate RLE
        # do not use full mask to optimize speed and space
        target_mask = mask[tl_y:tl_y+region_h, tl_x:tl_x+region_w]
        rle = mask_to_rle(np.array(target_mask))

        return (tl_x, tl_y, region_w, region_h), rle


class MaskReader(object):
    def __init__(self, json_filepath):
        """Initialize mask helper with the path to the mask json file"""
        with open(json_filepath) as f:
            self.masks = json.load(f)

    def get_object_mask(self, frame_name, object_index, width=None, height=None):
        """
        frame_name is name of the file; example: file_name for 0001.jpg: 0001
        object_index: index of the object bounding box in weak annotations file
        width and height are width and height of the image (needed for the output)
        """
        if frame_name in self.masks and object_index < len(self.masks[frame_name]):
            mask_encoding = self.masks[frame_name][object_index]
            bounds, rle = mask_encoding['bounds'], mask_encoding['rle']
            mask = rle_to_mask(rle, bounds[2], bounds[3])
            mask_full = np.zeros((height, width), dtype=np.uint8)
            mask_full[bounds[1]:bounds[1]+bounds[3], bounds[0]:bounds[0]+bounds[2]] = mask
            return mask_full
        return None

    def get_frame_mask(self, frame_name, width=None, height=None):
        """
        Similar as get_object_mask, difference: outputs a mask for all objects in the frame
        each object has a unique label in the mask (starts with 1)
        """
        if frame_name in self.masks and len(self.masks[frame_name]) > 0:
            for i in range(len(self.masks[frame_name])):
                m = self.get_object_mask(frame_name, i, width=width, height=height)
                if i == 0:
                    mask_full = m
                else:
                    mask_full[m > 0.5] = i + 1
            return mask_full
        return None
