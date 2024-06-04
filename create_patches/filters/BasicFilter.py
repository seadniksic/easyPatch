import numpy as np
import openslide as op


class BasicFilter:
    def __init__(self, mask_path):
        self.annotation_mask = op.OpenSlide(mask_path)
        self.accept_threshold = 10

    def filter(self, image, current_patch, anchor, level, patch_size):
        mask_patch = np.array(self.annotation_mask.read_region((*list(anchor),), level, (patch_size, patch_size)))[:,:,:3]

        if np.mean(mask_patch) > self.accept_threshold:
            current_patch = current_patch.astype(np.float32) / 1.5
            current_patch[:,:,2] += ((255 - current_patch[:, :, 2]) / 2)
            return True, current_patch.astype(np.uint8)
        else:
            return False, current_patch

