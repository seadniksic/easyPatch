from PIL import Image
import numpy as np
import openslide as op
import PIL
import os
PIL.Image.MAX_IMAGE_PIXELS = 71647758000

def save_to_cutout(WSI_PATH, MASK_PATH):
    im = Image.open(MASK_PATH, "r")
    arr = np.array(im)
    extended_arr = np.dstack((arr, arr))
    extended_arr = np.dstack((extended_arr, arr))
    extended_arr = np.dstack((extended_arr, arr))

    wsi_im = op.OpenSlide(WSI_PATH)
    level_1_x, level_1_y = wsi_im.level_dimensions[1]
    wsi_to_array = wsi_im.read_region((0,0), 1, (level_1_x, level_1_y))

    final_array = np.multiply(extended_arr, wsi_to_array)

    final_array = np.delete(final_array, 3, 2)
    out = Image.fromarray(final_array)
    out.save("cutout.jpeg")

    os.remove("mask_for_overlay.jpeg")

