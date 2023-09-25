import openslide as op
import numpy as np
import math
from PIL import Image
from tqdm import tqdm
import shutil
import h5py
import time
import os

QUALITY = 100

'''

    This function creates patches of a wsi and its corresponding mask, and stores those patches (and metadata) 
    in the following directory structure:

            image patches:   ./create_patches/patches/image
            mask patches:    ./create_patches/patches/mask
            metadata:        ./create_patches/patches

'''

def patch_image_and_mask(WSI_NAME, WSI_PATH, LEVEL, PATCH_SIZE):
    now = time.time()
    patched_image, patched_mask, metadata = patch_with_mask(WSI_NAME, WSI_PATH, LEVEL, PATCH_SIZE)

    #If there is an existing folder for patches of some ID, remove everything (older patches)

    if not os.path.exists("create_patches/patches"):
        os.makedirs("create_patches/patches")
    
    if WSI_NAME in os.listdir("create_patches/patches"):
        print(f'deleting {WSI_NAME}')
        shutil.rmtree("create_patches/patches/{}/".format(WSI_NAME))

    os.makedirs("create_patches/patches/{}".format(WSI_NAME))
    os.makedirs("create_patches/patches/{}/image".format(WSI_NAME))
    os.makedirs("create_patches/patches/{}/mask".format(WSI_NAME))
    
    for i in range(len(patched_image)):
        Image.fromarray(patched_image[i]).save("create_patches/patches/{}/image/{}_{}.jpeg".format(WSI_NAME, WSI_NAME, i), QUALITY=QUALITY, subsampling=0)
        Image.fromarray(patched_mask[i]).save("create_patches/patches/{}/mask/{}_{}.png".format(WSI_NAME, WSI_NAME, i), QUALITY=QUALITY, subsampling=0)
    with open("create_patches/patches/{}/overlap.txt".format(WSI_NAME), 'w') as f:        
        f.write("\n".join([str(dim) for dim in metadata]))
    print("Duration: " + str(time.time() - now))

'''

    This function creates patches of a wsi, and stores those patches (and metadata) 
    in the following directory structure:

            image patches:   ./create_patches/patches/image
            metadata:        ./create_patches/patches

'''

def patch_images_without_mask(INPUT_DIR, OUTPUT_DIR, LEVEL, PATCH_SIZE, overlap):

    output_file = h5py.File(os.path.join(OUTPUT_DIR, "patches.hdf5"), "w")

    for img_name in os.listdir(INPUT_DIR):
        patch_without_mask(os.path.join(INPUT_DIR, img_name), output_file, LEVEL, PATCH_SIZE, overlap)

    output_file.close()


'''

    This function creates patches of a wsi, and stores those patches (and metadata) 
    in the following directory structure:

            image patches:   ./create_patches/patches/image
            metadata:        ./create_patches/patches

'''


def patch_image_without_mask(WSI_NAME, WSI_PATH, LEVEL, PATCH_SIZE, overlap):
    now = time.time()

    if not os.path.exists(os.path.join("create_patches", "patches.hdf5")):
        f = h5py.File(os.path.join("create_patches", "patches.hdf5"), "w")
    else: 
        f = h5py.File(os.path.join("create_patches", "patches.hdf5"), "r")

    patch_without_mask(os.path.join(WSI_PATH, f'{WSI_NAME}.svs'), f, LEVEL, PATCH_SIZE, overlap)


    #If there is an existing folder for patches of some ID, remove everything (older patches)
    # if WSI_NAME in os.listdir("create_patches/patches"):
    #     print(f'deleting {WSI_NAME}')
    #     shutil.rmtree("create_patches/patches/{}/".format(WSI_NAME))

    # os.makedirs("create_patches/patches/{}".format(WSI_NAME))
    # os.makedirs("create_patches/patches/{}/image".format(WSI_NAME))
    
    for i in range(len(patched_image)):
        Image.fromarray(patched_image[i][:,:,:3]).save("create_patches/patches/{}/image/{}_{}.jpeg".format(WSI_NAME, WSI_NAME, i), QUALITY=QUALITY, subsampling=0)
    with open("create_patches/patches/{}/overlap.txt".format(WSI_NAME), 'w') as f:        
        f.write("\n".join([str(dim) for dim in metadata]))
    print("Duration: " + str(time.time() - now))


'''
    Main patching logic function, patches image and mask
'''
def patch_with_mask(image_name, image, patch_level, patch_size):     
    print("{}{}.svs".format(image, image_name))   
    im = op.OpenSlide("{}{}.svs".format(image, image_name))
    mask = np.array(Image.open("create_mask/masks/{}_mask.png".format(image_name)))
    
    #get x and y dimensions of input image at specified level
    print(patch_level)
    print(im.level_dimensions)
    print(im.level_dimensions[patch_level])
    im_x_dim = im.level_dimensions[patch_level][0]
    im_y_dim = im.level_dimensions[patch_level][1]

    #calculate number of patches in x and y direction
    num_x_patches = math.ceil(im_x_dim / patch_size)
    print("x: {}".format(num_x_patches))
    num_y_patches = math.ceil(im_y_dim / patch_size)
    print("y: {}".format(num_y_patches))
    
    #Find overlap necessary to make patches fit - if overlap is a decimal, rounding up will cut off some pixels at the edge of the image.  Cest la vi
    overlap_x = math.ceil((patch_size * num_x_patches - im_x_dim)/ (num_x_patches - 1))
    overlap_y = math.ceil((patch_size * num_y_patches - im_y_dim)/ (num_y_patches - 1))

    #This will be return tensor (4 at the end since openslide returns 4 channels)
    patched_image = np.empty([num_x_patches * num_y_patches, patch_size, patch_size, 3], dtype=np.uint8)
    patched_mask = np.empty([num_x_patches * num_y_patches, patch_size, patch_size], dtype=np.uint8)
    
    patch_count = 0  #Used to keep track of patch location in patched_image array
    anchor = [0,0] #top left corner of patch
    
    for i in range(num_y_patches):
        for j in range(num_x_patches):
            #Read patch and assign it to array that will be returned
            patched_image[patch_count] = np.array(im.read_region((*anchor,), patch_level, (patch_size, patch_size)))[:,:,:3]
            level_anchor = [int(anchor[0] / 4**patch_level), int(anchor[1] / 4**patch_level)]
            patched_mask[patch_count] = mask[level_anchor[1]:level_anchor[1] + patch_size, level_anchor[0]:level_anchor[0] + patch_size]
            patch_count += 1

            #shift reading frame to the right - im.level_downsamples used to upsample the shift amount since top left coord is based on level 0
            anchor[0] += round(im.level_downsamples[patch_level]) * (patch_size - overlap_x)
        
        anchor[1] += round(im.level_downsamples[patch_level]) * (patch_size - overlap_y)
        anchor[0] = 0

    print("overlap_x: {}, overlap_y: {}, patched_image_size: {}, num_x_patches: {}, num_y_patches: {}".format(overlap_x, overlap_y, len(patched_image), num_x_patches, num_y_patches))
        
    return patched_image, patched_mask, [overlap_x, overlap_y, num_x_patches, num_y_patches] #, im.get_thumbnail(im.level_dimensions[3]) 


'''
    Secondary patching logic function, only patches image (not mask)
'''
def patch_without_mask(image, output_file, patch_level, patch_size, overlap_level):     

    #create group in hdf5 file to store patches
    image_name = image.split(os.path.sep)[-1][:-4]
    group = output_file.create_group(image_name)  
    
    im = op.OpenSlide(image)

    try: 
        #get x and y dimensions of input image at specified level
        im_x_dim = im.level_dimensions[patch_level][0]
        im_y_dim = im.level_dimensions[patch_level][1]
    except:
        print(f"Level {patch_level} not available for image {image_name}.  Returning.")
        return

    #calculate number of patches in x and y direction
    num_x_patchest = math.ceil(im_x_dim / patch_size) 
    num_y_patchest = math.ceil(im_y_dim / patch_size)

    num_x_patches = num_x_patchest + overlap_level
    num_y_patches = num_y_patchest + overlap_level

    #Find overlap necessary to make patches fit
    overlap_x = math.ceil((patch_size * num_x_patches - im_x_dim)/ (num_x_patches - 1))
    overlap_y = math.ceil((patch_size * num_y_patches - im_y_dim)/ (num_y_patches - 1))

    #Ensure overlap in either direction does not cross the max threshold (1/2 patch size)
    max_overlap = int(patch_size / 2)
    if overlap_x > max_overlap or overlap_y > max_overlap:
        if overlap_x > max_overlap:
            print(f'Overlap (X = {overlap_x}) Exceeds Max Value of {max_overlap}')
            while overlap_x > max_overlap:
                num_x_patches -= 1
                overlap_x = math.ceil((patch_size * num_x_patches - im_x_dim)/ (num_x_patches - 1))
        if overlap_y > max_overlap:
            print(f'Overlap (Y = {overlap_y}) Exceeds Max Value of {max_overlap}')
            while overlap_y > max_overlap:
                num_y_patches -= 1
                overlap_y = math.ceil((patch_size * num_y_patches - im_y_dim)/ (num_y_patches - 1))
    

    patches = group.create_dataset("patches", (num_x_patches * num_y_patches, patch_size, patch_size, 4), "i8")
    
    patch_count = 0  #Used to keep track of patch location in patched_image array
    anchor = [0,0] #top left corner of patch
    
    for i in tqdm(range(num_y_patches)):
        for j in range(num_x_patches):
            #Read patch and assign it to array that will be returned
            patches[patch_count] = np.array(im.read_region((*anchor,), patch_level, (patch_size, patch_size)))              
            patch_count += 1
            #shift reading frame to the right - im.level_downsamples used to upsample the shift amount since top left coord is based on level 0
            anchor[0] += round(im.level_downsamples[patch_level]) * (patch_size - overlap_x)
            
        anchor[1] += round(im.level_downsamples[patch_level]) * (patch_size - overlap_y)
        anchor[0] = 0

    #Set metadata
    patches.attrs["overlap_x"] = overlap_x
    patches.attrs["overlap_y"] = overlap_y
    patches.attrs["num_x_patches"] = num_x_patches
    patches.attrs["num_y_patches"] = num_y_patches

    print("overlap_x: {}, overlap_y: {}, patched_image_size: {}, num_x_patches: {}, num_y_patches: {}".format(overlap_x, overlap_y, len(patches), num_x_patches, num_y_patches))
        
    return
