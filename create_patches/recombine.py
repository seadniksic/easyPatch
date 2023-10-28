from create_patches.patch import QUALITY
import numpy as np
import tqdm
import os
from PIL import Image
import matplotlib.pyplot as plt

'''
    This function takes a folder of patches and recombines them by using information recorded
    during the inital patching (metadata - overlap x, overlap y, # x patches, # y patches).  The general logic
    is that a patch can either be at the end of a row / column, or not.  This gives 4 possible
    states.  If it's at the end of a row, the right side isn't chopped off (it normally is 
    since the next patch has exactly the correct amount of overlap). Same with the 
    columns. If it's at the bottom of a column, the image's bottom isn't chopped (it 
    normally is).

'''
def recombine_from_patches_folder(PATH_TO_PATCHES, PATH_TO_METADATA):
    if PATH_TO_PATCHES[-1] != os.path.sep:
        PATH_TO_PATCHES += os.path.sep
        

    print("Reading metadata")
    with open(PATH_TO_METADATA + 'overlap.txt', "r") as f:
        overlap = []
        overlap.append(int(f.readline()))
        overlap.append(int(f.readline()))
        num_x_patches = int(f.readline())
        num_y_patches = int(f.readline())

    ext_len = (4 if len(os.listdir(PATH_TO_PATCHES)[0].split(".")[-1]) == 3 else 5) #allow for recombination of .jpeg or png patches

    sorted_patch_images = sorted(os.listdir(PATH_TO_PATCHES), key= lambda img : int(img.split('_')[-1][:-ext_len]))

    patches = []
    for img in sorted_patch_images:
        arr = np.array(Image.open(PATH_TO_PATCHES + img))
        patches.append(arr)

    patches = np.array(patches)
    
    start = 0
    print("Beginning Recombination")
    for i in range(num_y_patches):
        for patch_num in range(start, num_x_patches+start):
            patch = patches[patch_num]
            size = patch.shape

            if patch_num + 1 != num_x_patches+start:
            
                t = np.delete(patch, np.s_[size[1]-overlap[0]:size[1]], 1) #remove overlap from right side

                if i != num_y_patches - 1:
                    #option 1, neither end y or end x
                    t = np.delete(t, np.s_[size[0]-overlap[1]:size[0]], 0) #remove overlap from bottom
                else:
                    #if not option 1, option 2 is not end x but end y
                    pass
                
                #If this is the first patch, create a variable to track the total reconstruction (recon_img)
                if patch_num == start:
                    recon_img = t
                else:
                    recon_img = np.concatenate((recon_img, t), axis=1)

            else:
                if i != num_y_patches - 1:
                    #option 3, end x (not end y)

                    t = np.delete(patch, np.s_[size[0]-overlap[1]:size[0]], 0) #remove overlap from bottom
                    recon_img = np.concatenate((recon_img, t), axis=1)
                else:
                    #option 4, end x and end y
                    recon_img = np.concatenate((recon_img, patch), axis=1) #don't remove anything if final patch

        start += num_x_patches
        if i == 0:
            rows = recon_img
        else:
            rows = np.concatenate((rows, recon_img), axis=0)
    
    print("New shape: {}".format(rows.shape))
    im = Image.fromarray(rows)
    im.save(PATH_TO_METADATA + "recombination.jpeg", quality=100)

'''
    This function takes an array of patches and recombines them by using information recorded
    during the inital patching (overlap x, overlap y, # x patches, # y patches).  The general logic
    is that a patch can either be at the end of a row / column, or not.  This gives 4 possible
    states.  If it's at the end of a row, the right side isn't chopped off (it normally is 
    since the next patch has exactly the correct amount of overlap). Same with the 
    columns. If it's at the bottom of a column, the image's bottom isn't chopped (it 
    normally is).

'''

def recombine_from_patches_array(patches, overlap, num_x_patches, num_y_patches):
    start = 0
    print("Beginning Recombination")
    for i in range(num_y_patches):
        for patch_num in range(start, num_x_patches+start):
            patch = patches[patch_num]
            size = patch.shape

            if patch_num + 1 != num_x_patches+start:
            
                t = np.delete(patch, np.s_[size[1]-overlap[0]:size[1]], 1) #remove overlap from right side

                if i != num_y_patches - 1:
                    #option 1, neither end y or end x
                    t = np.delete(t, np.s_[size[0]-overlap[1]:size[0]], 0) #remove overlap from bottom
                else:
                    #if not option 1, option 2 is not end x but end y
                    pass
                
                #If this is the first patch in a row , create a variable to track the total reconstruction (recon_img)
                if patch_num == start:
                    recon_img = t
                else:
                    recon_img = np.concatenate((recon_img, t), axis=1)

            else:
                if i != num_y_patches - 1:
                    #option 3, end x (not end y)

                    t = np.delete(patch, np.s_[size[0]-overlap[1]:size[0]], 0) #remove overlap from bottom
                    recon_img = np.concatenate((recon_img, t), axis=1)
                else:
                    #option 4, end x and end y
                    recon_img = np.concatenate((recon_img, patch), axis=1) #don't remove anything if final patch

        start += num_x_patches
        if i == 0:
            rows = recon_img
        else:
            rows = np.concatenate((rows, recon_img), axis=0)
    
    print("New shape: {}".format(rows.shape))
    return rows
    # im.save(PATH_TO_METADATA + "recombination.jpeg", quality=100)

'''
    This function takes a folder of patches (intended to be pixel-wise numerical class predications)
    and recombines them by using information recorded during the inital patching 
    (metadata - overlap x, overlap y, # x patches, # y patches).  The general logic is that a patch can either
    be at the end of a row / column, or not.  This gives 4 possible states.  If it's not at the end
    of a row or column, the overlap between patches is averaged.

'''

def recombine_from_patches_folder_avg_overlap(PATH_TO_PATCHES, PATH_TO_METADATA):
    if PATH_TO_PATCHES[-1] != os.path.sep:
        PATH_TO_PATCHES += os.path.sep

    if PATH_TO_METADATA[-1] != os.path.sep:
        PATH_TO_METADATA += os.path.sep
        

    print("Reading metadata")
    with open(PATH_TO_METADATA + 'overlap.txt', "r") as f:
        overlap = []
        overlap.append(int(f.readline()))
        overlap.append(int(f.readline()))
        num_x_patches = int(f.readline())
        num_y_patches = int(f.readline())
        print(overlap)

    sorted_patch_images = sorted(os.listdir(PATH_TO_PATCHES), key= lambda img : int(img.split("_")[-1][:-5]))

    patches = []
    for img in sorted_patch_images:
        arr = np.array(Image.open(PATH_TO_PATCHES + img), dtype=np.uint16)
        patches.append(arr)

    patches = np.array(patches)
    
    start = 0
    print("Beginning Recombination")
    recon_img = patches[0][:-overlap[1], :-overlap[0]] #Set first patch in place

    for row_num in range(num_y_patches):
        for patch_num in range(start, num_x_patches+start):

            if patch_num == 0 and row_num == 0: #Already set the first patch
                continue
            
            curr_patch = patches[patch_num]
            
            if patch_num + 1 != num_x_patches + start: 

                t = curr_patch[:, :-overlap[0]]

                if patch_num  % num_x_patches != 0:
                    left_patch = patches[patch_num - 1]
                    t[:, :overlap[0]] = calc_avg_overlap(left_patch, curr_patch, overlap[0], 'x')

                if row_num + 1 != num_y_patches:
                    #option 1, neither end y or end x
                    
                    t = t[:-overlap[1]]

                    if row_num != 0:
                        up_patch = patches[patch_num - num_x_patches]
                        t[:overlap[1]] = calc_avg_overlap(up_patch[:, :-overlap[0]], t, overlap[1], 'y')

                else:
                    #if not option 1, option 2 is not end x but end y
                    pass

                if patch_num == start:
                    recon_img = t
                else:   
                    recon_img = np.concatenate((recon_img, t), axis=1)

            else:
                if row_num + 1 != num_y_patches:
                    #option 3, end x (not end y)
                    t = curr_patch[:-overlap[1]] #remove overlap from bottom
                    left_patch = patches[patch_num - 1]
                    t[:, :overlap[0]] = calc_avg_overlap(left_patch[:-overlap[1]], t, overlap[0], 'x')

                    if row_num != 0:
                        up_patch = patches[patch_num - num_x_patches]
                        t[:overlap[1]] = calc_avg_overlap(up_patch, t, overlap[1], 'y')

                    recon_img = np.concatenate((recon_img, t), axis=1)
                else:
                    #option 4, end x and end y
                    t = curr_patch
                    left_patch = patches[patch_num - 1]
                    t[:, :overlap[0]] = calc_avg_overlap(left_patch, t, overlap[0], 'x')
                    up_patch = patches[patch_num - num_x_patches]
                    t[:overlap[1]] = calc_avg_overlap(up_patch, t, overlap[1], 'y')
                    recon_img = np.concatenate((recon_img, curr_patch), axis=1) #don't remove anything if final patch

        start += num_x_patches
        if row_num == 0:
            rows = recon_img
        else:
            rows = np.concatenate((rows, recon_img), axis=0)


    
    print("New shape: {}".format(rows.shape))
    im = Image.fromarray(np.array(rows, dtype=np.uint8))
    im.save(PATH_TO_METADATA + "recombination.jpeg", quality=100)

'''
    This function takes an array of patches (intended to be pixel-wise numerical class predications)
    and recombines them by using information recorded during the inital patching 
    (overlap x, overlap y, # x patches, # y patches).  The general logic is that a patch can either
    be at the end of a row / column, or not.  This gives 4 possible states.  If it's not at the end
    of a row or column, the overlap between patches is averaged.

'''

def recombine_from_patches_array_avg_overlap(patches, overlap, num_x_patches, num_y_patches):
    
    #If we're patching an RGB image, make sure we can avg the overlaps without overflow
    if patches.dtype == np.uint8:
        patches = patches.astype(np.uint16)
    
    start = 0
    print("Beginning Recombination")
    recon_img = patches[0][:-overlap[1], :-overlap[0]] #Set first patch in place

    for row_num in range(num_y_patches):
        for patch_num in range(start, num_x_patches+start):

            if patch_num == 0 and row_num == 0: #Already set the first patch
                continue
            
            curr_patch = patches[patch_num]
            
            if patch_num + 1 != num_x_patches + start: 

                t = curr_patch[:, :-overlap[0]]

                if patch_num  % num_x_patches != 0:
                    left_patch = patches[patch_num - 1]
                    t[:, :overlap[0]] = calc_avg_overlap(left_patch, curr_patch, overlap[0], 'x')

                if row_num + 1 != num_y_patches:
                    #option 1, neither end y or end x
                    
                    t = t[:-overlap[1]]

                    if row_num != 0:
                        up_patch = patches[patch_num - num_x_patches]
                        t[:overlap[1]] = calc_avg_overlap(up_patch[:, :-overlap[0]], t, overlap[1], 'y')

                else:
                    #if not option 1, option 2 is not end x but end y
                    pass

                if patch_num == start:
                    recon_img = t
                else:   
                    recon_img = np.concatenate((recon_img, t), axis=1)

            else:
                if row_num + 1 != num_y_patches:
                    #option 3, end x (not end y)
                    t = curr_patch[:-overlap[1]] #remove overlap from bottom
                    left_patch = patches[patch_num - 1]
                    t[:, :overlap[0]] = calc_avg_overlap(left_patch[:-overlap[1]], t, overlap[0], 'x')

                    if row_num != 0:
                        up_patch = patches[patch_num - num_x_patches]
                        t[:overlap[1]] = calc_avg_overlap(up_patch, t, overlap[1], 'y')

                    recon_img = np.concatenate((recon_img, t), axis=1)
                else:
                    #option 4, end x and end y
                    t = curr_patch
                    left_patch = patches[patch_num - 1]
                    t[:, :overlap[0]] = calc_avg_overlap(left_patch, t, overlap[0], 'x')
                    up_patch = patches[patch_num - num_x_patches]
                    t[:overlap[1]] = calc_avg_overlap(up_patch, t, overlap[1], 'y')
                    recon_img = np.concatenate((recon_img, curr_patch), axis=1) #don't remove anything if final patch

        start += num_x_patches
        if row_num == 0:
            rows = recon_img
        else:
            rows = np.concatenate((rows, recon_img), axis=0)

    print("New shape: {}".format(rows.shape))
    return rows


def recombine_avg_overlap_refactored(patches, overlap, num_x_patches, num_y_patches):
    '''
    This function loops through all patches in a row, averaging and concatenating.
    Then, it proceeds to the next row, and concatenates
    '''
     #If we're patching an RGB image, make sure we can avg the overlaps without overflow
    if patches.dtype == np.uint8:
        patches = patches.astype(np.uint16)

    start = 0
    print("Beginning Recombination")
    recon_img = patches[0][:-overlap[1], :-overlap[0]] #Set first patch in place

    for row_num in range(num_y_patches):
        for patch_num in range(start, num_x_patches+start):

            if patch_num == 0 and row_num == 0: #Already set the first patch
                continue
            
            curr_patch = patches[patch_num]
            
            if patch_num + 1 != num_x_patches + start: 

                t = curr_patch[:, :-overlap[0]]

                if patch_num  % num_x_patches != 0:
                    left_patch = patches[patch_num - 1]
                    t[:, :overlap[0]] = calc_avg_overlap(left_patch, curr_patch, overlap[0], 'x')

                if row_num + 1 != num_y_patches:
                    #option 1, neither end y or end x
                    
                   t = t[:-overlap[1]]

                    if row_num != 0:
                        up_patch = patches[patch_num - num_x_patches]
                        t[:overlap[1]] = calc_avg_overlap(up_patch[:, :-overlap[0]], t, overlap[1], 'y')

                else:
                    #if not option 1, option 2 is not end x but end y
                    pass

                if patch_num == start:
                    recon_img = t
                else:   
                    recon_img = np.concatenate((recon_img, t), axis=1)

            else:
                if row_num + 1 != num_y_patches:
                    #option 3, end x (not end y)
                    t = curr_patch[:-overlap[1]] #remove overlap from bottom
                    left_patch = patches[patch_num - 1]
                    t[:, :overlap[0]] = calc_avg_overlap(left_patch[:-overlap[1]], t, overlap[0], 'x')

                    if row_num != 0:
                        up_patch = patches[patch_num - num_x_patches]
                        t[:overlap[1]] = calc_avg_overlap(up_patch, t, overlap[1], 'y')

                    recon_img = np.concatenate((recon_img, t), axis=1)
                else:
                    #option 4, end x and end y
                    t = curr_patch
                    left_patch = patches[patch_num - 1]
                    t[:, :overlap[0]] = calc_avg_overlap(left_patch, t, overlap[0], 'x')
                    up_patch = patches[patch_num - num_x_patches]
                    t[:overlap[1]] = calc_avg_overlap(up_patch, t, overlap[1], 'y')
                    recon_img = np.concatenate((recon_img, curr_patch), axis=1) #don't remove anything if final patch

        start += num_x_patches
        if row_num == 0:
            rows = recon_img
        else:
            rows = np.concatenate((rows, recon_img), axis=0)

    print("New shape: {}".format(rows.shape))
    return rows













    
def calc_avg_overlap(region1, region2, overlap, orientation):
    if orientation == 'x':
        overlap_left = region1[:, -overlap:]
        overlap_right = region2[:, :overlap]
        overlap_average = np.add(overlap_left, overlap_right) / 2
        return overlap_average
    elif orientation == 'y':
        overlap_up = region1[-overlap:]
        overlap_down = region2[:overlap]
        overlap_average = np.add(overlap_up, overlap_down) / 2
        return overlap_average
