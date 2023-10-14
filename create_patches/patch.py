import openslide as op
import numpy as np
import math
from PIL import Image
from tqdm import tqdm
import shutil
import h5py
import time
import os
import pyvips

QUALITY = 100

class Patcher:

    def __init__(self, patch_path):
        self.patches_file = h5py.File(os.path.join(patch_path, "patches.h5"), "a")
        
    def __getitem__(self, key):
        return self.PatchGrabber(key, self.patches_file)
    
    def __del__(self):
        self.patches_file.close()
    
    def patch(self, image_path: str, patch_size: int, overlap=0, level=0):

        #create group in hdf5 file to store patches
        image_name = image_path.split(os.path.sep)[-1][:-4]
        group = self.patches_file.create_group(image_name)  

        if f"{image_name}/patches" in self.patches_file:
            return
        
        im = op.OpenSlide(image_path)

        try: 
            #get x and y dimensions of input image at specified level
            im_x_dim = im.level_dimensions[level][0]
            im_y_dim = im.level_dimensions[level][1]
        except:
            print(f"Level {level} not available for image {image_name}.  Returning.")
            return

        #calculate number of patches in x and y direction
        num_x_patchest = math.ceil(im_x_dim / patch_size) 
        num_y_patchest = math.ceil(im_y_dim / patch_size)

        num_x_patches = num_x_patchest + overlap
        num_y_patches = num_y_patchest + overlap

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
        

        patches = group.create_dataset("patches", (num_x_patches * num_y_patches, 2), "i8")
        test_patches = []
        
        patch_count = 0  #Used to keep track of patch location in patched_image array
        anchor = np.array([0,0]) #top left corner of patch
        
        for i in tqdm(range(num_y_patches)):
            for j in range(num_x_patches):
                #Read patch and assign it to array that will be returned
                # patches[patch_count] = np.array(im.read_region((*anchor,), patch_level, (patch_size, patch_size)))
                patches[patch_count] = anchor.copy()
                # print(patches[patch_count])
                # print(test_patches)

                patch_count += 1
                #shift reading frame to the right - im.level_downsamples used to upsample the shift amount since top left coord is based on level 0
                anchor[0] += round(im.level_downsamples[level]) * (patch_size - overlap_x)
                
            anchor[1] += round(im.level_downsamples[level]) * (patch_size - overlap_y)
            anchor[0] = 0

        #Set metadata
        patches.attrs["overlap_x"] = overlap_x
        patches.attrs["overlap_y"] = overlap_y
        patches.attrs["num_x_patches"] = num_x_patches
        patches.attrs["num_y_patches"] = num_y_patches
        patches.attrs["level"] = level
        patches.attrs["patch_size"] = patch_size

        print("overlap_x: {}, overlap_y: {}, patched_image_size: {}, num_x_patches: {}, num_y_patches: {}".format(overlap_x, overlap_y, len(patches), num_x_patches, num_y_patches))


    def recombine(self, image_path, output_path):

        print("Beginning Recombination")

        patches_dataset = self.patches_file[image_path.split(os.path.sep)[-1][:-4]]["patches"]
        num_x_patches = patches_dataset.attrs["num_x_patches"]
        num_y_patches = patches_dataset.attrs["num_y_patches"]
        patch_size = patches_dataset.attrs["patch_size"]
        overlap = [patches_dataset.attrs["overlap_x"], patches_dataset.attrs["overlap_y"]]

        patches = np.empty([num_x_patches * num_y_patches, patch_size, patch_size, 4], dtype=np.uint8)

        print("Extracting Patches")
        for i, patch in enumerate(self.PatchGrabber(image_path, self.patches_file)):

                patches[i] = patch

        print("Array Size" + str(patches.size * patches.itemsize))


        #If we're patching an RGB image, make sure we can avg the overlaps without overflow
        ''' ** Update, this doubles the size of the array in the register so let's not do this '''
        # if patches.dtype == np.uint8:
        #     patches = patches.astype(np.uint16)
        
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
                        t[:, :overlap[0]] = self.calc_avg_overlap(left_patch, curr_patch, overlap[0], 'x')

                    if row_num + 1 != num_y_patches:
                        #option 1, neither end y or end x
                        
                        t = t[:-overlap[1]]

                        if row_num != 0:
                            up_patch = patches[patch_num - num_x_patches]
                            t[:overlap[1]] = self.calc_avg_overlap(up_patch[:, :-overlap[0]], t, overlap[1], 'y')

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
                        t[:, :overlap[0]] = self.calc_avg_overlap(left_patch[:-overlap[1]], t, overlap[0], 'x')

                        if row_num != 0:
                            up_patch = patches[patch_num - num_x_patches]
                            t[:overlap[1]] = self.calc_avg_overlap(up_patch, t, overlap[1], 'y')

                        recon_img = np.concatenate((recon_img, t), axis=1)
                    else:
                        #option 4, end x and end y
                        t = curr_patch
                        left_patch = patches[patch_num - 1]
                        t[:, :overlap[0]] = self.calc_avg_overlap(left_patch, t, overlap[0], 'x')
                        up_patch = patches[patch_num - num_x_patches]
                        t[:overlap[1]] = self.calc_avg_overlap(up_patch, t, overlap[1], 'y')
                        recon_img = np.concatenate((recon_img, curr_patch), axis=1) #don't remove anything if final patch

                
                

            start += num_x_patches
            if row_num == 0:
                rows = recon_img
            else:
                rows = np.concatenate((rows, recon_img), axis=0)
                print("Array Size" + str(rows.size * rows.itemsize))

        print("New shape: {}".format(rows.shape))

        return rows
        

    
        
    def calc_avg_overlap(self, region1, region2, overlap, orientation):
        if orientation == 'x':
            overlap_left = region1[:, -overlap:]
            overlap_right = region2[:, :overlap]
            overlap_average = (overlap_left / 2) + (overlap_right / 2)
            return overlap_average
        elif orientation == 'y':
            overlap_up = region1[-overlap:]
            overlap_down = region2[:overlap]
            overlap_average = (overlap_up / 2) + (overlap_down / 2)
            return overlap_average
    

    class PatchGrabber:

        def __init__(self, image_path, patches_file):
            self.patches_file = patches_file
            self.image_path = image_path
            self.image = op.OpenSlide(self.image_path)
            self.max_count = self.patches_file[self.image_path.split(os.path.sep)[-1][:-4]]["patches"].shape[0]

        def __iter__(self):
            self.patch_index = 0
            return self

        def __next__(self):
            if self.patch_index == self.max_count:
                raise StopIteration
            patches_dataset = self.patches_file[self.image_path.split(os.path.sep)[-1][:-4]]["patches"]
            patch_anchor = patches_dataset[self.patch_index]
            patch_level = patches_dataset.attrs["level"]
            patch_size = patches_dataset.attrs["patch_size"]
            # print(patch_anchor)
            # print(patch_anchor)
            self.patch_index += 1
            image_array =  np.array(self.image.read_region((*list(patch_anchor),), patch_level, (patch_size, patch_size)))[]
            
            return image_array
        
        def __del__(self):
            self.patches_file.close()


    






