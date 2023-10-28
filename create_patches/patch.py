import openslide as op
import numpy as np
import math
from PIL import Image
from tqdm import tqdm
import h5py
import os

class Patcher:

    def __init__(self, patch_path, use_hdf5=True):

        self.use_hdf5 = use_hdf5
        self.patch_path = patch_path

        if use_hdf5:
            try: 
                self.patches_file = h5py.File(os.path.join(patch_path, "patches.h5"), "w")
            except:
                self.patches_file = h5py.File(os.path.join(patch_path, "patches.h5"), "r")
                print(f"Patches file {os.path.join(patch_path, 'patches.h5')} already exists")
        else:
            self.patches_file = os.path.join(patch_path, 'patches')

        #create PatchHandler Object as abstraction for handling how / where to save patches
        self.patch_handler = self.PatchHandler(self.patches_file, use_hdf5)
        

    def __getitem__(self, key):

        image_name = key.split(os.path.sep)[-1][:-4]
        if image_name not in self.patches_file:
            print("No patches exist for the specified image")
        else:
            return self.PatchHandler(self.patch_path, self.use_hdf5).initiate_iterator(key)
    

    def __del__(self):
        if self.use_hdf5:
            self.patches_file.close()
    

    def patch(self, image_path: str, patch_size: int, overlap=0, level=0, mask_path: str=None, return_patches: bool=True):

        self.patch_handler.return_patches = return_patches

        #create group in hdf5 file to store patches
        image_name = image_path.split(os.path.sep)[-1][:-4]

        if self.patch_handler.patches_exist(image_name):
            del self.patches_file[image_name]

        self.patch_handler.make_group(image_name)

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
        
        self.patch_handler.create_dataset("patches", num_x_patches, num_y_patches, patch_size, "i8", return_patches, image_name)
        
        patch_count = 0  #Used to keep track of patch location in patched_image array
        anchor = np.array([0,0]) #top left corner of patch
        
        for i in tqdm(range(num_y_patches)):
            for j in range(num_x_patches):
                
                #Read patch and assign it to array that will be returned
                self.patch_handler.store_patch(im, patch_count, anchor, level, patch_size, image_name, return_patches)

                patch_count += 1
                #shift reading frame to the right - im.level_downsamples used to upsample the shift amount since top left coord is based on level 0
                anchor[0] += round(im.level_downsamples[level]) * (patch_size - overlap_x)
                
            anchor[1] += round(im.level_downsamples[level]) * (patch_size - overlap_y)
            anchor[0] = 0

        self.patch_handler.write_metadata(image_name, "patches", overlap_x, overlap_y, num_x_patches, num_y_patches, level, patch_size)

        if return_patches:
            return self.patch_handler.grab_patches()

    def recombine(self, image_path: str):

        name = image_path.split(os.path.sep)[-1][:-4]

        num_x_patches, num_y_patches, patch_size, patch_level, overlap = self.patch_handler.load_metadata(image_path)

        print(f"Extracting Patches from {name} ")
        
        patches = self.patch_handler.load_patches(self, image_path)
        
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
                # print("Array Size In Memory" + str(rows.size * rows.itemsize))

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
    

    class PatchIterator:
        def __init__(self, parent, image, patches_file, max_count, patch_size, patch_level, image_name):
            self.parent = parent
            self.patches_file = patches_file
            self.image = image
            self.image_name = image_name
            self.max_count = max_count
            self.patch_size = patch_size
            self.patch_level = patch_level

        def __iter__(self):
            self.patch_index = 0
            return self

        def __next__(self):
            if self.patch_index == self.max_count:
                raise StopIteration
            patch = self.parent.patch_handler.get_patch(self.image, self.patches_file[self.patch_index], self.patch_index, self.image_name, self.patch_size, self.patch_level)
            self.patch_index += 1
            return patch

    class PatchHandler:

        def __init__(self, patches_file, use_hdf5):

            self.patches_file = patches_file
         
            self.use_hdf5 = use_hdf5

        def grab_patches(self):
            temp_patch_holder = self.patched_image
            self.patched_image = np.array([])

            #return the refrerence to the array in memory
            return temp_patch_holder

            
        def patches_exist(self, group_name):

            if self.use_hdf5:
                return (True if group_name in self.patches_file else False)
            
            # TODO finish if not using hdf5

        def make_group(self, group_name):

            if self.use_hdf5:
                self.patches_file.create_group(group_name)
            else: 
                if not os.path.exists(os.path.join(self.patches_file, group_name)):
                    os.makedirs(os.path.join(self.patches_file, group_name))

        def create_dataset(self, name, num_x_patches, num_y_patches, patch_size, type, return_patches, image_name):

            if self.use_hdf5:
                self.patches_file[image_name].create_dataset(name, (num_x_patches * num_y_patches, 2), type)
            if return_patches:
                self.patched_image = np.empty([num_x_patches * num_y_patches, patch_size, patch_size, 3], dtype=np.uint8)

        def store_patch(self, image, patch_index, anchor, level, patch_size, image_name, return_patches):
        
            if self.use_hdf5:
                
                self.patches_file[image_name]["patches"][patch_index] = anchor.copy()

                if return_patches:
                    temp_image = image.read_region((*list(anchor),), level, (patch_size, patch_size))
                    self.patched_image[patch_index] = np.array(temp_image)[:,:,:3] #Just grab RGB not A

            else:

                if return_patches:
                    temp_image = image.read_region((*list(anchor),), level, (patch_size, patch_size))
                    self.patched_image[patch_index] = np.array(temp_image)[:,:,:3] #Just grab RGB not A

                temp_image.save(os.path.join(self.patches_file, image_name, "patches", f"{patch_index}.png"))

        def get_patch(self, image, patches_file, patch_index, image_name, patch_size, patch_level):

            if self.use_hdf5:
                anchor = patches_file[image_name]["patches"][patch_index]
                return np.array(image.read_region((*list(anchor),), patch_level, (patch_size, patch_size)))[:,:,:3] #Just grab RGB not A
            else:
                return np.array(Image.open(patches_file))[:,:,:3] 
                
        def write_metadata(self, image, dataset_name, overlap_x, overlap_y, num_x_patches, num_y_patches, level, patch_size):

            if self.use_hdf5:

                dataset = self.patches_file[image][dataset_name]

                dataset.attrs["overlap_x"] = overlap_x         
                dataset.attrs["overlap_x"] = overlap_x
                dataset.attrs["num_x_patches"] = num_x_patches
                dataset.attrs["num_y_patches"] = num_y_patches
                dataset.attrs["level"] = level
                dataset.attrs["overlap_y"] = overlap_y
                dataset.attrs["patch_size"] = patch_size

            else:
                metadata = [overlap_x, overlap_y, num_x_patches, num_y_patches, level, patch_size]
                with open(f"{self.patches_file}/{image}/overlap.txt", 'w') as f:        
                    f.write("\n".join([str(dim) for dim in metadata]))

        def load_metadata(self, image_path):

            image_name = image_path.split(os.path.sep)[-1][:-4]

            if self.use_hdf5:
                
                patches_dataset = self.patches_file[image_name]["patches"]
                num_x_patches = patches_dataset.attrs["num_x_patches"]
                num_y_patches = patches_dataset.attrs["num_y_patches"]
                patch_size = patches_dataset.attrs["patch_size"]
                patch_level = patches_dataset.attrs["level"]
                overlap = [patches_dataset.attrs["overlap_x"], patches_dataset.attrs["overlap_y"]]             

            else:

                with open(os.path.join(self.patches_file, image_path.split(os.path.sep)[-1][:-4], 'overlap.txt'), "r") as f:
                    overlap = []
                    overlap.append(int(f.readline()))
                    overlap.append(int(f.readline()))
                    num_x_patches = int(f.readline())
                    num_y_patches = int(f.readline())
                    patch_level = int(f.readline())
                    patch_size = int(f.readline())

            return num_x_patches, num_y_patches, patch_size, patch_level, overlap


        def load_patches(self, parent, image_path):

            image_name = image_path.split(os.path.sep)[-1][:-4]
            image_handler = op.OpenSlide(image_path)

            if self.use_hdf5:
                
                patches_dataset = self.patches_file[image_name]["patches"]
                num_x_patches = patches_dataset.attrs["num_x_patches"]
                num_y_patches = patches_dataset.attrs["num_y_patches"]
                patch_size = patches_dataset.attrs["patch_size"]
                patch_level = patches_dataset.attrs["level"]
                overlap = [patches_dataset.attrs["overlap_x"], patches_dataset.attrs["overlap_y"]]

                max_count = self.patches_file[image_path.split(os.path.sep)[-1][:-4]]["patches"].shape[0]             

            else:

                with open(os.path.join(self.patches_file, image_name, 'overlap.txt'), "r") as f:
                    overlap = []
                    overlap.append(int(f.readline()))
                    overlap.append(int(f.readline()))
                    num_x_patches = int(f.readline())
                    num_y_patches = int(f.readline())
                    patch_level = int(f.readline())
                    patch_size = int(f.readline())

                ext_len = (4 if len(os.listdir(os.path.join(self.patches_file, image_name, "patches"))[0].split(".")[-1]) == 3 else 5) #allow for recombination of .jpeg or png patches
                self.patches_file = [os.path.join(self.patches_file, image_name, "patches", img) for img in sorted(os.listdir(os.path.join(self.patches_file, image_name, "patches")), key= lambda img : int(img.split('_')[-1][:-ext_len]))]

                max_count = len(self.patches_file)

            patches = np.empty([num_x_patches * num_y_patches, patch_size, patch_size, 3], dtype=np.uint8)

            with tqdm(total=max_count) as progress_bar:
                for i, patch in tqdm(enumerate(parent.PatchIterator(parent, image_handler, self.patches_file, max_count, patch_size, patch_level, image_name))):
                    patches[i] = patch
                    progress_bar.update(1)

            return patches


        def __del__(self):
            if self.use_hdf5:
                self.patches_file.close()
                


    






