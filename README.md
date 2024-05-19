## easyPatch
-------------
** this repo is currently under construction, poorly documented, and prone to change

### About

This library allows you to easily patch histopathological images (whole slide images [WSI]).  Features include
 - generating patches using a desired zoom level and overlap
 - generating a binary mask from ground truth XML annotations
 - recombining patches into original image dimension 

easyPatch stores the generated patches in HDF5 files on disk as to not overwhelm RAM usage, as a single WSI can be constituted of thousands of patches depending on the zoom level. 

### Usage

This library is not registered on the PyPI as of yet. To use it please clone the source code and follow directions below.
All commands referenced below assume the user to be in the top level of the repo.

``` import create_patches.patch as patch```

To generate patches of multiple WSI images

``` patch.patch_images_without_mask(INPUT_DIR, OUTPUT_DIR, LEVEL, PATCH_SIZE, OVERLAP) ```

This command will create an HDF5 file in ```OUTPUT_DIR``` with a group corresponding to each WSI's name.  Each of these groups will have a dataset that holds all of the patches and associated overlap metadata which will be used during the recombine.
