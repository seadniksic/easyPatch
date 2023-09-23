**Instructions to run the pipeline.**

The current state of pipeline_v1.py:  will loop through all images and create masks, then will patch both masks and images.

Input: you will need to change the WSI_PATH (path to folder of svs images) and XML_PATH (path to folder of xml annotations) variablee.

Output: in create_mask/masks you will find all the masks that are created.  in create_patches/patches/[image name]/(masks or images) you will find patches of both the mask and the original image.