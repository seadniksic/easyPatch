import sys
sys.path.insert(0, ".")
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image, ImageOps
import PIL
import time
import cv2
import os
import openslide as op


#_______ Purpose _______#
#This script generates a binary mask (ground truth label) from xml for training of neural networks.
#Input:  path to matching wsi, level at which to produce mask
#Output on gen_mask() call:  Binary mask (0 and 255)

PIL.Image.MAX_IMAGE_PIXELS = 71647758000
#%%
def gen_mask(FILE_NAME, XML_PATH, WSI_PATH, LEVEL):
    #Open wsi and get height and width
    wsi_img = op.OpenSlide("{}/{}.svs".format(WSI_PATH, FILE_NAME))
    WIDTH = wsi_img.level_dimensions[LEVEL][0]
    HEIGHT = wsi_img.level_dimensions[LEVEL][1]
    print(WIDTH, HEIGHT)
    DOWNSAMPLE_FACTOR = 4**LEVEL

    #Create mask array
    img = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    print("Generating mask for {}".format(FILE_NAME))
    #Convert xml to list of vertices
    annotation = xml_to_points(FILE_NAME, XML_PATH, DOWNSAMPLE_FACTOR)
    #Convert list of vertices to binary mask
    points_to_mask(FILE_NAME, annotation, img, wsi_img, LEVEL)

def xml_to_points(FILE_NAME, XML_PATH, DOWNSAMPLE_FACTOR):
    root = ET.parse(XML_PATH + "{}.xml".format(FILE_NAME)).getroot()

    layers = {} #Each item in this list represents a distinct annotation class
    for count, layer in enumerate(root.findall('Annotation')):

        annotation = [] #List of regions in a specific annotation

        for Region in layer.findall('Regions/Region'):
            vertex_list = [] #List of vertices in a specific region
            vertex_list.append(int(Region.get('NegativeROA'))) 

            for vertex in Region.getiterator('Vertex'):
                point = [int(float(vertex.get('X'))), int(float(vertex.get('Y')))]
                point[0] = max(point[0], 0) / DOWNSAMPLE_FACTOR
                point[1] = max(point[1], 0) / DOWNSAMPLE_FACTOR
                vertex_list.append(point)
                
            annotation.append(vertex_list)
        
        layers[layer.get('Name')] = annotation
    
    return layers


def points_to_mask(FILE_NAME, layers, img, wsi_img, LEVEL):

    precedence = ['normal', 'low', 'high']
    #Create initial mask 
    for count, annotation_class in enumerate(precedence):
        if annotation_class in layers.keys():
            annotation_regions = layers[annotation_class]
            
            for Region in annotation_regions:
                negative = Region.pop(0)
                pts = np.array(Region, dtype=np.int32)
                if negative:      
                    cv2.fillPoly(img, [pts], 0)
                else:
                    cv2.fillPoly(img, [pts], count + 1)

 
    #Create second mask from initial image (with background thresholding)
    threshold_mask = np.array(ImageOps.grayscale(wsi_img.get_thumbnail((wsi_img.level_dimensions[LEVEL][0],wsi_img.level_dimensions[LEVEL][1]))))
    threshold_mask[threshold_mask > 220] = 0
    threshold_mask[threshold_mask > 0] = 1

    final_mask = img * threshold_mask
    print(f'max: {np.max(final_mask)}')
    final_mask[final_mask == 1] = 255
    final_mask[final_mask == 2] = 155
    final_mask[final_mask == 3] = 55
    print("saving")
    im = Image.fromarray(final_mask)
    if not os.path.exists("create_mask/masks/"):
        os.makedirs("create_mask/masks/")
    im.save("create_mask/masks/{}_mask.png".format(FILE_NAME))




    # 0 - background, nothing 
    # 1 - normal
    # 2 - low
    # 3 - high
# %%
