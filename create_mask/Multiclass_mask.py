#!/usr/bin/env python
# coding: utf-8

# In[45]:


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


# In[46]:


FILE_NAME= 'B18_Y'
WSI_PATH ='Slides/'
LEVEL=1


# In[60]:


PIL.Image.MAX_IMAGE_PIXELS = 71647758000
key = ['normal', 'low', 'high']

def gen_mask(FILE_NAME, WSI_PATH, LEVEL):
    #Open wsi and get height and width
    wsi_img = op.OpenSlide("Slides/{}.svs".format(FILE_NAME))
    WIDTH = wsi_img.level_dimensions[LEVEL][0]
    HEIGHT = wsi_img.level_dimensions[LEVEL][1]
    DOWNSAMPLE_FACTOR = 4**LEVEL

    #Create mask array
    img = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    print("Generating mask for {}".format(FILE_NAME))
    #Convert xml to list of vertices
    annotation = xml_to_points(FILE_NAME, DOWNSAMPLE_FACTOR, key)
    #Convert list of vertices to binary mask
    points_to_mask(FILE_NAME, annotation, img, wsi_img, LEVEL)

def xml_to_points(FILE_NAME, DOWNSAMPLE_FACTOR, key):
    tree = ET.parse('Slides/{}.xml'.format(FILE_NAME))
    root = tree.getroot()
    layers = [] #list of lists of vertexes.  Each sub list represents a distinct region
    for i, layer in enumerate(root.findall("./Annotation")):
        if (layer.attrib['Name']) == key[i]:
            annotation = []
            for Region in layer.findall('Regions/Region'):
                vertex_list = []
                vertex_list.append(int(Region.get('NegativeROA'))) 

                for vertex in Region.getiterator('Vertex'):
                    point = [int(float(vertex.get('X'))), int(float(vertex.get('Y')))]
                    point[0] = max(point[0], 0) / DOWNSAMPLE_FACTOR
                    point[1] = max(point[1], 0) / DOWNSAMPLE_FACTOR
                    vertex_list.append(point)
                
                annotation.append(vertex_list)
            layers.append(annotation)

    #if (len(layers))==3:
    return layers
    
    #else:
        #print('error')


def points_to_mask(FILE_NAME, layers, img, wsi_img, LEVEL):

    #Create initial mask 
    for count, annotation in enumerate(layers):
        for Region in annotation:
            negative = Region.pop(0)
            #print(Region)
            pts = np.array(Region, dtype=np.int32)
            cv2.fillPoly(img, [pts], count + 1)
 
    #Create second mask from initial image (with background thresholding)
    mask_2 = np.array(ImageOps.grayscale(wsi_img.get_thumbnail((wsi_img.level_dimensions[LEVEL][0],wsi_img.level_dimensions[LEVEL][1]))))
    mask_2[mask_2 > 220] = 0    
    mask_2[mask_2 > 0] = 1

    final_mask = img * mask_2
    final_mask[final_mask == 1] = 255
    final_mask[final_mask == 2] = 155
    final_mask[final_mask == 3] = 55
    print("saving")
    im = Image.fromarray(final_mask)
    if not os.path.exists("masks/"):
        os.makedirs("masks/")
    im.save("masks/{}_mask.jpeg".format(FILE_NAME))


# In[61]:


gen_mask(FILE_NAME, WSI_PATH, LEVEL)


# In[ ]:




