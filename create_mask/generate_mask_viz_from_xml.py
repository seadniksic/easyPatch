import sys
sys.path.insert(0, ".")
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import PIL
import time
import cv2
import overlay
import openslide as op

#_______ Purpose _______#
#This script generates a binary mask (ground truth label) from xml for training of neural networks.
#Input:  XML file specified by the XML_PATH variable, path to matching wsi, level at which to produce mask
#Output: if calling main():  Binary mask (0 and 255), temp mask for generating cutout (0 and 1), cutout image. 

HEIGHT = 0
WIDTH = 0
LEVEL = 1
DOWNSAMPLE_FACTOR = 4**LEVEL
XML_PATH = 'xml_example/xml_test_3.xml'
WSI_PATH = "C:\\Users\\sayba\\Documents\\University\\Research\\657cca23-0d37-445e-9a57-b3e385c5d4bd\\TCGA-P3-A5QF-01Z-00-DX1.EC33D817-A9FF-4406-9FE4-12D0F37AA189.svs"
MASK_OUTPUT_NAME = "mask_for_overlay.jpeg"

PIL.Image.MAX_IMAGE_PIXELS = 71647758000

def main():
    #Open wsi and get height and width
    wsi_img = op.OpenSlide(WSI_PATH)
    WIDTH = wsi_img.level_dimensions[LEVEL][0]
    HEIGHT = wsi_img.level_dimensions[LEVEL][1]
    print(wsi_img.level_dimensions)
    #Create mask array
    img = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    print("Generating mask")
    annotation = xml_to_points()
    points_to_mask(annotation, img)
    print("Generating overlay")
    overlay.save_to_cutout(WSI_PATH, MASK_OUTPUT_NAME)

def xml_to_points():
    root = ET.parse(XML_PATH).getroot()

    annotation = [] #list of lists of vertexes.  Each sub list represents a distinct region
    for Region in root.findall('Annotation/Regions/Region'):
        vertex_list = []
        vertex_list.append(int(Region.get('NegativeROA'))) 

        for vertex in Region.getiterator('Vertex'):
            point = [int(vertex.get('X')), int(vertex.get('Y'))]
            point[0] = max(point[0], 0) / DOWNSAMPLE_FACTOR
            point[1] = max(point[1], 0) / DOWNSAMPLE_FACTOR
            vertex_list.append(point)
            
        annotation.append(vertex_list)
    return annotation


def points_to_mask(annotation, img):
    for count, Region in enumerate(annotation):
        negative = Region.pop(0)
        pts = np.array(Region, dtype=np.int32)
        if negative == 1:
            cv2.fillPoly(img, [pts], 0)
        else:
            cv2.fillPoly(img, [pts], 255)
        
    
    im = Image.fromarray(img)
    im.save("binary_mask.jpeg")

    img[img == 255] = 1
    im = Image.fromarray(img)
    im.save("mask_for_overlay.jpeg")

main()