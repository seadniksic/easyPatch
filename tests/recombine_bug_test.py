''' Test recombine bug on smaller cases'''


from create_patches import patch
import openslide as op
from nn_utilities.deeplab import DeepLab2D
from create_patches import recombine
from nn_utilities import heatmap
from nn_utilities.unet_better import UNet
import torch
import os
from PIL import Image
import numpy as np
import torch.nn
import time

patch_size = 512
patch_level = 0
test_arr = []

def test(img_path):
    #Initialize saved model weights
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # model = DeepLab2D(3,3)
    # model.load_state_dict(torch.load(model_path))
    # model.eval()
    # model.to(device)
       
    patched_image, metadata = patch.patch_without_mask(img_path, patch_level, patch_size, 12)
    overlap = [metadata[0], metadata[1]]
    num_x_patches = metadata[2]
    num_y_patches = metadata[3]
    patched_image = patched_image[:,:,:,:3] #Remove alpha channel from patches

    
    # test_data = torch.from_numpy(patched_image).permute(0, 3, 1, 2)
    
    # final_masks = []
    # print('doing')
    # with torch.no_grad():
    #     for index, image in enumerate(test_data):
    #         image = image.unsqueeze(axis=0)
    #         image = image.to(device=device, dtype=torch.float32)
    #         masks_pred = model(image)
    #         masks_pred = masks_pred.to(device='cpu')
    #         final_masks.append(np.array(masks_pred.squeeze(dim=0).permute(1,2,0), dtype=np.float32))
    # final_masks = np.array(final_masks, dtype=np.int16)
    final_mask = recombine.recombine_from_patches_array_avg_overlap(patched_image, overlap, num_x_patches, num_y_patches)
    # heatmap.gen_heatmap(final_mask[:, :, 2], img_path.split(os.path.sep)[-1].split(".")[0])
    print('Heatmap generated')

    #Convert output to usable form
    # final_mask[final_mask < .1] = 0
    # final_mask = final_mask.argmax(axis=2, out=np.empty(final_mask.shape[:2] , dtype=np.uint8))

    # final_mask[final_mask == 1] = 55
    # final_mask[final_mask == 2] = 155
    
    final_mask = Image.fromarray(final_mask.astype(np.uint8))
    test_arr = final_mask
            
    final_mask.save("Recomb_bug_test_uint8overlap_T27n.jpeg", quality=100)
    return final_mask

now = time.time()
print(os.getcwd())
test('..\\data\\test_cases\\images\\T27_n.svs' )  
print(f'Processed for {time.time() - now} seconds')