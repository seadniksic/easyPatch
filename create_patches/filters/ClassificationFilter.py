import numpy as np
import openslide as op
import torch

class ClassificationFilter:
    def __init__(self, mask_path, model, transforms):
        self.annotation_mask = op.OpenSlide(mask_path)
        self.accept_threshold = 50
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model
        self.model = self.model.to(self.device)
        self.model.eval()
        self.transforms = transforms

    # If the filter returns true the patch is removed from the returned patches (assuming return patches is True)
    def filter(self, image, current_patch, anchor, level, patch_size, downsample):
        '''
        anchor coming in is at level 0, but we are taking patches of the main image at level <level>.
        the tricky part:  the mask is currently generated at the level <level> (of the wsi).  But this 
        corresponds to the level 0 of the mask.  So the <level> parameter here is the wsi level, the anchor
        is at wsi level 0, and the mask's level 0 actually corresponds to wsi level <level>.  Sheesh.
        '''

        mask_patch = np.array(self.annotation_mask.read_region((*[int(x / downsample) for x in anchor],), 0, (patch_size, patch_size)))[:,:,:3] 

        if np.mean(mask_patch) > self.accept_threshold:

            x = torch.tensor(current_patch[:,:,:3], dtype=torch.float32).to(self.device).permute((2,0,1)).unsqueeze(dim=0)

            if self.transforms:
                x = self.transforms(x)

            #prediction on current patch
            with torch.no_grad():
                prediction = self.model(x)

            prediction = torch.argmax(prediction, dim=1).item()

            print(prediction)

            current_patch = current_patch.astype(np.float32) / 1.5

            if prediction:
                current_patch[:,:,0] += ((255 - current_patch[:, :, 0]) / 2)
            else:
                current_patch[:,:,2] += ((255 - current_patch[:, :, 2]) / 2)
            
            return True, current_patch.astype(np.uint8)
        else:
            return False, current_patch


    
