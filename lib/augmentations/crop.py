import torch
import torchvision.transforms.functional as TF
from .shift import shift_landmarks 

# Shifts the image and landmarks by a random percentage in the shift range, or [-crop_range, +crop_range]
def random_crop(img_tensor, landmarks, crop_range, fill=0., return_crop=False):

    # Handle the crop_range
    if hasattr(crop_range, '__len__'):
        # Stored in a list/tuple/other
        assert (len(crop_range) == 1 or len(crop_range) == 2), f"crop_range in random_rotate should have a " \
                                                                 f"len of 1 or 2 ({crop_range} given)."
        if len(crop_range) == 1:
            crop_range = (-crop_range[0], crop_range[0])
    else:
        crop_range = (-crop_range, crop_range)

    #assert (crop_range[1] != crop_range[0]), f"Expect crop_range min and max to be different ({crop_range=} given)."
    crop_relative = torch.rand(4) * (crop_range[1] - crop_range[0]) + crop_range[0]
    crop_offsets = torch.hstack((crop_relative[0::2] * img_tensor.size(-1), crop_relative[1::2] * img_tensor.size(-2))).int()

    if return_crop:
        return *crop(img_tensor, landmarks, x1_shift=crop_offsets[0], y1_shift=crop_offsets[2], x2_shift=crop_offsets[1], y2_shift=crop_offsets[3], fill=fill), crop_relative
    return crop(img_tensor, landmarks, x1_shift=crop_offsets[0], y1_shift=crop_offsets[2], x2_shift=crop_offsets[1], y2_shift=crop_offsets[3], fill=fill)

def crop(img_tensor, landmarks, x1_shift=0, y1_shift=0, x2_shift=0, y2_shift=0, fill=0.):
    landmarks_p = crop_bbox(landmarks, x1_shift, y1_shift, x2_shift, y2_shift)
    
    # If shifting results in an image boundary being surpassed, we shift the entire image instead
    if torch.any(landmarks_p < 0) or torch.any(landmarks_p[0::2] > img_tensor.shape[-1]) or torch.any(landmarks_p[1::2] > img_tensor.shape[-2]):
        img_crop = TF.affine(img_tensor, angle=0, translate=[-x1_shift, -y1_shift], scale=1, shear=0, fill=fill)
        return img_crop, landmarks

    return img_tensor, landmarks_p
    

# Only works when the shifted bbox is within the image dimensions
# otherwise you have to (inversely) shift with affine transform to pad the image
def crop_bbox(bbox, x1_shift, y1_shift, x2_shift, y2_shift):
    bbox_p = bbox.detach().clone()
    bbox_p = bbox_p.reshape(4)
    
    # Shift the landmarks
    bbox_p[0] += x1_shift
    bbox_p[1] += y1_shift
    
    bbox_p[2] += x2_shift
    bbox_p[3] += y2_shift
    
    return bbox_p

def crop_landmarks(landmarks, x1_shift, y1_shift):
    return shift_landmarks(landmarks, x1_shift, y1_shift)

## TODO:
# Do we want to change the input image? Or just the landmarks?
# What if we're using corners and landmarks? (bbox)
# What if we're using actual landmarks?