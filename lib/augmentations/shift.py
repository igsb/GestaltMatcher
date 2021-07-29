import torch
import torchvision.transforms.functional as TF

# Shifts the image and landmarks by a random number in the shift range, or [-shift_range, +shift_range], relative to image dimensions
# note: The shift is relative to the image dimensions, not that of the landmarks
def random_shift(img_tensor, landmarks, shift_range, fill=0., return_shift=False):

    # Handle the shift_range
    if hasattr(shift_range, '__len__'):
        # Stored in a list/tuple/other
        assert (len(shift_range) == 2 or len(shift_range) == 1), f"shift_range in random_rotate should have a " \
                                                                 f"len of 1 or 2 ({shift_range} given)."
        if len(shift_range) == 1:
            shift_range = (-shift_range[0], shift_range[0])
    else:
        shift_range = (-shift_range, shift_range)

    assert (shift_range[1] != shift_range[0]), f"Expect shift_range min and max to be different ({shift_range=} given)."
    
    shift_relative = torch.rand(2) * (shift_range[1] - shift_range[0]) + shift_range[0]
    shift_offsets = torch.hstack((shift_relative[0] * img_tensor.size(-1), shift_relative[1] * img_tensor.size(-2))).int()
    
    if return_shift:
        return *shift(img_tensor, landmarks, x_shift=shift_offsets[0], y_shift=shift_offsets[1], fill=fill), shift_offsets
    return shift(img_tensor, landmarks, x_shift=shift_offsets[0], y_shift=shift_offsets[1], fill=fill)

# Shifts the image and/or landmarks be absolute values from x_shift and y_shift
def shift(img_tensor, landmarks, x_shift=0, y_shift=0, fill=0.):
    landmarks_p = shift_landmarks(landmarks, x_shift, y_shift)
    
    # If shifting results in an image boundary being surpassed, we shift the entire image instead
    if torch.any(landmarks_p < 0) or torch.any(landmarks_p[0::2] > img_tensor.shape[-1]) or torch.any(landmarks_p[1::2] > img_tensor.shape[-2]):
        img_shift = TF.affine(img_tensor, angle=0, translate=[-x_shift, -y_shift], scale=1, shear=0, fill=fill)
        return img_shift, landmarks
    
    return img_tensor, landmarks_p
    

# Only works when the shifted bbox is within the image dimensions
# otherwise you have to (inversely) shift with affine transform to pad the image
def shift_landmarks(landmarks, x_shift, y_shift):
    landmarks_p = landmarks.detach().clone()
    # Shift the landmarks
    landmarks_p[0::2] += x_shift
    landmarks_p[1::2] += y_shift
    
    return landmarks_p