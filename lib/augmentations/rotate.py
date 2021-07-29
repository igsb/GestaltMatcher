import torch
import numpy as np
import torchvision.transforms
import torchvision.transforms.functional as TF


# Rotate the image and landmarks by a random angle in the angle range, or [-angle_range, +angle_range]
def random_rotate(img_tensor, landmarks, angle_range, fill=0., expand=True, return_angle=False,
                  interpolation=torchvision.transforms.InterpolationMode.BILINEAR):

    # Handle the angle_range
    if hasattr(angle_range, '__len__'):
        # Stored in a list/tuple/other
        assert (len(angle_range) == 2 or len(angle_range) == 1), f"angle_range in random_rotate should have a " \
                                                                 f"len of 1 or 2 ({angle_range} given)."
        if len(angle_range) == 1:
            angle_range = (-angle_range[0], angle_range[0])
    else:
        angle_range = (-angle_range, angle_range)

    assert (angle_range[1] != angle_range[0]), f"Expect angle_range min and max to be different ({angle_range=} given)."
    angle = int(torch.rand(1) * (angle_range[1] - angle_range[0]) + angle_range[0])

    if return_angle:
        return *rotate(img_tensor, landmarks, angle=angle, fill=fill, expand=expand, interpolation=interpolation), angle
    return rotate(img_tensor, landmarks, angle=angle, fill=fill, expand=expand, interpolation=interpolation)


# Rotate the image and landmarks by a given angle
def rotate(img_tensor, landmarks, angle, fill=0., expand=True,
           interpolation=torchvision.transforms.InterpolationMode.BILINEAR):
    img_rot = TF.rotate(img_tensor, angle=angle, interpolation=interpolation, expand=expand, fill=fill)

    # Get image size before rotation - used for rotation around origin
    # ... and after rotation in order to correctly pad the landmark coordinates (if expand is True)
    old_size = img_tensor.shape[-2:]
    new_size = img_rot.shape[-2:]
    
    landmarks_rot = rotate_landmarks(landmarks, angle=angle, old_size=old_size, new_size=new_size, expand=expand)
    return img_rot, landmarks_rot


# Function to rotate the landmarks
# note: When expand is False the landmarks coordinates may end up outside the image boundaries ...
def rotate_landmarks(landmarks, angle, old_size, new_size, expand=False):
    if expand is not True:
        assert (old_size == new_size), f"Expand == False, so expected old_size to be equal to new_size " \
                                       f"({old_size=},{new_size=} given)."

    angle = -angle  # counter clockwise rotation
    transformation_matrix = torch.tensor([
        [+np.cos(np.radians(angle)), +np.sin(np.radians(angle))],
        [-np.sin(np.radians(angle)), +np.cos(np.radians(angle))]
    ])
    
    landmarks_p = landmarks.reshape(-1,2) - np.array([old_size[1] * 0.5, old_size[0] * 0.5])  # rotate around image center
    landmarks_p = np.matmul(landmarks_p, transformation_matrix) + np.array([old_size[1] * 0.5, old_size[0] * 0.5])

    if expand is True:
        # handle expansion
        landmarks_p = pad_landmarks(landmarks_p, (new_size[1] - old_size[1]) * 0.5, (new_size[0] - old_size[0]) * 0.5)

    return landmarks_p


# Function to pad the landmarks
# used when rotate with expand=True is used
def pad_landmarks(landmarks, x_pad, y_pad):
    return landmarks + np.array([x_pad, y_pad])
