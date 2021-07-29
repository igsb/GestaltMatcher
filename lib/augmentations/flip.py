## flip.py

import torch
import torchvision.transforms.functional as TF

#
def random_hflip(image_tensor, landmarks, p=0.5):
    if torch.rand(1) < p:
        landmarks_p = hflip_landmarks(landmarks, image_tensor.shape[-2:])
        return TF.hflip(image_tensor), landmarks_p
    else:
        return image_tensor, landmarks

def random_vflip(image_tensor, landmarks, p=0.5):
    if torch.rand(1) < p:
        landmarks_p = vflip_landmarks(landmarks, image_tensor.shape[-2:])
        return TF.vflip(image_tensor), landmarks_p
    else:
        return image_tensor, landmarks

def hflip(image_tensor, landmarks):
    return random_hflip(image_tensor, landmarks, p=1.0)
    
def vflip(image_tensor, landmarks):
    return random_vflip(image_tensor, landmarks, p=1.0)

def hflip_landmarks(landmarks, size):
    landmarks_p = landmarks.detach().clone()
    landmarks_p[0::2] = size[1] - landmarks_p[0::2]
    
    print(f"{landmarks_p=}")
    
    #return landmarks_p.flip(dims=[0]) # This could be usful when the order of the landmarks is important (e.g. left hand, right hand)
    return landmarks_p.flip(dims=[0])

def vflip_landmarks(landmarks, size):
    landmarks_p = landmarks.detach().clone()
    landmarks_p[1::2] = size[0] - landmarks_p[1::2]
    
    print(f"{landmarks_p=}")
        
    return landmarks_p