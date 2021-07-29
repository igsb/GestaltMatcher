## utils.py
import torch
import numpy as np

# Converts the bbox (of size[4] or size [2,2]) to corners (of size[4,2])
def bbox_to_corners(bbox):
    w,h = bbox[2]-bbox[0], bbox[3]-bbox[1]
    corners = np.array([bbox[0], bbox[1], bbox[0]+w, bbox[1], bbox[0], bbox[1]+h, bbox[2], bbox[3]])
    return corners.reshape(4,2)

# Covnerts corners (of size [4,2]) to bbox (of size [2,2])
def corners_to_bbox(corners):
    bbox = torch.cat([torch.min(corners, dim=0)[0], torch.max(corners, dim=0)[0]]).int()
    return bbox.reshape(2,2)

# Retrieve only the image tensor within the given bbox coordinates
def get_bbox_image(img_tensor, bbox):
    bbox_reshaped = bbox.reshape(4)
    img_crop = img_tensor[:,bbox_reshaped[1]:bbox_reshaped[3],bbox_reshaped[0]:bbox_reshaped[2]]
    return img_crop

# Pad the given image tensor on the (short) sides, 
# resulting in a square image tensor with the given image in the center.  
def padded_square_image(img_tensor, fill=0.):
    img_padded = torch.ones(3, max(img_tensor.shape[1:]), max(img_tensor.shape[1:])) * fill
    w_diff, h_diff = img_padded.size(2) - img_tensor.size(2), img_padded.size(1) - img_tensor.size(1)
    img_padded[:, int(h_diff/2.):int(h_diff/2. + img_tensor.size(1)), int(w_diff/2.):int(w_diff/2. + img_tensor.size(2))] = img_tensor
    return img_padded