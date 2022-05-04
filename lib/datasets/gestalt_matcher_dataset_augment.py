## gestalt_matcher_dataset_augment.py
# GestaltMatcherDB with more extensive augmentation:
# rotation, translation, cropping, flipping, color jittering

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

import lib.augmentations as LA


# Function to normalize an image's pixel values
# Either in range [0,1] or [0,255]
def normalize(img, type='float'):
    normalized = (img - img.min()) / (img.max() - img.min())
    if type == 'int':
        return (normalized * 255).int()

    # Else: float
    return normalized


# Function to plot a tensor as image
# optionally waits for button press before continuing
def imshow(img_t, wait=False):
    plt.close()
    plt.imshow(img_t.permute(1, 2, 0))
    if wait:
        plt.waitforbuttonpress()


class GestaltMatcherDataset_augment(Dataset):
    def __init__(self, imgs_dir=-1, target_file_path=-1, in_channels=1, target_size=100, img_postfix='_rot',
                 augment=True, lookup_table=None):

        self.img_postfix = img_postfix
        self.target_size = target_size
        self.in_channels = in_channels

        if imgs_dir == -1:
            self.imgs_dir = "../data/GestaltMatcherDB/images_rot/"  # ~400 images
        else:
            self.imgs_dir = imgs_dir

        if target_file_path == -1:
            self.target_file = "../data/GestaltMatcherDB/case_a_train.csv"
        else:
            self.target_file = target_file_path

        self.targets = self.handle_target_file()
        self.bboxes = pd.read_csv("../data/GestaltMatcherDB/v1.0.1/gmdb_images_rot_v1.0.1face_coords.csv")

        if lookup_table:
            self.lookup_table = lookup_table
        else:
            self.lookup_table = self.targets["label"].value_counts().index.tolist()
            print(self.lookup_table)
            self.lookup_table.sort()

        self.augment = augment
        self.augment_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        ])

    def __len__(self):
        return len(self.targets)

    def get_lookup_table(self):
        return self.lookup_table

    def preprocess(self, img, bbox):
        # Sometimes the bbox seems to be invalid ... this handles it
        bbox = np.clip(bbox, 0, 10000)

        # plt.imshow(img)
        # plt.waitforbuttonpress()

        img = transforms.ToTensor()(img)
        # imshow(LA.get_bbox_image(img, bbox.reshape(-1, 2)), wait=True)

        # Extensively augments the image (rotate, flip, shift, crop, color jitter)
        if self.augment:
            corners = LA.bbox_to_corners(bbox)
            img, corners = LA.random_rotate(img, corners, 5, fill=0.5)
            # imshow(LA.get_bbox_image(img, LA.corners_to_bbox(corners)), wait=True)

            img, bbox = LA.random_crop(img, LA.corners_to_bbox(corners.reshape(-1, 2)), 0.02, fill=0.5)
            # imshow(LA.get_bbox_image(img, bbox.reshape(-1, 2)), wait=True)

            img, bbox = LA.random_shift(img, bbox.reshape(-1, 2), 0.02, fill=0.5)
            img = LA.get_bbox_image(img, bbox.reshape(-1, 2))
            # imshow(img, wait=True)
            img = self.augment_transform(img)
        else:
            # Only get the RoI
            img = LA.get_bbox_image(img, bbox.reshape(-1, 2))

        img = transforms.Resize((self.target_size, self.target_size))(img)  # Default size is (100,100)

        # desired number of channels is 1, so we convert to gray
        if self.in_channels == 1:
            img = transforms.Grayscale(1)(img)
        # imshow(img, wait=True)

        return normalize(img)

    def __getitem__(self, i, to_augment=True):
        img = Image.open(f"{self.imgs_dir}{self.targets.iloc[i]['image_id']}{self.img_postfix}.jpg")
        target_id = self.lookup_table.index(self.targets.iloc[i]['label'])
        bbox = self.bboxes.loc[self.bboxes.img == f"{self.targets.iloc[i]['image_id']}{self.img_postfix}.jpg"].values[0,
               1:]

        # Debugging line:
        # print(f"{self.targets.iloc[i]['image_id']}{self.img_postfix}.jpg \t{bbox=}")

        img = self.preprocess(img, bbox)

        return img, target_id

    def id_to_name(self, class_id):
        return self.lookup_table[class_id]

    def get_distribution(self):
        return list(self.targets.label.value_counts())

    def handle_target_file(self):
        df = pd.read_csv(self.target_file, delimiter=',').astype(int)

        ## Keep only frontal face photos
        # df = df[df["image_type"] == "Frontal face"] # Already taken care of in *.csv

        ## Keep only the rows that have a valid class/gene_name
        # df = df[~pd.isnull(df["gene_names"])]

        return df
