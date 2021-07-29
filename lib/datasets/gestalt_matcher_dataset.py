## gestalt_matcher_dataset.py
# GestaltMatcherDB with more only basic augmentation:
# flipping, color jittering

import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


# Function to normalize an image's pixel values
# Either in range [0,1] or [0,255]
def normalize(img, type='float'):
    normalized = (img - img.min()) / (img.max() - img.min())
    if type == 'int':
        return (normalized * 255).int()

    # Else: float
    return normalized


class GestaltMatcherDataset(Dataset):
    def __init__(self, imgs_dir=-1, target_file_path=-1, in_channels=1, target_size=100, img_postfix='', augment=True,
                 lookup_table=None):

        self.img_postfix = img_postfix
        self.target_size = target_size
        self.in_channels = in_channels

        if imgs_dir == -1:
            self.imgs_dir = "../data/GestaltMatcherDB/images_cropped/"
        else:
            self.imgs_dir = imgs_dir

        if target_file_path == -1:
            self.target_file = "../data/GestaltMatcherDB/case_a_train.csv"
        else:
            self.target_file = target_file_path

        self.targets = self.handle_target_file()

        # self.lookup_table = sorted(set(self.targets))

        if lookup_table:
            self.lookup_table = lookup_table
        else:
            # self.lookup_table = self.targets["gene_names"].value_counts().index.tolist()
            self.lookup_table = self.targets["label"].value_counts().index.tolist()
            self.lookup_table.sort()

        self.augment = augment
        self.augment_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            # transforms.Normalize((x,x,x), (x,x,x)) # Don't improve performance...
        ])

    def __len__(self):
        return len(self.targets)

    def get_lookup_table(self):
        return self.lookup_table

    def preprocess(self, img):
        resize = transforms.Resize((self.target_size, self.target_size))  # Default size is (100,100)
        img = resize(img)
        if self.augment:
            img = self.augment_transform(img)

        # desired number of channels is 1, so we convert to gray
        if self.in_channels == 1:
            img = transforms.Grayscale(1)(img)
        img = transforms.ToTensor()(img)
        return normalize(img)

    def __getitem__(self, i, to_augment=True):
        img = Image.open(f"{self.imgs_dir}{self.targets.iloc[i]['image_id']}{self.img_postfix}.jpg")
        target_id = self.lookup_table.index(self.targets.iloc[i]['label'])

        img = self.preprocess(img)

        return img, target_id

    def id_to_name(self, class_id):
        return self.lookup_table[class_id]

    def get_distribution(self):
        return list(self.targets.label.value_counts())

    def handle_target_file(self):
        df = pd.read_csv(self.target_file, delimiter=',')

        ## Keep only frontal face photos
        # df = df[df["image_type"] == "Frontal face"] # Already taken care of in *.csv

        ## Keep only the rows that have a valid class/gene_name
        # df = df[~pd.isnull(df["gene_names"])]

        return df
