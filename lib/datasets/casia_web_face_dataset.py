import os
from glob import glob
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from torchvision import transforms


class CasiaWebFaceDataset(Dataset):
    def __init__(self, imgs_dir=-1, target_file=-1, in_channels=1, target_size=100, augment=True):
        self.target_size = target_size
        self.in_channels = in_channels
        if imgs_dir == -1:
            self.imgs_dir = "../data/CASIA-WebFace/CASIA-cropped/" # ~35k images
        else:
            self.imgs_dir = imgs_dir

        if target_file == -1:
            self.target_file = "../data/CASIA-WebFace/CASIA-WebFace_ID2Name.txt"
        else:
            self.target_file = target_file

        self.target_file = pd.read_csv(self.target_file, sep=' ', names=['class_id', 'class_name'], dtype='string')

        self.ids = [y for x in os.walk(self.imgs_dir) for y in glob(os.path.join(x[0], '*.jpg'))]

        self.augment = augment
        self.augment_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        ])

    def __len__(self):
        return len(self.ids)

    def preprocess(self, img):
        resize = transforms.Resize((self.target_size, self.target_size)) # Default size is (100,100)
        img = resize(img)
        if self.augment:
            img = self.augment_transform(img)

        # desired number of channels is 1, so we convert to gray
        if self.in_channels == 1:
            img = transforms.Grayscale(1)(img)

        return transforms.ToTensor()(img)

    def __getitem__(self, i, to_augment=True):
        idx = self.ids[i].split('/')[-1]
        class_id, img_file = idx.split('\\')
        target_id = self.target_file[self.target_file['class_id']==class_id].index[0]

        img = Image.open(f"{self.imgs_dir}{class_id}/{img_file}")
        img = self.preprocess(img)

        return img, target_id

    def id_to_name(self, id):
        return self.target_file.iloc[id]['class_name']
