import os.path as osp

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

torch.manual_seed(2024)

AVERAGE_IMAGE_VALUE = 115

LABEL_ID_DIC = {
    "n01440764": 0,
    "n02102040": 1,
    "n02979186": 2,
    "n03000684": 3,
    "n03028079": 4,
    "n03394916": 5,
    "n03417042": 6,
    "n03425413": 7,
    "n03445777": 8,
    "n03888257": 9,
}
LABEL_ID_INV_DIC = {v: k for k, v in LABEL_ID_DIC.items()}

LABEL_NAME_DIC = {
    0: "Tench",
    1: "English springer",
    2: "Cassette player",
    3: "Chainsaw",
    4: "Church",
    5: "French horn",
    6: "Garbage truck",
    7: "Gas pump",
    8: "Golf ball",
    9: "Parachute",
}
LABEL_NAME_INV_DIC = {v: k for k, v in LABEL_NAME_DIC.items()}

TRANSFORM = transforms.Compose([transforms.CenterCrop(320), transforms.ToTensor()])


class ImageDataset(Dataset):
    def __init__(self, data_dir, data_file, transform=None, val=False):
        self.data_dir = data_dir
        data_summary = pd.read_csv(osp.join(data_dir, data_file))
        self.data_summary = data_summary[data_summary.is_valid == val].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.data_summary)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_file = self.data_summary.loc[idx, "path"]
        image = Image.open(osp.join(self.data_dir, image_file)).convert("RGB")

        label = self.data_summary.loc[idx, "noisy_labels_0"]
        one_hot_label = torch.zeros(len(LABEL_ID_DIC.keys()))
        one_hot_label[LABEL_ID_DIC[label]] = 1

        if self.transform:
            image = self.transform(image)

        return image, one_hot_label
