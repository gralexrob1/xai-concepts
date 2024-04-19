import os.path as osp

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

DATA_PATH = "../imagenette"
DATA_SUMMARY = "noisy_imagenette.csv"

LABEL_DIC = {
    "n01440764": "Tench",
    "n02102040": "English springer",
    "n02979186": "Cassette player",
    "n03000684": "Chainsaw",
    "n03028079": "Church",
    "n03394916": "French horn",
    "n03417042": "Garbage truck",
    "n03425413": "Gas pump",
    "n03445777": "Golf ball",
    "n03888257": "Parachute",
}


class ImageDataset(Dataset):
    def __init__(self, csv_file, transform=None, val=False):
        data_summary = pd.read_csv(csv_file)
        self.data_summary = data_summary[data_summary.is_valid == val].reset_index()
        self.transform = transform

    def __len__(self):
        return len(self.data_summary)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_file = self.data_summary.loc[idx, "path"]
        image = Image.open(osp.join(DATA_PATH, image_file)).convert("RGB")

        label = self.data_summary.loc[idx, "noisy_labels_0"]

        if self.transform:
            image = self.transform(image)

        return image, label
