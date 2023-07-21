import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class Waterbirds(Dataset):
    def __init__(self, data_dir, split, transforms=None, seed=42):
        self.data = pd.read_csv(os.path.join(data_dir, "metadata.csv"))
        self.birds = ["landbird", "waterbird"]
        self.background = ["land", "water"]
        self.data_dir = data_dir
        self.transforms = transforms

        self.data = self.data.loc[self.data["split"] == 0]

        train_data = self.data.sample(frac=0.8, random_state=seed)
        val_test_data = self.data.drop(train_data.index)
        val_data = val_test_data.sample(frac=0.5, random_state=seed)
        test_data = val_test_data.drop(val_data.index)

        if split == "train":
            self.data = train_data.reset_index(drop=True)
        elif split == "val":
            self.data = val_data.reset_index(drop=True)
        elif split == "test":
            self.data = test_data.reset_index(drop=True)
        else:
            raise ValueError("split does not exist")

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        y = item["y"].item()
        filename = item["img_filename"]

        path = os.path.join(self.data_dir, filename)
        X = Image.open(os.path.join(self.data_dir, filename)).convert("RGB")

        if self.transforms is not None:
            X = self.transforms(X)

        y = torch.tensor(y)

        return X, y, path

    def __len__(self):
        return self.data.shape[0]
