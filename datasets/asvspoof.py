import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ASVSpoof(Dataset):
    def __init__(
        self, root, split="train", attack_type="A01", subset="train", transforms=None
    ):
        self.root = root
        self.subset = subset
        self.data = pd.read_csv(
            f"{self.root}/labels_asv.txt",
            sep=" ",
            names=["pref", "file_prefix", "_", "attack_type", "label"],
            index_col="file_prefix",
        )
        self.data["file_name"] = None

        data_files = os.listdir(os.path.join(root, f"ASVspoof2019_LA_{subset}/mel"))
        for f_name in data_files:
            file_pref = f_name[:12]
            self.data.loc[file_pref, "file_name"] = f_name

        self.data = self.data[~self.data.file_name.isnull()]

        self.data = self.data[
            (self.data.label == "bonafide") | (self.data.attack_type == attack_type)
        ]
        self.data.label = self.data.apply(
            lambda x: 0 if x.label == "bonafide" else 1, axis=1
        )

        train_data = self.data.sample(frac=0.8, random_state=42)
        val_test_data = self.data.drop(train_data.index)
        val_data = val_test_data.sample(frac=0.5, random_state=42)
        test_data = val_test_data.drop(val_data.index)

        if split == "train":
            self.data = train_data
        elif split == "val":
            self.data = val_data
        else:
            self.data = test_data

        self.transforms = transforms

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        filename = item["file_name"]
        path = os.path.join(self.root, f"ASVspoof2019_LA_{self.subset}/mel", filename)

        X = np.load(path)
        X = (X - X.min()) / (X.max() - X.min())
        X = np.expand_dims(X, 2)

        X = X.repeat(3, axis=2).astype(np.float)

        y = item["label"]
        y = torch.tensor(y)

        if self.transforms is not None:
            X = self.transforms(X)

        return X, y, path

    def __len__(self):
        return self.data.shape[0]
