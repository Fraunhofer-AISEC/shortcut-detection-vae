import torch
from PIL import Image
from torch.utils.data import Dataset


class LemonQuality(Dataset):
    def __init__(self, paths, labels, transforms):
        super(LemonQuality, self).__init__()
        self.paths = paths
        self.labels = labels
        self.transforms = transforms

    def __getitem__(self, idx):
        path = self.paths[idx]
        X = Image.open(self.paths[idx]).convert("RGB")

        if self.transforms is not None:
            X = self.transforms(X)

        y = torch.tensor(self.labels[idx])

        return X, y, path

    def __len__(self):
        return len(self.paths)
