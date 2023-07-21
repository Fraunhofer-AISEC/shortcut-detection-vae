import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import MNIST


class ColoredMNIST(MNIST):
    def __init__(self, root, transform, download, split="train", p_corr=0.995):
        train = True if split == "train" else False
        super().__init__(root=root, train=train, transform=transform, download=download)

        self.split = split
        self.p_corr = p_corr
        self.label_to_color = {i: c for i, c in enumerate(plt.get_cmap("tab10").colors)}

    def __getitem__(self, idx):
        if self.split == "test":
            idx = idx + super().__len__() // 2

        X, y = super().__getitem__(idx)

        X = np.repeat(X, 3, axis=0)
        color = self.label_to_color[y]

        if np.random.random() < self.p_corr:
            color = self.label_to_color[y]
        else:
            color_idx = np.random.choice(10)
            color = self.label_to_color[color_idx]

        X = X * np.expand_dims(color, [1, 2])

        return X.float(), y, ""

    def __len__(self):
        # split MNIST test set into val and test set
        if self.split == "val":
            return super().__len__() // 2
        elif self.split == "test":
            return round(super().__len__() / 2)
        else:
            return super().__len__()


class ColoredMNIST5(MNIST):
    def __init__(self, root, transform, download, split="train", p_corr=0.995):
        train = True if split == "train" else False
        super().__init__(root=root, train=train, transform=transform, download=download)

        self.split = split
        self.p_corr = p_corr
        self.label_to_color = ["#ff0000", "#d0ff00", "#00ff5c", "#0074ff", "#b800ff"]
        self.class_map = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]

    def __getitem__(self, idx):
        if self.split == "test":
            idx = idx + super().__len__() // 2

        X, y = super().__getitem__(idx)
        y = self.class_map[y]
        X = np.repeat(X, 3, axis=0)

        if np.random.random() < self.p_corr:
            color_hex = self.label_to_color[y]
        else:
            color_hex = np.random.choice(self.label_to_color)
        color = matplotlib.colors.to_rgb(color_hex)

        X = X * np.expand_dims(color, [1, 2])

        return X.float(), y, ""

    def __len__(self):
        # split MNIST test set into val and test set
        if self.split == "val":
            return super().__len__() // 2
        elif self.split == "test":
            return round(super().__len__() / 2)
        else:
            return super().__len__()
