import pandas as pd
from torchvision.datasets import CelebA


class CelebAGenderHair(CelebA):
    def __init__(self, root, split, transform, download):
        super().__init__(root=root, split=split, transform=transform, download=download)
        self.indices = []

        # load attributes of all images
        self.attributes = pd.DataFrame(
            self.attr, columns=self.attr_names[:-1], index=None
        )

        # indices of males with black hair
        indices_1 = self.attributes.query(
            "Male==1 & Black_Hair==1 & Blond_Hair==0"
        ).index

        # indices of females with blond hair
        indices_0 = self.attributes.query(
            "Male==0 & Black_Hair==0 & Blond_Hair==1"
        ).index

        # aggregate the indices and create labels
        self.indices = indices_1.append(indices_0)
        self.labels = [1] * len(indices_1) + [0] * len(indices_0)

    def __getitem__(self, idx):
        X, _ = super().__getitem__(self.indices[idx])
        y = self.labels[idx]
        return X, y, self.filename[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


class CelebASingleClass(CelebA):
    def __init__(self, root, split, transform, download, class_id):
        super().__init__(root=root, split=split, transform=transform, download=download)
        self.cls = class_id

    def __getitem__(self, idx):
        X, y = super().__getitem__(idx)
        y = y[self.cls]
        return X, y, self.filename[idx]
