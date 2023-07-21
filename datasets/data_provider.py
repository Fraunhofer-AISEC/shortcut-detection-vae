from pathlib import Path

import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from datasets.asvspoof import ASVSpoof
from datasets.celeba import CelebAGenderHair, CelebASingleClass
from datasets.colored_mnist import ColoredMNIST, ColoredMNIST5
from datasets.covid.chestxray14h5 import ChestXray14H5Dataset
from datasets.covid.domainconfoundeddatasets import DomainConfoundedDataset
from datasets.covid.githubcovid import GitHubCOVIDDataset
from datasets.lemon_quality import LemonQuality
from datasets.waterbirds import Waterbirds


def get_transform(img_size):
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size), antialias=None),
        ]
    )


def get_datasets(config, train_transform=get_transform, test_transform=get_transform):
    if config.data.name == "waterbirds":
        dataset_path = Path(config.data.base_path, "waterbird_complete95_forest2water2")
        train_dataset = Waterbirds(
            data_dir=dataset_path,
            split="train",
            transforms=train_transform(img_size=config.data.img_size),
            seed=config.seed,
        )
        val_dataset = Waterbirds(
            data_dir=dataset_path,
            split="val",
            transforms=test_transform(img_size=config.data.img_size),
            seed=config.seed,
        )
        test_dataset = Waterbirds(
            data_dir=dataset_path,
            split="test",
            transforms=test_transform(img_size=config.data.img_size),
            seed=config.seed,
        )
        config.data.channels = 3
        config.data.num_classes = 2

    elif config.data.name == "celeba_gender_hair":
        dataset_path = Path(config.data.base_path)
        train_dataset = CelebAGenderHair(
            root=str(dataset_path),
            split="train",
            transform=train_transform(img_size=config.data.img_size),
            download=True,
        )
        val_dataset = CelebAGenderHair(
            root=str(dataset_path),
            split="valid",
            transform=test_transform(img_size=config.data.img_size),
            download=True,
        )
        test_dataset = CelebAGenderHair(
            root=str(dataset_path),
            split="test",
            transform=test_transform(img_size=config.data.img_size),
            download=True,
        )
        config.data.channels = 3
        config.data.num_classes = 2

    elif config.data.name == "celeba":
        dataset_path = Path(config.data.base_path)
        class_id = 9  # blond hair
        train_dataset = CelebASingleClass(
            root=str(dataset_path),
            split="train",
            transform=train_transform(img_size=config.data.img_size),
            download=True,
            class_id=class_id,
        )
        val_dataset = CelebASingleClass(
            root=str(dataset_path),
            split="valid",
            transform=test_transform(img_size=config.data.img_size),
            download=True,
            class_id=class_id,
        )
        test_dataset = CelebASingleClass(
            root=str(dataset_path),
            split="test",
            transform=test_transform(img_size=config.data.img_size),
            download=True,
            class_id=class_id,
        )
        config.data.channels = 3
        config.data.num_classes = 2

    elif config.data.name == "cmnist":
        dataset_path = Path(config.data.base_path, "mnist")
        train_dataset = ColoredMNIST(
            root=str(dataset_path),
            split="train",
            transform=train_transform(img_size=config.data.img_size),
            download=True,
        )
        val_dataset = ColoredMNIST(
            root=str(dataset_path),
            split="val",
            transform=test_transform(img_size=config.data.img_size),
            download=True,
        )
        test_dataset = ColoredMNIST(
            root=str(dataset_path),
            split="test",
            transform=test_transform(img_size=config.data.img_size),
            download=True,
        )
        config.data.channels = 3
        config.data.num_classes = 10

    elif config.data.name == "cmnist_5":
        dataset_path = Path(config.data.base_path, "mnist")
        train_dataset = ColoredMNIST5(
            root=str(dataset_path),
            split="train",
            transform=train_transform(img_size=config.data.img_size),
            download=True,
        )
        val_dataset = ColoredMNIST5(
            root=str(dataset_path),
            split="val",
            transform=test_transform(img_size=config.data.img_size),
            download=True,
        )
        test_dataset = ColoredMNIST5(
            root=str(dataset_path),
            split="test",
            transform=test_transform(img_size=config.data.img_size),
            download=True,
        )
        config.data.channels = 3
        config.data.num_classes = 5

    elif config.data.name == "lemon_quality":
        dataset_path = Path(config.data.base_path, "lemon_dataset")
        good_quality_paths = sorted(
            [str(p) for p in (dataset_path / "good_quality").iterdir()]
        )
        bad_quality_paths = sorted(
            [str(p) for p in (dataset_path / "bad_quality").iterdir()]
        )
        background_paths = sorted(
            [str(p) for p in (dataset_path / "empty_background").iterdir()]
        )
        paths = good_quality_paths + bad_quality_paths + background_paths

        labels = [0] * len(good_quality_paths)
        labels += [1] * len(bad_quality_paths)
        labels += [2] * len(background_paths)

        X_train, X_test, y_train, y_test = train_test_split(
            paths, labels, test_size=0.2, random_state=config.seed
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_test, y_test, test_size=0.5, random_state=config.seed
        )

        train_dataset = LemonQuality(
            X_train,
            y_train,
            transforms=get_transform(img_size=config.data.img_size),
        )
        val_dataset = LemonQuality(
            X_val,
            y_val,
            transforms=get_transform(img_size=config.data.img_size),
        )
        test_dataset = LemonQuality(
            X_test,
            y_test,
            transforms=get_transform(img_size=config.data.img_size),
        )

        config.data.channels = 3
        config.data.num_classes = 3

    elif config.data.name == "covid":
        dataset_path = Path(config.data.base_path, "covid")
        train_dataset = DomainConfoundedDataset(
            ChestXray14H5Dataset(
                root=dataset_path,
                fold="train",
                labels="chestx-ray14",
                random_state=config.seed,
                transform=get_transform(img_size=config.data.img_size),
            ),
            GitHubCOVIDDataset(
                root=dataset_path,
                fold="train",
                labels="chestx-ray14",
                random_state=config.seed,
                transform=get_transform(img_size=config.data.img_size),
            ),
        )

        val_dataset = DomainConfoundedDataset(
            ChestXray14H5Dataset(
                root=dataset_path,
                fold="val",
                labels="chestx-ray14",
                random_state=config.seed,
                transform=get_transform(img_size=config.data.img_size),
            ),
            GitHubCOVIDDataset(
                root=dataset_path,
                fold="val",
                labels="chestx-ray14",
                random_state=config.seed,
                transform=get_transform(img_size=config.data.img_size),
            ),
        )

        test_dataset = DomainConfoundedDataset(
            ChestXray14H5Dataset(
                root=dataset_path,
                fold="test",
                labels="chestx-ray14",
                random_state=config.seed,
                transform=get_transform(img_size=config.data.img_size),
            ),
            GitHubCOVIDDataset(
                root=dataset_path,
                fold="test",
                labels="chestx-ray14",
                random_state=config.seed,
                transform=get_transform(img_size=config.data.img_size),
            ),
        )

        config.data.channels = 3
        config.data.num_classes = 2

    elif config.data.name == "asv_spoof":
        dataset_path = Path(config.data.base_path, "LA")
        train_dataset = ASVSpoof(
            root=str(dataset_path),
            split="train",
            attack_type="A01",
            subset="train",
            transforms=train_transform(img_size=config.data.img_size),
        )
        val_dataset = ASVSpoof(
            root=str(dataset_path),
            split="val",
            attack_type="A01",
            subset="train",
            transforms=test_transform(img_size=config.data.img_size),
        )
        test_dataset = ASVSpoof(
            root=str(dataset_path),
            split="test",
            attack_type="A01",
            subset="train",
            transforms=test_transform(img_size=config.data.img_size),
        )

        config.data.channels = 3
        config.data.num_classes = 2

    else:
        raise ValueError("Dataset not specified.")

    return train_dataset, val_dataset, test_dataset, config


def get_dataloaders(train_dataset, val_dataset, test_dataset, bs):
    worker_init_fn = getattr(train_dataset, "init_worker", None)
    # DataLoader for training data
    train_dl = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=2,
        worker_init_fn=worker_init_fn,
    )

    # DataLoader for validation data
    worker_init_fn = getattr(val_dataset, "init_worker", None)
    val_dl = DataLoader(
        val_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=2,
        worker_init_fn=worker_init_fn,
    )

    # DataLoader for test data
    worker_init_fn = getattr(test_dataset, "init_worker", None)
    test_dl = DataLoader(
        test_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=2,
        worker_init_fn=worker_init_fn,
    )

    return train_dl, val_dl, test_dl
