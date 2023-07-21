import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from robustness.tools.vis_tools import show_image_row
from torch.utils.data import DataLoader

from datasets.data_provider import get_datasets, get_transform
from models.classifiers import CNNClassifier
from utils.heatmap_utils import (
    create_images,
    generate_penultimate_features,
    get_topk_predictive_features,
)
from utils.setup_utils import get_config, seed_everything


def evaluate_heatmaps(
    classifier, dataset, dataloader, class_index, output_path, device
):
    penultimate_features, targets = generate_penultimate_features(
        classifier, dataloader, device=device
    )
    print(penultimate_features.shape)

    # imagenet_subset = ImageNetSubset(IMAGENET_PATH, all_indices)

    topk_feature_indices = get_topk_predictive_features(
        class_index, classifier, penultimate_features, k=5
    )
    grid_imgs = []
    grid_heatmaps = []
    img_labels = []
    heatmap_labels = []

    for i, feature_index in enumerate(topk_feature_indices):
        sorted_indices = np.argsort(-penultimate_features[:, feature_index])
        # filter by class
        sorted_indices = np.array(
            [i for i in sorted_indices if targets[i] == class_index]
        )
        # sorted_indices = sorted_indices[targets == class_index]
        max_5_indices = sorted_indices[:5]

        images, heatmaps = create_images(
            max_5_indices, feature_index, dataset, classifier, device
        )
        grid_imgs.append(images)
        grid_heatmaps.append(heatmaps)
        img_labels.append(f"{i} feature{feature_index}")
        heatmap_labels.append(f"{i} feature{feature_index}")

    show_image_row(
        grid_imgs,
        img_labels,
        tlist=[],
        fontsize=18,
        filename=f"{output_path}/images_class={class_index}",
    )
    show_image_row(
        grid_heatmaps,
        heatmap_labels,
        tlist=[],
        fontsize=18,
        filename=f"{output_path}/heatmap_class={class_index}",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="name of the input dataset",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="cuda visible device",
    )
    parser.add_argument(
        "--config_file",
        help="Path to the configuration file containing additional settings for the training process",
        default="config.yaml",
        type=str,
    )
    parser.add_argument(
        "--class_index",
        type=int,
        help="index of class to produce the heatmap for",
        default=0,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="classifier checkpoint",
        default="",
    )
    parser.add_argument(
        "--output_dir",
        help="Path to the directory where checkpoints, logs and intermediate outputs are saved",
        default="output",
        type=str,
    )

    args = parser.parse_args()
    config = get_config(args.config_file)
    config.data.name = args.dataset_name

    # Set up the necessary configurations and devices
    seed_everything(seed=config.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_path = Path(args.output_dir, config.data.name, "CNNClassifier")
    os.makedirs(output_path, exist_ok=True)

    _, _, test_dataset, config = get_datasets(
        config,
        train_transform=get_transform,
        test_transform=get_transform,
    )

    worker_init_fn = getattr(test_dataset, "init_worker", None)
    test_dl = DataLoader(
        test_dataset,
        batch_size=config.classifier.batch_size,
        shuffle=False,
        num_workers=4,
        worker_init_fn=worker_init_fn,
    )

    class_index = args.class_index
    assert (class_index >= 0) and (class_index < config.data.num_classes), class_index

    classifier = CNNClassifier(
        input_channels=config.data.channels,
        classes=config.data.num_classes,
    )
    classifier = classifier.to(device)
    classifier.load_state_dict(torch.load(args.checkpoint))
    classifier.eval()

    evaluate_heatmaps(
        classifier=classifier,
        dataset=test_dataset,
        dataloader=test_dl,
        class_index=class_index,
        output_path=output_path,
        device=device,
    )
