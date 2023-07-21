"""
Script for evaluating the predictiveness of the latent space of VAE models.

This script analyzes the predictiveness of the latent space of VAE models by training a classifier with the VAE as a backbone. It computes the predictiveness of each dimension in the latent space based on different hyperparameter configurations.

Usage:
python evaluate_predictiveness.py [--dataset_name DATASET_NAME] [--device DEVICE] [--lr LR [LR ...]] [--kld_weight KLD_WEIGHT [KLD_WEIGHT ...]] [--cls_weight CLS_WEIGHT [CLS_WEIGHT ...]] [--latent_dim LATENT_DIM [LATENT_DIM ...]] [--batch_size BATCH_SIZE [BATCH_SIZE ...]] [--type TYPE] [--config_file CONFIG_FILE] [--output_dir OUTPUT_DIR] [-v VERSION]

Arguments:
    --dataset_name (str): Name of the input dataset for evaluation.
    --device (str): CUDA visible device for running the training/evaluation process.
    --lr (List[float]): Learning rate(s) to be used for training the VAE.
    --kld_weight (List[float]): KL divergence weight(s) for the VAE loss.
    --cls_weight (List[float]): Classifier weight(s) for the VAE loss. Defaults to 0 (recommended).
    --latent_dim (List[int]): Dimensionality of the latent space.
    --batch_size (List[int]): Batch size(s) for training.
    --type (str): Specification of the VAE type.
    --config_file (str): Path to the configuration file containing additional settings for the training process.
    --output_dir (str): Path to the directory where checkpoints, logs, and intermediate outputs are saved.
    -v, --version (int): The version number of the experiment to perform evaluation (only relevant for evaluation). Defaults to -1.

Notes:
- This script requires the dataset, model, and utils modules.
- The output directory will be created if it doesn't exist and will contain the predictiveness analysis results.

Example:
python evaluate_predictiveness.py --dataset_name lemon_quality --device 0 --lr 0.001 0.01 --kld_weight 0.01 0.1 --latent_dim 32 --batch_size 128 --type vae --config_file config.yaml --output_dir output -v 0

This script performs an evaluation of the predictiveness of the latent space of VAE models. It trains a classifier with the VAE as a backbone and computes the predictiveness of each dimension in the latent space using different hyperparameter configurations.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch import optim

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from datasets.data_provider import get_dataloaders, get_datasets, get_transform
from models.classifiers import CNNClassifier
from utils.setup_utils import get_config
from utils.eval_utils import evaluate_classifier
from utils.train_utils import train_classifier

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
        "--output_dir",
        help="Path to the directory where checkpoints, logs and intermediate outputs are saved",
        default="output",
        type=str,
    )

    args = parser.parse_args()
    config = get_config(args.config_file)
    config.data.name = args.dataset_name

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_path = Path(args.output_dir, config.data.name, "CNNClassifier")
    os.makedirs(output_path, exist_ok=True)

    # Get the datasets
    train_dataset, val_dataset, test_dataset, config = get_datasets(
        config,
        train_transform=get_transform,
        test_transform=get_transform,
    )
    # Get the dataloaders
    train_dl, val_dl, test_dl = get_dataloaders(
        train_dataset, val_dataset, test_dataset, bs=config.classifier.batch_size
    )

    classifier = CNNClassifier(
        classes=config.data.num_classes, input_channels=config.data.channels
    )
    classifier = classifier.to(device)

    optimizer = optim.Adam(
        classifier.parameters(),
        lr=config.classifier.lr,
        weight_decay=config.classifier.weight_decay,
    )
    scheduler = None  # Alternative: StepLR(optimizer, step_size=10, gamma=0.75)

    criterion = nn.CrossEntropyLoss()

    # Train classifier with frozen VAE backbone
    classifier, val_acc, val_loss = train_classifier(
        classifier=classifier,
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        epochs=config.classifier.epochs,
        path=output_path,
    )

    # Load the best checkpoint
    classifier.load_state_dict(torch.load(output_path / "classifier.pth"))

    # Evaluate the classifier on test data
    classifier.eval()
    test_acc, test_loss = evaluate_classifier(
        classifier=classifier,
        dataloader=test_dl,
        device=device,
        criterion=criterion,
    )

    print("Best test accuracy: ", test_acc)
    print("Best test loss: ", test_loss)
