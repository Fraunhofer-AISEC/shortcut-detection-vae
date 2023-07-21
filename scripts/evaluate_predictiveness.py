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

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from easydict import EasyDict
from torch import optim
from torch.utils.data import DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from datasets.data_provider import get_dataloaders, get_datasets, get_transform
from models.classifiers import VAEClassifier
from models.resnet_vae import ResnetVAE
from utils.setup_utils import (
    get_latest_experiment_version,
    get_parser,
    setup_experiment,
)
from utils.eval_utils import compute_predictiveness_per_dimension, evaluate_classifier
from utils.train_utils import train_classifier


def evaluate_predictiveness(
    config: EasyDict,
    vae,
    train_dl: DataLoader,
    val_dl: DataLoader,
    test_dl: DataLoader,
    output_path: Path,
    device: torch.device,
) -> None:
    """
    Evaluates the predictiveness of the latent space of a VAE by training
    a classifier with the VAE as a backbone.
    Computes the predictiveness of each dimension in the latent space using
    the trained classifier.

    Args:
        config (EasyDict): Configuration dictionary.
        args (argparse.Namespace): Arguments representing different hyperparameter values.
        train_dl (DataLoader): Training data loader.
        val_dl (DataLoader): Validation data loader.
        test_dl (DataLoader): Test data loader.
        output_path (Path): Path to save the output.
        device (torch.device): Device to run the evaluation on.

    Returns:
        None
    """
    classifier = VAEClassifier(
        vae=vae,
        latent_dim=config.vae.latent_dim,
        classes=config.data.num_classes,
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

    # Examine the importance of latent dimensions for classification (only with VAEClassifier)
    compute_predictiveness_per_dimension(classifier, config)

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


if __name__ == "__main__":
    parser = get_parser(
        description="Read arguments for evaluating the predictiveness of latent variables"
    )
    # add additional argument for evaluation
    parser.add_argument(
        "-v",
        "--version",
        help="The version no. of the experiment to perform evaluation",
        default=-1,
        type=int,
    )
    config, args, device = setup_experiment(parser)

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

    # Perform evalution on all experiments determined by the combination of provided parameters.
    for bs, dim, lr, kld_weight, cls_weight in zip(
        args.batch_size, args.latent_dim, args.lr, args.kld_weight, args.cls_weight
    ):
        try:
            config.vae.batch_size = bs
            config.vae.latent_dim = dim
            config.vae.lr = lr
            config.vae.kld_weight = kld_weight
            config.vae.cls_weight = cls_weight

            path = Path(
                args.output_dir,
                config.data.name,
                config.vae.type,
                f"latentdim={dim}_bs={bs}_lr={lr}_kldweight={kld_weight}_clsweight={cls_weight}",
            )

            # Get the most recent experiment version
            if args.version != -1:
                last_exp_no = args.version
            else:
                last_exp_no = get_latest_experiment_version(path)

            if last_exp_no == -1:
                print(f"Experiment directory {path} doesn't exist, skipping...")
                continue

            # Generate output directory
            config.vae.checkpoint = str(
                Path(path, f"version_{last_exp_no}", "checkpoint.pth")
            )
            output_path = Path(
                config.vae.checkpoint.replace(args.output_dir, "latent_analysis")
            ).parent
            output_path.mkdir(parents=True, exist_ok=True)
            print(f"Saving predictiveness analysis in {output_path}")

            file = open(output_path / "output_evaluate_predictiveness.txt", "wt")
            sys.stdout = file

            vae = ResnetVAE(
                input_size=config.data.img_size,
                input_channels=config.data.channels,
                latent_dim=config.vae.latent_dim,
                kld_weight=config.vae.kld_weight,
                cls_weight=config.vae.cls_weight,
                num_classes=config.data.num_classes,
            )

            # Load the model with checkpoint weights
            checkpoint = torch.load(config.vae.checkpoint)
            vae.load_state_dict(checkpoint["model_state_dict"])

            # Enable eval mode.
            vae.eval()
            vae = vae.to(device)

            evaluate_predictiveness(
                config=config,
                vae=vae,
                train_dl=train_dl,
                val_dl=val_dl,
                test_dl=test_dl,
                output_path=output_path,
                device=device,
            )

            file.close()
            sys.stdout = sys.__stdout__

        except Exception as e:
            print(e)
