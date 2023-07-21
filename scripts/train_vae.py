"""
Train a VAE model and generate intermediate outputs.

This script trains a Variational Autoencoder (VAE) model using the provided command line arguments and configuration file.
It saves intermediate outputs, such as model checkpoints and evaluation results, to the output directory.

Command line arguments:
    --config_file (str): Path to the configuration file containing additional settings for the training process.
    --output_dir (str): Path to the directory where checkpoints, logs and intermediate outputs will be saved.
    --dataset_name (str): Name of the input dataset to be used for training.
    --device (str): CUDA visible device for running the training process.
    --lr (float or list of floats): Learning rate(s) to be used for training.
    --kld_weight (float or list of floats): KL divergence weight(s) for the VAE loss.
    --cls_weight (float or list of floats): Classifier weight(s) for the VAE loss.
    --latent_dim (int or list of ints): Dimensionality of the latent space.
    --batch_size (int or list of ints): Batch size(s) for training.
    --type (str): Specification of the VAE type.

Returns:
    None
"""

import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.optim import Adam
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from datasets.data_provider import get_dataloaders, get_datasets, get_transform
from models.resnet_vae import ResnetVAE
from utils.setup_utils import get_experiment_path, get_parser, setup_experiment
from utils.train_utils import plot_model_reconstructions_samples

warnings.filterwarnings("ignore")


def evaluate(model: torch.nn.Module, val_loader: DataLoader) -> float:
    """
    Evaluate the model on the validation data.

    Args:
        model (torch.nn.Module): The trained model.
        val_loader (DataLoader): DataLoader for the validation data.

    Returns:
        float: The average validation loss.

    """
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for X, y, _ in tqdm(val_loader, desc="Evaluating"):
            X = X.to(device)
            y = y.to(device)
            loss, _ = model(X, y)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    return val_loss


def train_step(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    lr_scheduler: _LRScheduler = None,
) -> float:
    """
    Trains the model on the training data.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (torch.utils.data.DataLoader): The data loader for the training data.
        optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler, optional):
            The learning rate scheduler for adjusting the learning rate during training.

    Returns:
        float: The average training loss.

    """
    model.train()
    train_loss = 0

    for X, y, _ in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()
        X = X.to(device)
        y = y.to(device)
        loss, _ = model(X, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if lr_scheduler:
            lr_scheduler.step()

    train_loss /= len(train_loader)
    return train_loss


def train(model, train_dl, val_dl, optimizer, lr_scheduler, config, writer, viz_img):
    best_loss = np.inf
    patience_count = 0
    for epoch in range(config.vae.epochs):
        print(f"Epoch: {epoch}")
        start_time = time.time()
        train_loss = train_step(
            model=model,
            train_loader=train_dl,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )
        val_loss = evaluate(model=model, val_loader=val_dl)
        end_time = time.time()
        print(f"Time: {end_time-start_time:.2f} s")

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/test", val_loss, epoch)

        if val_loss < best_loss:
            print(f"Train Loss: {train_loss} | Validation Loss: {val_loss}")
            print(f"Loss decreased from {best_loss} to {val_loss}")

            best_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                },
                path / "checkpoint.pth",
            )
            patience_count = 0

        else:
            patience_count += 1

        if patience_count == config.vae.patience:
            print(f"Early stopping after {epoch} epochs")
            checkpoint = torch.load(path / "checkpoint.pth")
            model.load_state_dict(checkpoint["model_state_dict"])
            epoch = checkpoint["epoch"]

            model.eval()
            plot_model_reconstructions_samples(
                model,
                viz_img,
                path,
                best=True,
                dataset=config.data.name,
                epoch=epoch,
            )
            model.train()

            break

        if epoch % config.vae.val_interval == 0:
            model.eval()
            # reconstructions = model.reconstruct_images(X)
            plot_model_reconstructions_samples(
                model, viz_img, path, dataset=config.data.name, epoch=epoch
            )
            model.train()


def plot_image_grid(viz_img, config, path):
    img_grid = (
        torchvision.utils.make_grid(
            viz_img.cpu(), nrow=int(math.sqrt(viz_img.shape[0]))
        )
        .permute(1, 2, 0)
        .numpy()
    )
    if config.data.name == "asv_spoof":
        img_grid = img_grid[:, :, 0]
    plt.figure(figsize=(10, 10))
    plt.imshow(img_grid)
    plt.savefig(path / "original.png")


if __name__ == "__main__":
    parser = get_parser(description="Read arguments for training VAE")
    config, args, device = setup_experiment(parser)

    train_dataset, val_dataset, test_dataset, config = get_datasets(
        config, train_transform=get_transform, test_transform=get_transform
    )

    # hyperparameter search over batch size, latent dim, learning rate and loss weights

    for bs, dim, lr, kld_weight, cls_weight in zip(
        args.batch_size, args.latent_dim, args.lr, args.kld_weight, args.cls_weight
    ):
        try:
            config.vae.batch_size = bs
            config.vae.latent_dim = dim
            config.vae.lr = lr
            config.vae.kld_weight = kld_weight
            config.vae.cls_weight = cls_weight

            train_dl, val_dl, test_dl = get_dataloaders(
                train_dataset, val_dataset, test_dataset, bs=config.vae.batch_size
            )

            # specify path
            path = Path(
                args.output_dir,
                config.data.name,
                config.vae.type,
                f"latentdim={dim}_bs={bs}_lr={lr}_kldweight={kld_weight}_clsweight={cls_weight}",
            )
            path = get_experiment_path(path)
            print(f"Saving experiment logs and output in {path}")

            # save config
            with open(path / "config.json", "w") as f:
                json.dump(config, f)

            # tensorboard
            writer = SummaryWriter(path)

            # get original val images
            dataiter = iter(val_dl)
            viz_img, y, p = next(dataiter)
            viz_img = viz_img[:25]
            viz_img = viz_img.to(device)

            # plot original image
            plot_image_grid(viz_img=viz_img, config=config, path=path)

            model = ResnetVAE(
                input_size=config.data.img_size,
                input_channels=config.data.channels,
                latent_dim=config.vae.latent_dim,
                kld_weight=config.vae.kld_weight,
                cls_weight=config.vae.cls_weight,
                num_classes=config.data.num_classes,
            )
            model = model.to(device)

            optimizer = Adam(model.parameters(), lr=lr)
            lr_scheduler = None

            train(
                model=model,
                train_dl=train_dl,
                val_dl=val_dl,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                config=config,
                writer=writer,
                viz_img=viz_img,
            )

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(e)
