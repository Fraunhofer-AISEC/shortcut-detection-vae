import math
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.eval_utils import evaluate_classifier


def plot_model_output(model_output: torch.Tensor, path: str, dataset: str) -> None:
    """
    Plots the model's image outputs into a 2D grid and saves the plot.

    Args:
        model_output (torch.Tensor): The model's image outputs as a tensor.
        path (str): The path to save the plot.
        dataset (str): The dataset name.

    Returns:
        None
    """
    img = (
        torchvision.utils.make_grid(
            model_output,
            nrow=int(math.sqrt(model_output.shape[0])),
        )
        .permute(1, 2, 0)
        .numpy()
    )
    if dataset == "asv_spoof":
        img = img[:, :, 0]
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.savefig(path)
    plt.close()


def plot_model_reconstructions_samples(
    model: torch.nn.Module,
    X: torch.Tensor,
    path: Path,
    dataset: str,
    epoch: int,
    best: bool = False,
) -> None:
    """
    Plots model reconstructions and samples, and saves them as images.

    Args:
        model (torch.nn.Module): The VAE model.
        X (torch.Tensor): The input tensor.
        path (Path): The path to save the images.
        dataset (str): The dataset name.
        epoch (int): The current epoch number.
        best (bool, optional): Whether the provided model has the best weights. Defaults to False.

    Returns:
        None
    """
    reconstructions = model.reconstruct_images(X)

    if best:
        plot_path = path / f"reconstruction_best_{epoch}.png"
    else:
        plot_path = path / f"reconstruction_{epoch}.png"

    plot_model_output(
        model_output=reconstructions.cpu(), path=plot_path, dataset=dataset
    )

    samples = model.sample(num_samples=X.shape[0])

    if best:
        plot_path = path / f"sample_best_{epoch}.png"
    else:
        plot_path = path / f"sample_{epoch}.png"

    plot_model_output(model_output=samples.cpu(), path=plot_path, dataset=dataset)


def train_classifier(
    classifier: torch.nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    criterion: torch.nn.Module,
    device: str,
    epochs: int,
    path: str,
) -> Tuple[torch.nn.Module, float, float]:
    """
    Trains a classifier model using the provided data loaders and optimization settings.

    Args:
        classifier (nn.Module): The classifier model.
        train_dataloader (DataLoader): The training data loader.
        val_dataloader (DataLoader): The validation data loader.
        optimizer (optim.Optimizer): The optimizer for model training.
        scheduler (optim.lr_scheduler._LRScheduler): The scheduler for adjusting learning rate.
        criterion (nn.Module): The loss criterion.
        device (str): The device to run the training process on.
        epochs (int): The number of epochs to train for.
        path (str): The path to save the trained model.

    Returns:
        Tuple[nn.Module, float, float]: The trained classifier model, validation accuracy, and validation loss.
    """
    best_loss = float("inf")
    val_acc, val_loss = 0.0, 0.0

    for epoch in range(epochs):
        classifier.train()
        running_loss = 0
        n_correct = 0
        n = 0

        for X, y, _ in tqdm(train_dataloader):
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()

            y_pred = classifier(X)
            loss = criterion(y_pred, y)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_correct += (y_pred.argmax(1) == y).sum().item()
            n += len(y)

        train_loss = running_loss / len(train_dataloader)
        train_acc = n_correct / n
        print(f"Epoch {epoch}: train_loss={train_loss}, train_acc={train_acc}")

        if epoch % 5 == 0:
            classifier.eval()
            val_acc, val_loss = evaluate_classifier(
                classifier=classifier,
                dataloader=val_dataloader,
                device=device,
                criterion=criterion,
            )
            print(f"Epoch {epoch}: val_loss={val_loss}, val_acc={val_acc}")
            classifier.train()

            if val_loss < best_loss:
                torch.save(classifier.state_dict(), path / "classifier.pth")
                best_loss = val_loss

        if scheduler:
            scheduler.step()

    return classifier, val_acc, val_loss
