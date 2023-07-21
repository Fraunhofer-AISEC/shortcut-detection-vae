from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchvision
from easydict import EasyDict
from PIL import Image
from scipy.stats import entropy, wasserstein_distance
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.data_provider import get_transform


def compute_predictiveness_per_dimension(classifier, config):
    predictiveness = []
    for dim in range(config.vae.latent_dim):
        weights = torch.abs(classifier.fc.weight[:, dim])
        pred_dim = weights.sum()
        predictiveness.append(pred_dim.item())

    print("Predictiveness sorted: ", np.sort(predictiveness)[::-1])
    print("Dimensions sorted: ", np.argsort(predictiveness)[::-1])


def evaluate_classifier(classifier, dataloader, device, criterion):
    classifier.eval()
    accuracy = 0
    f1 = 0
    loss = 0
    n_correct = 0
    n = 0
    for X, y, _ in tqdm(dataloader):
        X, y = X.to(device), y.to(device)

        with torch.no_grad():
            y_pred = classifier(X)

        loss += criterion(y_pred, y).item()
        n_correct += (y_pred.argmax(1) == y).sum().item()
        n += len(y)
        f1 += f1_score(
            y_true=y.cpu().detach(),
            y_pred=y_pred.argmax(dim=1).cpu().detach(),
            average="micro",
        )
    loss = loss / len(dataloader)
    accuracy = n_correct / n
    f1 = f1 / len(dataloader)
    return accuracy, loss


def examine_importance_of_latent_dims(classifier, cls, k, verbose):
    dim_argmax = classifier.fc.weight[cls].topk(k, largest=True).indices.cpu().numpy()
    dim_argmin = classifier.fc.weight[cls].topk(k, largest=False).indices.cpu().numpy()

    if verbose:
        print(f"Class: {cls}")
        print(f"argmax: {dim_argmax}")
        print(f"argmin: {dim_argmin}")

    return dim_argmin, dim_argmax


def retrieve_latent_code(
    model: torch.nn.Module, device: Union[str, torch.device], dataloader: DataLoader
) -> pd.DataFrame:
    """
    Retrieve the latent codes from a VAE-like model for the provided dataset and store them in a pandas dataframe.

    Args:
        model (VAE-like model): The VAE-like model to retrieve the latent codes from.
        device (str or torch.device): The device to run the model on.
        dataloader (torch.utils.data.DataLoader): The dataloader for the dataset.

    Returns:
        pandas.DataFrame: The pandas dataframe containing the retrieved latent codes, along with their associated labels and paths.
    """
    df = None

    for X, y, p in tqdm(dataloader):
        X, y = X.to(device), y.to(device)

        with torch.no_grad():
            mu, log_var = model.encode(X.float())

        df_z = pd.DataFrame(mu.cpu()).astype("float")
        df_y = pd.DataFrame(y.cpu(), columns=["y"]).astype("int")
        df_p = pd.DataFrame(p, columns=["path"]).astype("string")

        df_batch = pd.concat([df_y, df_z, df_p], axis=1)
        df = pd.concat([df, df_batch], axis=0)

    return df


def compute_dimensionwise_wasserstein_distance(df, config):
    distance_matrix = np.zeros(
        (config.vae.latent_dim, config.data.num_classes, config.data.num_classes)
    )
    for dim in tqdm(range(config.vae.latent_dim)):
        for cls_1 in range(config.data.num_classes):
            # Construct empirical PDF with these two samples
            for cls_2 in range(config.data.num_classes):
                # wasserstein distance takes observations as input
                val1 = df[df["y"] == cls_1][dim].to_numpy()
                val2 = df[df["y"] == cls_2][dim].to_numpy()
                distance_matrix[dim, cls_1, cls_2] = wasserstein_distance(val1, val2)

    distances = distance_matrix.max(axis=(1, 2))

    return distances


def compute_dimensionwise_kld(df: pd.DataFrame, config: EasyDict) -> List[float]:
    """
    Compute the Kullback-Leibler (KL) divergence for each dimension in the dataset from the normal distribution.

    Args:
        df (pd.DataFrame): The pandas dataframe containing the latent variable values.
        config (EasyDict): Configuration settings.

    Returns:
        List[float]: List of KL divergences for each dimension in the dataset.
    """
    min, max = -10, 10
    dim_klds = []

    for dim in tqdm(range(config.vae.latent_dim)):
        hist_Q = np.histogram(
            df[dim].to_numpy(),
            bins=100,
            range=(min, max),
            density=True,
        )

        hist_P = np.histogram(
            np.random.normal(size=df.shape[0]),
            bins=100,
            range=(min, max),
            density=True,
        )

        hist_Q = hist_Q[0] + 1e-8
        hist_Q = hist_Q / hist_Q.sum()
        hist_P = hist_P[0] + 1e-8
        hist_P = hist_P / hist_P.sum()

        kld = entropy(hist_Q, hist_P)
        dim_klds.append(kld)

    return dim_klds


def examine_change_of_latent_dim(
    model: torch.nn.Module,
    train_dl: torch.utils.data.DataLoader,
    df: pd.DataFrame,
    latent_dims: int,
    steps: int,
    display_original_img: bool,
    path: Path,
    dataset: str,
    num_classes: int,
    device: torch.device,
):
    """
    Perform a latent space traversal of a VAE-like model by varying each dimension of the latent space
    and generating reconstructed images for visualization. The reconstructed images are saved in the provided path.

    Args:
        model (torch.nn.Module): The VAE-like model.
        train_dl (torch.utils.data.Dataloader): The train dataloader.
        df (pd.DataFrame): The pandas dataframe containing the values of the latent dimensions for the entire dataset.
        latent_dims (int): The number of latent dimensions in the model.
        steps (int): The number of steps for the latent space traversal.
        display_original_img (bool): Flag indicating whether to display the original images in the subplot.
        path (Path): The path to save the generated images.
        dataset (str): The dataset name.
        num_classes (int): number of classes
        device (torch.device): Device to run the evaluation on

    Returns:
        None: The function saves the generated images to the specified path.
    """

    # Choose one random image per class
    images_per_class = 1
    images = []
    for cls in range(num_classes):
        cls_cnt = 0
        for X, y, _ in tqdm(train_dl):
            X, y = X.to(device), y.to(device)
            idx = torch.where(y == cls)[0]
            if len(idx) > 0:
                images.append(X[idx[0]])
                cls_cnt += 1
            if cls_cnt == images_per_class:
                cls_cnt = 0
                break

    images = torch.stack(images, dim=0)

    indices = []
    for i in range(images_per_class):
        for cls in range(num_classes):
            indices.append(cls * images_per_class + i)

    images = images[indices]

    # iterate over dimensions, perform changes
    for dim_changed in tqdm(range(latent_dims)):
        columns = steps + 1 if display_original_img else steps
        fig, axs = plt.subplots(
            images.shape[0], columns, figsize=(columns, images.shape[0])
        )

        if display_original_img:
            # display original image in the first column
            for i in range(images.shape[0]):
                img = images[i].cpu().permute(1, 2, 0)
                if dataset == "asv_spoof":
                    img = img[:, :, 0]
                axs[i, 0].imshow(img)
                axs[i, 0].axis("off")

        for i, val in enumerate(
            np.linspace(
                start=df[dim_changed].min(), stop=df[dim_changed].max(), num=steps
            )
        ):
            idx = i + 1 if display_original_img else i
            with torch.no_grad():
                mu, logvar = model.encode(images.float())
                # Perform changes to the latent dimension
                mu[:, dim_changed] = val
                output = model.decode(mu)

            for j in range(images.shape[0]):
                img = output[j].cpu().permute(1, 2, 0)
                if dataset == "asv_spoof":
                    img = img[:, :, 0]
                axs[j, idx].imshow(img)
                axs[j, idx].axis("off")

        plt.subplots_adjust(wspace=None, hspace=None)

        # # Adjust the layou, rearrange the axes for no overlap and save the images
        fig.tight_layout()
        plt.savefig(
            path / f"latent_traversal_dim={dim_changed}.jpg", bbox_inches="tight"
        )
        plt.close()


def visualize_distribution_per_latent_dim(
    df: pd.DataFrame, latent_dims: int, num_classes: int, path: Path, config: EasyDict
) -> None:
    """
    Visualize the class-wise distribution of latent values for each dimension in the dataset
    by plotting KDE plots for each class.

    Args:
        df (pd.DataFrame): The pandas DataFrame containing the latent values.
        latent_dims (int): The number of latent dimensions.
        num_classes (int): The number of classes in the dataset.
        path (Path): The path to save the generated plots.
        config (EasyDict): The configuration dictionary.

    Returns:
        None: The function saves the generated plots to the specified path.
    """
    for dim in tqdm(range(latent_dims)):
        plt.figure(figsize=(8, 4))

        for i in range(num_classes):
            # Define labels for the lemon dataset
            if config.data.name == "lemon_quality":
                if i == 0:
                    label = "good"
                elif i == 1:
                    label = "bad"
                elif i == 2:
                    label = "empty"
                else:
                    raise ValueError(f"No label specified for class {i}.")
            else:
                label = i

            # Plot KDE plot for each class
            fig = sns.kdeplot(
                x=dim,
                data=df[df["y"] == i].reset_index(),
                label=label,
            )
            fig.set(xlabel=None)
            fig.set(ylabel=None)

        plt.legend()
        plt.savefig(path / f"kde_dim={dim}.jpg", bbox_inches="tight")
        plt.tight_layout()
        plt.close()


def visualize_extreme_inputs(
    df: pd.DataFrame, config: EasyDict, path: Path, row: int, col: int
) -> None:
    """
    Plot the k (specified by row, col) images in the dataset corresponding to the k min and max
    values of each latent dimension.

    Args:
        df (pd.DataFrame): The pandas DataFrame containing the latent values.
        config (EasyDict): The configuration dictionary.
        path (Path): The path to save the generated images.
        row (int): The number of rows in the output image grid.
        col (int): The number of columns in the output image grid.

    Returns:
        None: The function saves the generated images to the specified path.
    """
    transform = get_transform(config.data.img_size)
    num = row * col

    for dim in tqdm(range(config.vae.latent_dim)):
        # Get paths corresponding to k min and max values of the dimension
        min_paths = df.sort_values(dim)[:num]["path"].to_numpy()
        max_paths = df.sort_values(dim)[-num:]["path"].to_numpy()

        # Generate and save images for k min values of the dimension
        images = []
        for p in min_paths:
            img = Image.open(p).convert("RGB")
            img = transform(img)
            images.append(img)

        images = torch.stack(images)
        images = torchvision.utils.make_grid(images, nrow=col).permute(1, 2, 0).numpy()

        plt.figure(figsize=(col, row))
        plt.axis("off")
        plt.imshow(images)
        plt.savefig(path / f"images_min_dim={dim}.png", bbox_inches="tight")
        plt.close()

        # Generate and save images for k min values of the dimension
        images = []
        for p in max_paths:
            img = Image.open(p).convert("RGB")
            img = transform(img)
            images.append(img)

        images = torch.stack(images)
        images = torchvision.utils.make_grid(images, nrow=col).permute(1, 2, 0).numpy()

        plt.figure(figsize=(col, row))
        plt.axis("off")
        plt.imshow(images)
        plt.savefig(path / f"images_max_dim={dim}.png", bbox_inches="tight")
        plt.close()
