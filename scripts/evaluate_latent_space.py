"""
Script for evaluating the latent space of VAE models.

This script performs various analyses and visualizations on the latent space of VAE models. It loads model checkpoints, samples images, computes latent encodings for train data instances, identifies dimensions with large KL divergence, computes Wasserstein distances between latent dimension distributions, visualizes class-wise latent dimension distributions, performs latent traversal on the dataset, and visualizes images with minimum and maximum values per latent dimension.

The outputs generated by this script provide insights and help in identifying important features in the latent space.

Usage:
    python evaluate_latent_space.py [--dataset_name DATASET_NAME] [--device DEVICE] [--lr LR [LR ...]] [--kld_weight KLD_WEIGHT [KLD_WEIGHT ...]] [--cls_weight CLS_WEIGHT [CLS_WEIGHT ...]] [--latent_dim LATENT_DIM [LATENT_DIM ...]] [--batch_size BATCH_SIZE [BATCH_SIZE ...]] [--type TYPE] [--config_file CONFIG_FILE] [--output_dir OUTPUT_DIR] [-v VERSION]

Arguments:
    --dataset_name (str): Name of the input dataset for evaluation.
    --device (str): CUDA visible device for running the evaluation process.
    --lr (List[float]): Learning rate(s) used for training the VAE.
    --kld_weight (List[float]): KL divergence weight(s) for the VAE loss.
    --cls_weight (List[float]): Classifier weight(s) for the VAE loss. Defaults to 0 (recommended).
    --latent_dim (List[int]): Dimensionality of the latent space.
    --batch_size (List[int]): Batch size(s) for training.
    --type (str): Specification of the VAE type.
    --config_file (str): Path to the configuration file containing additional settings for the training process.
    --output_dir (str): Path to the directory where checkpoints, logs, and intermediate outputs are saved.
    -v, --version (int): The version number of the experiment to perform evaluation (only relevant for evaluation). Defaults to -1.

"""

import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from easydict import EasyDict
from torch.utils.data import DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from datasets.data_provider import get_dataloaders, get_datasets, get_transform
from models.resnet_vae import ResnetVAE
from utils.setup_utils import (
    get_latest_experiment_version,
    get_parser,
    setup_experiment,
)
from utils.eval_utils import (
    compute_dimensionwise_kld,
    compute_dimensionwise_wasserstein_distance,
    examine_change_of_latent_dim,
    retrieve_latent_code,
    visualize_distribution_per_latent_dim,
    visualize_extreme_inputs,
)
from utils.train_utils import plot_model_output

# Disable warnings and set figure DPI
warnings.filterwarnings("ignore")
plt.rcParams["figure.dpi"] = 300


def evaluate_latent_space(
    config: EasyDict, model, train_dl: DataLoader, output_path, device
) -> None:
    """
    Evaluate the latent space of VAE models.
    Identify latent dimensions with largest maximum pairwise Wasserstein distance (MPWD).
    Visualize latent space traversal for each dimension.

    Args:
        config (EasyDict): The configuration dictionary.
        args (argparse.Namespace): The parsed command-line arguments.

    Returns:
        None: The function saves the results of the latent analysis.
    """

    # Sample and save generated images
    img = model.sample(16).cpu()
    img_path = output_path / "sample.png"
    plot_model_output(img, path=img_path, dataset=config.data.name)
    img = model.sample(16)
    img = torchvision.utils.make_grid(img.cpu()).permute(1, 2, 0).numpy()
    if config.data.name == "asv_spoof":
        img = img[:, :, 0]

    plt.figure(figsize=(15, 15))
    plt.imshow(img)
    plt.savefig(output_path / "sample.png")
    plt.close()

    # Retrieve latent encodings for all train instances
    print("Extracting latent encoding for all train instances.")
    df = retrieve_latent_code(model=model, device=device, dataloader=train_dl)

    # Identify dimensions with the largest KL divergence
    print("Identify dimensions with largest KL divergence")
    dim_klds = compute_dimensionwise_kld(df.drop(columns=["path"]), config)
    klds_sorted = np.sort(dim_klds)[::-1]
    dims_sorted = np.argsort(dim_klds)[::-1]
    print("KLDs max-min:", klds_sorted)
    print("Dimensions (KLD) max-min:", dims_sorted)

    # Compute Wasserstein distance between class distributions per latent dimension
    print(
        "Compute wasserstein distance between class distributions per latent dimension."
    )
    distances = compute_dimensionwise_wasserstein_distance(
        df.drop(columns=["path"]), config
    )
    print("distances.shape", distances.shape)
    wsdist_sorted = np.sort(distances)[::-1]
    dim_sorted = np.argsort(distances)[::-1]
    print("Wasserstein distance max-min: ", wsdist_sorted)
    print("Dimensions (WS distance) max-min: ", dim_sorted)

    # Visualize latent distributions associated with target classes
    print("Visualize latent distributions associated with target classes.")
    visualize_distribution_per_latent_dim(
        df=df,
        latent_dims=config.vae.latent_dim,
        num_classes=config.data.num_classes,
        path=output_path,
        config=config,
    )

    # Visualize latent traversal per dimension
    print("Visualize latent traversal per dimension.")
    steps = 6
    display_original_img = False
    examine_change_of_latent_dim(
        model=model,
        train_dl=train_dl,
        df=df,
        latent_dims=config.vae.latent_dim,
        steps=steps,
        display_original_img=display_original_img,
        path=output_path,
        dataset=config.data.name,
        num_classes=config.data.num_classes,
        device=device,
    )

    # Visualize images corresponding to min and max values per latent dimension
    print("Visualize images corresponding to min and max values per latent dimension.")
    visualize_extreme_inputs(df, config, path=output_path, row=3, col=9)


if __name__ == "__main__":
    parser = get_parser(
        description="Read arguments for statistical analysis and latent traversal of VAE"
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

    # Perform evalution on all experiments determined by the combination of provided parameters.
    for dim, bs, lr, kld_weight, cls_weight in zip(
        args.latent_dim, args.batch_size, args.lr, args.kld_weight, args.cls_weight
    ):
        try:
            config.vae.batch_size = bs
            config.vae.latent_dim = dim
            config.vae.lr = lr
            config.vae.kld_weight = kld_weight
            config.vae.cls_weight = cls_weight

            train_dl, _, _ = get_dataloaders(
                train_dataset, val_dataset, test_dataset, bs=config.vae.batch_size
            )

            path = Path(
                args.output_dir,
                config.data.name,
                config.vae.type,
                f"latentdim={config.vae.latent_dim}_bs={bs}_lr={lr}_kldweight={kld_weight}_clsweight={cls_weight}",
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
            print(f"Saving latent analysis in {output_path}")

            file = open(output_path / "output_latent_analysis.txt", "wt")
            sys.stdout = file

            model = ResnetVAE(
                input_size=config.data.img_size,
                input_channels=config.data.channels,
                latent_dim=config.vae.latent_dim,
                kld_weight=config.vae.kld_weight,
                cls_weight=config.vae.cls_weight,
                num_classes=config.data.num_classes,
            )

            # Load the model with checkpoint weights
            checkpoint = torch.load(config.vae.checkpoint)
            model.load_state_dict(checkpoint["model_state_dict"])

            # Enable eval mode.
            model.eval()
            model = model.to(device)

            evaluate_latent_space(
                config=config,
                model=model,
                train_dl=train_dl,
                output_path=output_path,
                device=device,
            )

            file.close()
            sys.stdout = sys.__stdout__

        except Exception as e:
            print(f"Exception occured: {e}")
            continue
