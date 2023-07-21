import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from easydict import EasyDict


def get_config(config_path: str = "config.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config = EasyDict(config)
    return config


def get_experiment_path(path: str):
    """
    Creates and returns the path of the current experiment directory, taking into account the versioning of experiments.

    Args:
        path (str): Path to the experiment directory (created with version_0 if does not exist )

    Returns:
        str: The path of the created experiment directory.

    Example:
        >>> get_experiment_path("output/asv_spoof/ResnetVAE/latentdim=10_bs=32_lr=0.001_kldweight=1.25_clsweight=0.0")
        output/asv_spoof/ResnetVAE/latentdim=10_bs=32_lr=0.001_kldweight=1.25_clsweight=0.0/version_1
    """
    new_exp_path = path / "version_0"
    if path.exists():
        last_exp_no = get_latest_experiment_version(path)
        new_exp_path = path / f"version_{last_exp_no+1}"
    new_exp_path.mkdir(parents=True)
    return new_exp_path


def get_latest_experiment_version(exp_dir_path: Path):
    last_exp_no = -1
    existing_exp = sorted(list(exp_dir_path.glob("version_*")))
    if len(existing_exp) != 0:
        last_exp_no = int(existing_exp[-1].name.split("_")[-1])
    return last_exp_no


def seed_everything(seed: int = 42):
    """
    Set random seeds for reproducibility.

    Args:
        seed (int): The seed value to use for random number generation.

    Returns:
        None

    Note:
        This method sets the random seed for various libraries and modules to ensure reproducibility.
        It sets the seed for the `random`, `numpy`, `torch`, and `torch.cuda` modules, as well as configures
        `torch.backends.cudnn` for deterministic behavior.

    Example:
        >>> seed_everything(42)
    """
    random.seed(seed)
    os.environ["PYTHONASSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_parser(description="Read arguments for training/evaluating the VAE"):
    """
    Create an argument parser for configuring the program.

    Returns:
        argparse.ArgumentParser: The argument parser object.
    """
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "--dataset_name",
        type=str,
        help="name of the input dataset to be used for training",
    )

    parser.add_argument(
        "--device",
        type=str,
        help="CUDA visible device for running the training process",
    )

    parser.add_argument(
        "--lr",
        help="learning rate(s) to be used for training",
        nargs="+",
        default=[10e-3, 10e-4, 10e-5],
        type=float,
    )

    parser.add_argument(
        "--kld_weight",
        help="KL divergence weight(s) for the VAE loss",
        nargs="+",
        default=[100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001],
        type=float,
    )

    parser.add_argument(
        "--cls_weight",
        help="classifier weight(s) for the VAE loss, defaults to 0(recommended)",
        nargs="+",
        default=[0.0],
        type=float,
    )

    parser.add_argument(
        "--latent_dim",
        help="dimensionality of the latent space",
        nargs="+",
        default=[32],
        type=int,
    )

    parser.add_argument(
        "--batch_size",
        help="batch size(s) for training",
        nargs="+",
        default=[32],
        type=int,
    )

    parser.add_argument(
        "--type",
        help="specification of the VAE type",
        default="ResnetVAE",
        type=str,
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

    return parser


def setup_experiment(parser):
    args = parser.parse_args()

    config = get_config(args.config_file)
    # override some config parameters with corresponding commands line arguments
    config.data.name = args.dataset_name
    config.vae.type = args.type

    # Set up the necessary configurations and devices
    seed_everything(seed=config.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return config, args, device
