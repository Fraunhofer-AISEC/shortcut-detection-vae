"""adapted from https://github.com/AntixK/PyTorch-VAE"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet50


class ResnetVAE(nn.Module):
    """
    Variational Autoencoder (VAE) with a ResNet backbone that adapts to variable input image size.
    It uses a ResNet-50 model as an encoder to extract features from the input image.
    The extracted features are then passed through fully connected layers to obtain the mean and log variance of the
    latent space distribution.
    The latent space is sampled using the 'reparametrization trick'. The sample is then passed through a decoder to reconstruct
    the input image.
    The VAE optionally includes a classification layer that predicts a class label from the latent representation.
    Args:
        input_size (int): The size of the input images (both width and height). Default is 128.
        latent_dim (int): The dimensionality of the latent space. Default is 128.
        input_channels (int): The number of input channels. Default is 3 (RGB images).
        kld_weight (float): The weight for the Kullback-Leibler divergence loss term. Default is 0.001.
        cls_weight (float): The weight for the classification loss term. Default is 0.0.
        hidden_dims (List[int]): The list of hidden dimensions for the decoder layers. If None, the default list is used,
            which is determined based on the input size.
        num_classes (int): The number of classes for classification. If None, no classification layer is used.
    Attributes:
        input_channels (int): The number of input channels.
        latent_dim (int): The dimensionality of the latent space.
        kld_weight (float): The weight for the Kullback-Leibler divergence loss term.
        cls_weight (float): The weight for the classification loss term.
        num_classes (int): The number of classes for classification.
        input_size (int): The size of the input images (both width and height).
        hidden_dims (List[int]): The list of hidden dimensions for the decoder layers.
    Methods:
        encode(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            Encodes the input tensor using the ResNet encoder.
        decode(x: torch.Tensor) -> torch.Tensor:
            Decodes the input tensor using the decoder.
        reparametrize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
            Reparametrizes the latent space using the mean and log variance.
        loss_fn(
            inputs: torch.Tensor,
            outputs: torch.Tensor,
            mu: torch.Tensor,
            log_var: torch.Tensor,
            pred: torch.Tensor,
            labels: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            Computes the total loss based on reconstruction loss, Kullback-Leibler divergence loss, and
            classification loss.
        forward(input: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
            Forward pass of the network.
        sample(num_samples: int) -> torch.Tensor:
            Generates random samples from the latent space distribution.
        reconstruct_images(test_sample: torch.Tensor) -> torch.Tensor:
            Reconstructs the input images.
    """

    def __init__(
        self,
        input_size: int = 128,
        latent_dim: int = 128,
        input_channels: int = 3,
        kld_weight: float = 0.001,
        cls_weight: float = 0.0,
        hidden_dims: Optional[List[int]] = None,
        num_classes: Optional[int] = None,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.kld_weight = kld_weight
        self.cls_weight = cls_weight
        self.num_classes = num_classes
        self.input_size = input_size

        self.hidden_dims = hidden_dims
        if hidden_dims is None:
            self.hidden_dims = []
            current_x = self.input_size
            p = 1
            s = 2
            k = 3
            h = 32  # First hidden layer channel
            while current_x >= 4:  # We start decoding with 4*4
                self.hidden_dims.append(h)
                current_x = (current_x + 2 * p - k) // s + 1  # Conv kernel size formula
                h *= 2

        # Encoder
        resnet = resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.encoder = nn.Sequential(*modules)

        self.fc_mu = nn.Linear(2048, latent_dim)
        self.fc_var = nn.Linear(2048, latent_dim)

        # Classifier
        self.classification_layer = nn.Linear(self.latent_dim, self.num_classes)

        # Decoder
        modules = []
        self.decoder_input = nn.Linear(latent_dim, self.hidden_dims[-1] * (4**2))
        self.hidden_dims.reverse()

        for i in range(len(self.hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        self.hidden_dims[i],
                        self.hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(self.hidden_dims[i + 1]),
                    nn.ReLU(),
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                self.hidden_dims[-1],
                self.hidden_dims[-1],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(self.hidden_dims[-1]),
            nn.ReLU(),
            nn.Conv2d(
                self.hidden_dims[-1],
                out_channels=self.input_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes the input tensor using the encoder.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Encoded mean and log variance tensors.
        """
        x = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(x)
        features = self.encoder(x)
        features = torch.flatten(features, start_dim=1)
        mu = self.fc_mu(features)
        log_var = self.fc_var(features)
        return mu, log_var

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decodes the input tensor using the decoder.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Decoded output tensor.
        """
        rec = self.decoder_input(x)
        rec = rec.view(-1, self.hidden_dims[0], 4, 4)
        rec = self.decoder(rec)
        rec = self.final_layer(rec)
        return rec

    def reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparametrizes the latent space using the mean and log variance.
        Args:
            mu (torch.Tensor): Mean tensor.
            logvar (torch.Tensor): Log variance tensor.
        Returns:
            torch.Tensor: Reparametrized tensor.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def loss_fn(
        self,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        pred: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the total loss based on reconstruction loss, Kullback-Leibler divergence loss, and classification loss.
        Args:
            inputs (torch.Tensor): Input tensor.
            outputs (torch.Tensor): Output tensor.
            mu (torch.Tensor): Mean tensor.
            log_var (torch.Tensor): Log variance tensor.
            pred (torch.Tensor): Prediction tensor.
            labels (torch.Tensor): Target labels.
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Total loss, Kullback-Leibler divergence loss, and
                reconstruction loss.
        """
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        )
        rec_loss = F.binary_cross_entropy(outputs, inputs, reduction="none")
        rec_loss = torch.mean(torch.sum(rec_loss, dim=(1, 2, 3)), dim=0)
        cls_loss = F.cross_entropy(pred, labels, reduction="mean")
        return (
            rec_loss + self.kld_weight * kld_loss + self.cls_weight * cls_loss,
            kld_loss,
            rec_loss,
        )

    def forward(
        self, input: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass of the network.
        Args:
            input (torch.Tensor): Input tensor.
            labels (torch.Tensor): Target labels.
        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: Total loss and list of tensors containing output, input,
                mean, log variance, latent tensor, Kullback-Leibler divergence loss, and reconstruction loss.
        """
        # ResNet was trained with normalization
        mu, log_var = self.encode(input)
        pred = self.classification_layer(mu)
        z = self.reparametrize(mu, log_var)
        output = self.decode(z)

        loss, kl_loss, rec_loss = self.loss_fn(input, output, mu, log_var, pred, labels)
        return loss, [output, input, mu, log_var, z, kl_loss, rec_loss]

    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Generates random samples from the latent space.
        Args:
            num_samples (int): Number of samples to generate.
        Returns:
            torch.Tensor: Generated samples.
        """
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(next(self.parameters()).device)
        samples = self.decode(z)
        return samples

    def reconstruct_images(self, test_sample: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs the input images.
        Args:
            test_sample (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Reconstructed images.
        """
        z, _ = self.encode(test_sample)
        predictions = self.decode(z)
        return predictions
