"""adapted from https://github.com/AntixK/PyTorch-VAE"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaVAE(nn.Module):
    """
    Closed-Form Variational Autoencoder with optional classifier loss.
    Args:
        latent_dim (int): Dimensionality of the latent space. Default is 128.
        input_channels (int): Number of input channels. Default is 3.
        kld_weight (float): Weight for the Kullback-Leibler divergence loss. Default is 0.001.
        cls_weight (float): Weight for the classifier loss. Default is 0.0.
        hidden_dims (List[int]): List of dimensions for the hidden layers. Default is None.
        num_classes (int): Number of classes for classification. Default is None.
    Attributes:
        input_channels (int): Number of input channels.
        latent_dim (int): Dimensionality of the latent space.
        kld_weight (float): Weight for the Kullback-Leibler divergence loss.
        cls_weight (float): Weight for the classifier loss.
        num_classes (int): Number of classes for classification.
        encoder (nn.Sequential): Encoder module.
        fc_mu (nn.Linear): Linear layer for mean of the latent space.
        fc_var (nn.Linear): Linear layer for log variance of the latent space.
        classification_layer (nn.Linear): Linear layer for classification.
        decoder_input (nn.Linear): Linear layer for input to the decoder.
        decoder (nn.Sequential): Decoder module.
        final_layer (nn.Sequential): Final layer of the decoder.
    """

    def __init__(
        self,
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

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Encoder
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        input_channels,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                )
            )
            input_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * (4**2), latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * (4**2), latent_dim)

        self.classification_layer = nn.Linear(self.latent_dim, self.num_classes)

        # Decoder
        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * (4**2))

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.ReLU(),
                )
            )
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1],
                hidden_dims[-1],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.ReLU(),
            nn.Conv2d(
                hidden_dims[-1],
                out_channels=self.input_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode the input image tensor.
        Args:
            x (torch.Tensor): Input image tensor.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the mean and log variance of the latent space.
        """
        features = self.encoder(x)
        features = torch.flatten(features, start_dim=1)
        mu = self.fc_mu(features)
        log_var = self.fc_var(features)
        return mu, log_var

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode the input tensor.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Decoded tensor.
        """
        rec = self.decoder_input(x)
        rec = rec.view(-1, 512, 4, 4)
        rec = self.decoder(rec)
        rec = self.final_layer(rec)
        return rec

    def reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparametrize the latent space.
        Args:
            mu (torch.Tensor): Mean of the latent space.
            logvar (torch.Tensor): Log variance of the latent space.
        Returns:
            torch.Tensor: Reparameterized latent space tensor.
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
        Calculate the total loss for the model.
        Args:
            inputs (torch.Tensor): Input images.
            outputs (torch.Tensor): Decoded images.
            mu (torch.Tensor): Mean of the latent space.
            log_var (torch.Tensor): Log variance of the latent space.
            pred (torch.Tensor): Predicted class probabilities.
            labels (torch.Tensor): True class labels.
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing the total loss,
                Kullback-Leibler divergence loss, and reconstruction loss.
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
        Forward pass of the model.
        Args:
            input (torch.Tensor): Input images.
            labels (torch.Tensor): True class labels.
        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: Tuple containing the total loss and a list
                of tensors including the output images, input images, mean, log variance, latent space,
                Kullback-Leibler divergence loss, and reconstruction loss.
        """
        input = input.float()
        mu, log_var = self.encode(input)
        pred = self.classification_layer(mu)
        z = self.reparametrize(mu, log_var)
        output = self.decode(z)

        loss, kl_loss, rec_loss = self.loss_fn(input, output, mu, log_var, pred, labels)
        return loss, [output, input, mu, log_var, z, kl_loss, rec_loss]

    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Generate samples from the model.
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
        Reconstruct input images using the model.
        Args:
            test_sample (torch.Tensor): Input images to reconstruct.
        Returns:
            torch.Tensor: Reconstructed images.
        """
        z, _ = self.encode(test_sample)
        predictions = self.decode(z)
        return predictions
