"""adapted from https://www.kaggle.com/code/maunish/training-vae-on-imagenet-pytorch"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NonClosedFormVAE(nn.Module):
    """
    Variational Autoencoder (VAE) with non-closed form loss computation.
    This VAE architecture consists of an encoder and a decoder.
    The encoder maps input images to the latent space, and the decoder reconstructs the images from the latent space.
    The latent space is sampled using the reparametrization trick.
    Args:
        input_size (int): Size of the input images (assumed to be square).
        latent_dim (int): Dimensionality of the latent space.
        input_channels (int): Number of input channels in the images.
        kld_weight (float): Weight for the KL divergence term in the VAE loss.
    Attributes:
        input_size (int): Size of the input images.
        input_channels (int): Number of input channels in the images.
        latent_dim (int): Dimensionality of the latent space.
        kld_weight (float): Weight for the KL divergence term in the VAE loss.
        bottleneck_size (int): Size of the bottleneck layer before the latent space.
    """

    def __init__(
        self,
        input_size: int = 128,
        latent_dim: int = 128,
        input_channels: int = 3,
        kld_weight: float = 0.1,
    ):
        super().__init__()

        self.input_size = input_size
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.kld_weight = kld_weight

        self.bottleneck_size = input_size
        for _ in range(5):
            self.bottleneck_size = (self.bottleneck_size - 3) // 2 + 1

        # Encoder layers
        self.conv1 = nn.Conv2d(self.input_channels, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * (2**2), 2 * self.latent_dim)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.scale = nn.Parameter(torch.tensor([0.0]))

        # Decoder layers
        self.fc2 = nn.Linear(self.latent_dim, 512 * (self.bottleneck_size**2))
        self.conv6 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2)
        self.conv7 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2)
        self.conv8 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2)
        self.conv9 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2)
        self.conv10 = nn.ConvTranspose2d(
            32, self.input_channels, kernel_size=3, stride=2
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the input images to the latent space.
        Args:
            x (torch.Tensor): Input images.
        Returns:
            torch.Tensor: Mean and log-variance of the approximate posterior distribution.
        """
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = self.pool(self.relu(self.conv5(x)))
        x = self.flatten(x)
        x = self.fc1(x)
        mean, logvar = torch.split(x, self.latent_dim, dim=1)
        return mean, logvar

    def decode(self, eps: torch.Tensor) -> torch.Tensor:
        """
        Decode the latent space representations to reconstruct the images.
        Args:
            eps (torch.Tensor): Latent space representations.
        Returns:
            torch.Tensor: Reconstructed images.
        """
        x = self.relu(self.fc2(eps))
        x = torch.reshape(x, (-1, 512, self.bottleneck_size, self.bottleneck_size))
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x))
        x = self.relu(self.conv8(x))
        x = self.relu(self.conv9(x))
        x = self.conv10(x)
        x = F.interpolate(
            x,
            size=(self.input_size, self.input_size),
        )
        return x

    def reparametrize(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """
        Reparametrize the latent space representations using the reparametrization trick.
        Args:
            mean (torch.Tensor): Mean of the approximate posterior distribution.
            std (torch.Tensor): Standard deviation of the approximate posterior distribution.
        Returns:
            torch.Tensor: Reparametrized latent space representations.
        """
        q = torch.distributions.Normal(mean, std)
        return q.rsample()

    def kl_loss(
        self, z: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the KL divergence between the approximate posterior and the prior distribution.
        Args:
            z (torch.Tensor): Latent space representations.
            mean (torch.Tensor): Mean of the approximate posterior distribution.
            std (torch.Tensor): Standard deviation of the approximate posterior distribution.
        Returns:
            torch.Tensor: KL divergence loss.
        """
        p = torch.distributions.Normal(torch.zeros_like(mean), torch.ones_like(std))
        q = torch.distributions.Normal(mean, torch.exp(std / 2))

        log_pz = p.log_prob(z)
        log_qzx = q.log_prob(z)

        kl_loss = log_qzx - log_pz
        kl_loss = kl_loss.sum(-1)
        return kl_loss

    def gaussian_likelihood(
        self, inputs: torch.Tensor, outputs: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the negative log-likelihood of the reconstructed images under the Gaussian distribution.
        Args:
            inputs (torch.Tensor): Input images.
            outputs (torch.Tensor): Reconstructed images.
            scale (torch.Tensor): Scaling factor for the Gaussian distribution.
        Returns:
            torch.Tensor: Gaussian likelihood loss.
        """
        dist = torch.distributions.Normal(outputs, torch.exp(scale))
        log_pxz = dist.log_prob(inputs)
        return log_pxz.sum(dim=(1, 2, 3))

    def loss_fn(
        self,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        z: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the VAE loss.
        Args:
            inputs (torch.Tensor): Input images.
            outputs (torch.Tensor): Reconstructed images.
            z (torch.Tensor): Latent space representations.
            mean (torch.Tensor): Mean of the approximate posterior distribution.
            std (torch.Tensor): Standard deviation of the approximate posterior distribution.
        Returns:
            torch.Tensor: Total VAE loss, KL divergence loss, and negative log-likelihood loss.
        """
        kl_loss = self.kl_loss(z, mean, std)
        rec_loss = self.gaussian_likelihood(inputs, outputs, self.scale)
        return (
            torch.mean(self.kld_weight * kl_loss - rec_loss),
            torch.mean(kl_loss),
            torch.mean(-rec_loss),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the VAE.
        Args:
            inputs (torch.Tensor): Input images.
        Returns:
            torch.Tensor: Total VAE loss and other relevant information.
        """
        mean, logvar = self.encode(inputs)
        std = torch.exp(logvar / 2)
        std = torch.clamp(std, min=1e-8)
        z = self.reparametrize(mean, std)
        outputs = self.decode(z)

        loss, kl_loss, rec_loss = self.loss_fn(inputs, outputs, z, mean, std)
        return loss, (outputs, z, mean, std, kl_loss, rec_loss)

    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Generate samples from the VAE by sampling from the prior distribution.
        Args:
            num_samples (int): Number of samples to generate.
        Returns:
            torch.Tensor: Generated samples.
        """
        p = torch.distributions.Normal(
            torch.zeros((num_samples, self.latent_dim)),
            torch.ones((num_samples, self.latent_dim)),
        )
        z = p.rsample().to(next(self.parameters()).device)
        outputs = self.decode(z)
        dist = torch.distributions.Normal(outputs, torch.exp(self.scale))
        img = dist.rsample()
        return img

    def reconstruct_images(self, test_sample: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct images using the VAE.
        Args:
            test_sample (torch.Tensor): Input images to be reconstructed.
        Returns:
            torch.Tensor: Reconstructed images.
        """
        z, _ = self.encode(test_sample)
        predictions = self.decode(z)
        return predictions
