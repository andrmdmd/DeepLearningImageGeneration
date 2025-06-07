import torch
import torch.nn as nn
from diffusers.models import UNet2DModel

from configs import Config


class ClassicModel(nn.Module):
    def __init__(self, in_channels: int, base_dim: int, num_classes: int):
        super(ClassicModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, base_dim, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(base_dim, base_dim * 2, 5)
        self.fc1 = nn.Linear(base_dim * 2 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.flatten(start_dim=1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DCGANGenerator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        return self.main(input)


class DCGANDiscriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)

class VAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=64, hidden_dims=None, input_size=64):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]
        self.hidden_dims = hidden_dims

        # Encoder
        modules = []
        prev_dim = in_channels
        for h_dim in hidden_dims:
            modules.append(
                nn.Conv2d(prev_dim, h_dim, kernel_size=4, stride=2, padding=1)
            )
            modules.append(nn.ReLU())
            prev_dim = h_dim
        self.encoder = nn.Sequential(*modules)

        # Compute final feature map size after all conv layers
        # For 64x64 input and len(hidden_dims) layers, each halves the size
        self.final_feat_size = input_size // (2 ** len(hidden_dims))
        linear_in_features = hidden_dims[-1] * self.final_feat_size * self.final_feat_size

        self.fc_mu = nn.Linear(linear_in_features, latent_dim)
        self.fc_logvar = nn.Linear(linear_in_features, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, linear_in_features)

        # Decoder
        modules = []
        hidden_dims_rev = hidden_dims[::-1]
        prev_dim = hidden_dims_rev[0]
        for h_dim in hidden_dims_rev[1:]:
            modules.append(
                nn.ConvTranspose2d(prev_dim, h_dim, kernel_size=4, stride=2, padding=1)
            )
            modules.append(nn.ReLU())
            prev_dim = h_dim
        modules.append(
            nn.ConvTranspose2d(prev_dim, in_channels, kernel_size=4, stride=2, padding=1)
        )
        modules.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*modules)

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(z.size(0), self.hidden_dims[-1], self.final_feat_size, self.final_feat_size)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

def build_model(cfg: Config) -> ClassicModel:
    return ClassicModel(
        cfg.model.in_channels, cfg.model.base_dim, cfg.model.num_classes
    )

def build_generator(cfg: Config):
    return DCGANGenerator(nz=cfg.training.dcgan.nz, ngf=cfg.training.dcgan.ngf, nc=cfg.data.in_channels)

def build_discriminator(cfg: Config):
    return DCGANDiscriminator(nc=cfg.data.in_channels, ndf=cfg.training.dcgan.ndf)

def build_vae(cfg: Config):
    return VAE(
        in_channels=cfg.data.in_channels,
        latent_dim=cfg.training.vae.latent_dim,
        hidden_dims=cfg.training.vae.hidden_dims if hasattr(cfg.model, "hidden_dims") else None,
    )

def build_unet2d_model(cfg: Config) -> UNet2DModel:
    return UNet2DModel(
        sample_size=cfg.data.image_size,
        in_channels=cfg.data.in_channels,
        out_channels=cfg.model.out_channels,
        layers_per_block=2,
        block_out_channels=(
            cfg.model.base_dim,
            cfg.model.base_dim,
            cfg.model.base_dim * 2,
            cfg.model.base_dim * 2,
            cfg.model.base_dim * 4,
            cfg.model.base_dim * 4,
        ),
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",  # a regular ResNet upsampling block
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
