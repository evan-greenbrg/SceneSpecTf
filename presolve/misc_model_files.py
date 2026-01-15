import torch
from torch import nn
from torch.nn.utils import spectral_norm


class CubeToVector(nn.Module):
    def __init__(self, nbands: int, nblocks:int = 3):
        super().__init__()
        self.nblocks = nblocks
        self.conv_block = ConvBlock(nbands, kernel_size=256)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        for i in range(self.nblocks):
            x = self.conv_block(x)

        x = self.pool(x)

        return x


class ConvDecoder(nn.Module):
    def __init__(self, dim_input: int, nbands: int,
                 rows_output: int, cols_output: int,
                 nblocks: int):
        """
        rows_output and rows_input are divisible by 8
        """
        super().__init__()

        _reduc = 2**(2 * nblocks)
        self.rows_proj = round(rows_output / _reduc)
        self.cols_proj = round(cols_output / _reduc)

        self.head = nn.Linear(
            dim_input, 
            nbands * self.rows_proj * self.cols_proj
        )

        self.upsample = nn.ConvTranspose2d(nbands, nbands, kernel_size=2, stride=2, padding=0)
        self.bn = nn.BatchNorm2d(nbands)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.final = nn.Conv2d(nbands, nbands, kernel_size=1)

        layers = []
        for i in range(nblocks * 2):
            layers.append(self.upsample)
            layers.append(self.bn)
            layers.append(self.act)

        self.upsample = nn.Sequential(*layers)


    def forward(self, x):
        x = self.head(x)
        x = x.view(-1, nbands, self.rows_proj, self.cols_proj)
        x = self.upsample(x)
        x = self.final(x)
        # x = torch.sigmoid(x)

        return x


class DualBranchDecoder(nn.Module):
    def __init__(self, dim_input: int, nbands: int,
                 rows_output: int, cols_output: int,
                 heads: int, nblocks: int):
        """
        rows_output and rows_input are divisible by 8
        """
        super().__init__()
        self.heads = heads
        self.dim_input = dim_input

        _reduc = 2**(2 * nblocks)
        self.rows_proj = round(rows_output / _reduc)
        self.cols_proj = round(cols_output / _reduc)

        self.head = nn.Linear(
            dim_input, 
            nbands * self.rows_proj * self.cols_proj
        )

        self.upsample = nn.ConvTranspose2d(
            nbands, nbands, 
            kernel_size=2, stride=2, padding=0
        )
        self.bn = nn.BatchNorm2d(nbands)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.final = nn.Conv2d(nbands, nbands, kernel_size=1)

        self.dual_upsample = nn.Sequential(
            self.upsample,
            self.bn,
            self.act,
            self.upsample,
            self.bn,
            self.act
        )

        layers = []
        for i in range((nblocks - 1) * 2):
            layers.append(self.upsample)
            layers.append(self.bn)
            layers.append(self.act)

        self.shared_upsample = nn.Sequential(*layers)

    def forward(self, x):
        x = [x[:, i, :] for i in range(self.heads)]

        x = [
            self.head(xi).view(-1, nbands, self.rows_proj, self.cols_proj)
            for xi in x
        ]
        x = [
            self.dual_upsample(xi)
            for xi in x
        ]
        x = torch.sum(torch.stack(x, dim=2), dim=2)
        x = self.shared_upsample(x)
        x = self.final(x)
        # x = torch.sigmoid(x)

        return x


class AAE(nn.Module):
    def __init__(self, nbands: int, rows_input: int, cols_input: int,
                 dim_output: int, nblocks: int = 3):
        super().__init__()
        self.encoder = ConvEncoder(
            nbands=nbands,
            rows_input=rows_input,
            cols_input=cols_input,
            dim_output=dim_output,
            nblocks=nblocks,
        )
        self.decoder = ConvDecoder(
            dim_input=dim_output,
            nbands=nbands,
            rows_output=rows_input,
            cols_output=cols_input,
            nblocks=nblocks
        )
        self.disc = Discriminator(dim_input=dim_output)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)

        return recon, z

    def initialize_weights(self):
        """Initialize weights for the model."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

