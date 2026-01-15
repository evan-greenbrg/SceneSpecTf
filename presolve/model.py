import torch
from torch import nn
from torch.nn.utils import spectral_norm


class ConvBlock(nn.Module):
    def __init__(self, nbands: int, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        # stride = 2 halvs h, w
        self.conv = nn.Conv2d(
            in_channels=nbands,
            out_channels=nbands,
            kernel_size=kernel_size,
            stride=stride,
            padding=1
        )
        self.bn = nn.BatchNorm2d(nbands)
        self.act = nn.GELU()
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)

        return x


class ConvEncoder(nn.Module):
    def __init__(self, nbands: int, 
                 rows_input: int, cols_input: int, 
                 dim_output: int, heads: int,
                 nblocks: int):
        """
        rows_output and rows_input are divisible by 8
        """
        super().__init__()

        _reduc = 2**(2 * nblocks)
        self.rows_proj = round(rows_input / _reduc)
        self.cols_proj = round(cols_input / _reduc)
        self.cnn_blocks = nn.Sequential(*[
            ConvBlock(nbands, kernel_size=2, stride=2)
            for i in range(nblocks)
        ])
        # Advantage of Linear layer here vs a pooling?
        self.heads = heads
        self.dim_output = dim_output
        self.head = nn.Linear(
            nbands * self.rows_proj * self.cols_proj,
            heads * dim_output
        )
        self.max = nn.ReLU(inplace=True)
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.cnn_blocks(x)
        x = x.view(x.size(0), -1)
        x = self.softplus(self.head(x))
        x = x.view(-1, self.heads, self.dim_output)

        return x


class Discriminator(nn.Module):
    def __init__(self, dim_input: int, heads: int, hidden_dim: list = [128], use_dropout: bool = False):
        super().__init__()
        self.heads = heads
        layers = []
        for h in hidden_dim:
            layers.append(spectral_norm(nn.Linear(dim_input, h)))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if use_dropout:
                layers.append(nn.Dropout(0.3))
            dim_input = h

        layers.append(spectral_norm(nn.Linear(hidden_dim[-1], 1)))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = [x[:, i, :] for i in range(self.heads)]
        x = [
            self.layers(xi).squeeze(dim=1)
            for xi in x
        ]
        x = torch.stack(x, dim=1)

        return x
