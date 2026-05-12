from torch import nn
from torch.nn import BatchNorm1d, BatchNorm2d, Conv2d, Dropout, Linear, Module, PReLU, Sequential
from torch.nn import functional as F


class Flatten(Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)


class ConvBlock(Module):
    def __init__(self, in_channels, out_channels, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super().__init__()
        self.conv = Conv2d(
            in_channels,
            out_channels=out_channels,
            kernel_size=kernel,
            groups=groups,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = BatchNorm2d(out_channels)
        self.prelu = PReLU(out_channels)

    def forward(self, x):
        return self.prelu(self.bn(self.conv(x)))


class LinearBlock(Module):
    def __init__(self, in_channels, out_channels, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super().__init__()
        self.conv = Conv2d(
            in_channels,
            out_channels=out_channels,
            kernel_size=kernel,
            groups=groups,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class DepthWise(Module):
    def __init__(self, in_channels, out_channels, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels=groups, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_dw = ConvBlock(groups, out_channels=groups, groups=groups, kernel=kernel, stride=stride, padding=padding)
        self.project = LinearBlock(groups, out_channels, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.residual = residual

    def forward(self, x):
        shortcut = x
        x = self.project(self.conv_dw(self.conv(x)))
        if self.residual:
            x = shortcut + x
        return x


class Residual(Module):
    def __init__(self, channels, num_blocks, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        self.model = Sequential(
            *[
                DepthWise(
                    channels,
                    channels,
                    residual=True,
                    kernel=kernel,
                    stride=stride,
                    padding=padding,
                    groups=groups,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x):
        return self.model(x)


class MobileFaceNet(Module):
    """MobileFaceNet backbone adapted for 112x112 face and sketch inputs."""

    def __init__(self, embedding_size=512, dropout=0.0, out_h=7, out_w=7):
        super().__init__()
        self.conv1 = ConvBlock(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = ConvBlock(64, 64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)
        self.conv_23 = DepthWise(64, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        self.conv_3 = Residual(64, num_blocks=4, groups=128)
        self.conv_34 = DepthWise(64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        self.conv_4 = Residual(128, num_blocks=6, groups=256)
        self.conv_45 = DepthWise(128, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512)
        self.conv_5 = Residual(128, num_blocks=2, groups=256)
        self.conv_6_sep = ConvBlock(128, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = LinearBlock(512, 512, groups=512, kernel=(out_h, out_w), stride=(1, 1), padding=(0, 0))
        self.flatten = Flatten()
        self.dropout = Dropout(dropout)
        self.linear = Linear(512, embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_dw(x)
        x = self.conv_23(x)
        x = self.conv_3(x)
        x = self.conv_34(x)
        x = self.conv_4(x)
        x = self.conv_45(x)
        x = self.conv_5(x)
        x = self.conv_6_sep(x)
        x = self.conv_6_dw(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.linear(x)
        x = self.bn(x)
        return F.normalize(x, p=2, dim=1)
