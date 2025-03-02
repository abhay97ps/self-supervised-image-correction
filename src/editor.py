import torch
import torch.nn as nn
from functools import partial


def activation_func(activation):
    return nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]


class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)


conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)


def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(conv(in_channels, out_channels, *args, **kwargs), nn.BatchNorm2d(out_channels))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super().__init__()
        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
        self.blocks = nn.Identity()
        self.activate = activation_func(activation)
        self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        if self.should_apply_shortcut:
            residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = self.activate(x)
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels


class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, sampling=1, conv=conv3x3, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.expansion, self.sampling, self.conv = expansion, sampling, conv
        self.shortcut = nn.Sequential(
            nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                      stride=self.sampling, bias=False),
            nn.BatchNorm2d(self.expanded_channels)) if self.should_apply_shortcut else None

    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels


class ResNetBasicBlock(ResNetResidualBlock):
    expansion = 1

    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels,
                    conv=self.conv, bias=False, stride=self.sampling),
            activation_func(self.activation),
            conv_bn(self.out_channels, self.expanded_channels,
                    conv=self.conv, bias=False),
        )


class ResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1, *args, **kwargs):
        super().__init__()
        # 'We perform sampling directly by convolutional layers that have a stride of 2.'
        sampling = 2 if in_channels != out_channels else 1
        self.blocks = nn.Sequential(
            block(in_channels, out_channels,
                  sampling=sampling, *args, **kwargs),
            *[block(out_channels * block.expansion,
                    out_channels, sampling=1, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x


class ResNetEncoder(nn.Module):
    def __init__(self, in_channels=3, blocks_sizes=[64, 128, 256, 512], depths=[2, 2, 2, 2],
                 activation='relu', block=ResNetBasicBlock, *args, **kwargs):
        super().__init__()
        self.blocks_sizes = blocks_sizes

        self.gate = nn.Sequential(
            nn.Conv2d(
                in_channels, self.blocks_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.blocks_sizes[0]),
            activation_func(activation)
            #nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList([
            ResNetLayer(blocks_sizes[0], blocks_sizes[0], n=depths[0], activation=activation,
                        block=block, *args, **kwargs),
            *[ResNetLayer(in_channels * block.expansion,
                          out_channels, n=n, activation=activation,
                          block=block, *args, **kwargs)
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, depths[1:])]
        ])

    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x


class TConv2dAuto(nn.ConvTranspose2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)


tconv3x3 = partial(TConv2dAuto, kernel_size=3, bias=False)


class ResNetDecoderResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, sampling=1, out_pad=0, conv=tconv3x3, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.expansion, self.sampling, self.conv, self.out_pad = expansion, sampling, conv, out_pad
        self.shortcut = nn.Sequential(
            nn.ConvTranspose2d(self.in_channels, self.expanded_channels, kernel_size=1,
                               stride=self.sampling, bias=False, output_padding=self.out_pad),
            nn.BatchNorm2d(self.expanded_channels)) if self.should_apply_shortcut else None

    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels


class ResNetDecoderBasicBlock(ResNetDecoderResidualBlock):
    expansion = 1

    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels,
                    conv=self.conv, bias=False, stride=self.sampling, output_padding=self.out_pad),
            activation_func(self.activation),
            conv_bn(self.out_channels, self.expanded_channels,
                    conv=self.conv, bias=False),
        )


class ResNetDecoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, block=ResNetDecoderBasicBlock, n=1, *args, **kwargs):
        super().__init__()
        # 'We perform sampling directly by convolutional layers that have a stride of 2.'
        sampling = 2 if in_channels != out_channels else 1
        out_pad = 1 if in_channels != out_channels else 0
        self.blocks = nn.Sequential(
            block(in_channels, out_channels,
                  sampling=sampling, out_pad=out_pad, *args, **kwargs),
            *[block(out_channels * block.expansion,
                    out_channels, sampling=1, out_pad=0, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x


class ResNetDecoder(nn.Module):
    def __init__(self, out_channels=3, blocks_sizes=[512, 256, 128, 64], depths=[2, 2, 2, 2],
                 activation='relu', block=ResNetDecoderBasicBlock, *args, **kwargs):
        super().__init__()
        self.blocks_sizes = blocks_sizes

        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList([
            *[ResNetDecoderLayer(in_channels * block.expansion, out_channels, n=n, activation=activation, block=block,
                                 *args, **kwargs) for (in_channels, out_channels), n in zip(self.in_out_block_sizes, depths[1:])],
            ResNetDecoderLayer(blocks_sizes[-1], blocks_sizes[-1], n=depths[0],
                               activation=activation, block=block, *args, **kwargs)
        ])

        self.exit = nn.Sequential(
            nn.ConvTranspose2d(blocks_sizes[-1], out_channels, kernel_size=7,
                               stride=2, padding=3, bias=False, output_padding=1),
            nn.BatchNorm2d(out_channels),
            activation_func(activation)
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.exit(x)
        return x


class Editor(nn.Module):

    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, *args, **kwargs)
        self.decoder = ResNetDecoder(out_channels, *args, **kwargs)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def editor18(in_channels, *args, **kwargs):
    return Editor(in_channels, in_channels, depths=[2, 2, 2, 2], *args, **kwargs)


