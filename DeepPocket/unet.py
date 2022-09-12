'''
Unet architecture for segmentation of pocket structures.
'''
import torch
from torch import nn
# import torch.nn.functional as F
# pylint: disable=E1101,R0913,R0902

class DoubleConv(nn.Module):
    '''
    Unet architecture for segmentation of pocket structures.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, padding=1):
        super().__init__()
        self.block = nn.Sequential(
                                nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding),
                                nn.BatchNorm3d(out_channels),
                                nn.ReLU(),
                                nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding),
                                nn.BatchNorm3d(out_channels),
                                nn.ReLU())

    def forward(self, inp):
        '''
        Unet Forward pass
        '''
        out = self.block(inp)
        return out


class Down(nn.Module):
    '''
    Convolutional block for the Unet architecture
    '''
    def __init__(self, in_channels, out_channels, kernel_size_pad,stride=2):
        super().__init__()
        self.block = nn.Sequential(
                            nn.MaxPool3d(kernel_size_pad, stride=stride),
                             DoubleConv(in_channels, out_channels, 3))

    def forward(self, inp):
        '''
        Forward propagation from input to output
        '''
        out = self.block(inp)
        return out


class Up(nn.Module):
    '''
        Deconvolutional block for the Unet architecture
    '''
    def __init__(self, in_channels, out_channels,
                     kernel_size_up,padding=0,stride=2, out_pad=0, upsample=None):
        super().__init__()
        if upsample:
            self.up_s = nn.Upsample(scale_factor=2, mode=upsample, align_corners=True)
        else:
            self.up_s = nn.ConvTranspose3d(in_channels, in_channels // 2,
                                    kernel_size_up, stride=stride, padding=padding,
                                           output_padding=out_pad)

        self.conv_t = DoubleConv(in_channels, out_channels, 3)

    def forward(self, layer1, layer2):
        '''
        Unet Forward pass
        '''
        out = self.up_s(layer1)
        out = self.conv_t(torch.cat((layer2, out), dim=1))
        return out


class Unet(nn.Module):
    '''
    Unet class for segmentation of pocket structures.
    '''
    def __init__(self, n_classes, upsample):
        super().__init__()
        self.n_classes = n_classes

        self.in1 = DoubleConv(14, 32, 3)
        self.down1 = Down(32, 64, 3)
        self.down2 = Down(64, 128, 3)
        self.down3 = Down(128, 256, 3)
        factor = 2 if upsample else 1
        self.down4 = Down(256, 512 // factor, 3)
        self.up1 = Up(512, 256 // factor, 3, upsample=upsample,stride=2,out_pad=0)
        self.up2 = Up(256, 128 // factor, 3, upsample=upsample)
        self.up3 = Up(128, 64 // factor, 3, upsample=upsample,out_pad=1)
        self.up4 = Up(64, 32, 3, upsample=upsample)
        self.conv = nn.Conv3d(32, self.n_classes, 1)

    def forward(self, layer):
        '''
        Unet Forward propagation from input to output
        '''
        layer1 = self.in1(layer)
        layer2 = self.down1(layer1)
        layer3 = self.down2(layer2)
        layer4 = self.down3(layer3)
        layer5 = self.down4(layer4)
        layer = self.up1(layer5, layer4)
        layer = self.up2(layer, layer3)
        layer = self.up3(layer, layer2)
        layer = self.up4(layer, layer1)
        logits = self.conv(layer)
        return logits
