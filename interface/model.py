import torch
import torch.nn as nn


class ResUNet(nn.Module):
    def __init__(self, in_channels=270, num_classes=24):
        super(ResUNet, self).__init__()
        self.num_classes = num_classes

        # Residual block
        def res_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        # Encoder with residual blocks
        self.enc1 = res_block(in_channels, 64)
        self.enc2 = res_block(64, 128)
        self.enc3 = res_block(128, 256)
        self.pool = nn.MaxPool2d(2)

        # 1x1 convolution for residual matching in encoders
        self.res1 = nn.Conv2d(in_channels, 64, kernel_size=1, bias=False)
        self.res2 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.res3 = nn.Conv2d(128, 256, kernel_size=1, bias=False)

        # Decoder
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = res_block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = res_block(128, 64)
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder with residual connections
        enc1 = self.enc1(x) + self.res1(x)  # Residual connection on input
        enc2 = self.enc2(self.pool(enc1)) + self.res2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2)) + self.res3(self.pool(enc2))

        # Decoder
        dec3 = self.dec3(torch.cat([self.up3(enc3), enc2], dim=1))
        dec2 = self.dec2(torch.cat([self.up2(dec3), enc1], dim=1))

        return self.final(dec2)
