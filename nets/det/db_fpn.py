import torch
import torch.nn as nn


class DBFPN(nn.Module):
    """Feature Pyramid Network for DBNet.

    Takes multi-scale features from backbone and fuses them
    into a single feature map for the DB head.
    """

    def __init__(self, inner_channel, layer_out_channels):
        super(DBFPN, self).__init__()
        self.inner_channel = inner_channel

        # 1x1 convs to align all feature levels to inner_channel
        self.in5 = nn.Conv2d(
            in_channels=layer_out_channels[-1],
            out_channels=inner_channel,
            kernel_size=(1, 1), stride=(1, 1), bias=False,
        )
        self.in4 = nn.Conv2d(
            in_channels=layer_out_channels[-2],
            out_channels=inner_channel,
            kernel_size=(1, 1), stride=(1, 1), bias=False,
        )
        self.in3 = nn.Conv2d(
            in_channels=layer_out_channels[-3],
            out_channels=inner_channel,
            kernel_size=(1, 1), stride=(1, 1), bias=False,
        )
        self.in2 = nn.Conv2d(
            in_channels=layer_out_channels[-4],
            out_channels=inner_channel,
            kernel_size=(1, 1), stride=(1, 1), bias=False,
        )

        self.up5 = nn.Upsample(scale_factor=2, mode="nearest")
        self.up4 = nn.Upsample(scale_factor=2, mode="nearest")
        self.up3 = nn.Upsample(scale_factor=2, mode="nearest")

        self.out5 = nn.Sequential(
            nn.Conv2d(inner_channel, inner_channel // 4, (3, 3), (1, 1), (1, 1), bias=False),
            nn.Upsample(scale_factor=8, mode="nearest"),
        )
        self.out4 = nn.Sequential(
            nn.Conv2d(inner_channel, inner_channel // 4, (3, 3), (1, 1), (1, 1), bias=False),
            nn.Upsample(scale_factor=4, mode="nearest"),
        )
        self.out3 = nn.Sequential(
            nn.Conv2d(inner_channel, inner_channel // 4, (3, 3), (1, 1), (1, 1), bias=False),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )
        self.out2 = nn.Conv2d(
            inner_channel, inner_channel // 4, (3, 3), (1, 1), (1, 1), bias=False,
        )

    def forward(self, features):
        """Forward pass.

        Args:
            features: list/tuple of [c2, c3, c4, c5] at 1/4, 1/8, 1/16, 1/32 scales

        Returns:
            fused feature map of shape (N, inner_channel, H, W)
        """
        c2, c3, c4, c5 = features

        # Align channels
        in5 = self.in5(c5)
        in4 = self.in4(c4)
        in3 = self.in3(c3)
        in2 = self.in2(c2)

        # Top-down feature fusion
        out5 = in5
        out4 = self.up5(in5) + in4
        out3 = self.up4(in4) + in3
        out2 = self.up3(in3) + in2

        # Reduce channels and upsample to common scale
        p5 = self.out5(out5)
        p4 = self.out4(out4)
        p3 = self.out3(out3)
        p2 = self.out2(out2)

        fuse = torch.cat((p5, p4, p3, p2), dim=1)
        return fuse
