import torch
import torch.nn as nn


class DBHead(nn.Module):
    """Differentiable Binarization (DB) head for DBNet.

    Produces a probability map, threshold map, and binary map
    for text detection.
    """

    def __init__(self, inner_channel, k=50):
        super(DBHead, self).__init__()
        self.k = k

        self.binarize = nn.Sequential(
            nn.Conv2d(inner_channel, inner_channel // 4, (3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(inner_channel // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channel // 4, inner_channel // 4, (2, 2), (2, 2)),
            nn.BatchNorm2d(inner_channel // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channel // 4, 1, (2, 2), (2, 2)),
            nn.Sigmoid(),
        )

        self.thresh = nn.Sequential(
            nn.Conv2d(inner_channel, inner_channel // 4, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(inner_channel // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channel // 4, inner_channel // 4, (2, 2), (2, 2)),
            nn.BatchNorm2d(inner_channel // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channel // 4, 1, (2, 2), (2, 2)),
            nn.Sigmoid(),
        )

    @staticmethod
    def step_function(x, y):
        return torch.reciprocal(1 + torch.exp(-50 * (x - y)))

    def forward(self, fuse):
        """Forward pass.

        Args:
            fuse: fused feature map from FPN (N, inner_channel, H, W)

        Returns:
            eval mode: (N, 1, H, W) probability map
            train mode: (N, 3, H, W) [prob, thresh, binary_thresh]
        """
        prob = self.binarize(fuse)
        if not self.training:
            return prob

        thresh_out = self.thresh(fuse)
        binary_thresh = self.step_function(prob, thresh_out)
        return torch.cat([prob, thresh_out, binary_thresh], dim=1)
