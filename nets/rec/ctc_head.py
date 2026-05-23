import torch
import torch.nn as nn
import torch.nn.functional as F


class CTCHead(nn.Module):
    """CTC head for recognition.

    Projects the encoder hidden states to class probabilities
    via a linear layer followed by log_softmax.
    """

    def __init__(self, in_features, out_features, bias=True):
        super(CTCHead, self).__init__()

        self.fc = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        )
        self.fc.weight.data.normal_(0, 0.01)
        if self.fc.bias is not None:
            self.fc.bias.data.zero_()

    def forward(self, x):
        """Forward pass.

        Args:
            x: tensor of shape (T, N, in_features)

        Returns:
            tensor of shape (T, N, out_features) after log_softmax
        """
        x = self.fc(x)
        x = F.log_softmax(x, dim=2)
        return x
