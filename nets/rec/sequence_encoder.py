import torch
import torch.nn as nn


class SequenceEncoder(nn.Module):
    """RNN sequence encoder, supports GRU and LSTM.

    Maps an input sequence to a sequence of hidden states.
    """

    def __init__(self, input_size, hidden_size, num_layers=2, bidirectional=True, rnn_type="GRU"):
        super(SequenceEncoder, self).__init__()

        assert rnn_type in ["LSTM", "GRU"], \
            "rnn_type must be 'LSTM' or 'GRU', got {}".format(rnn_type)

        rnn_cls = nn.LSTM if rnn_type == "LSTM" else nn.GRU
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False,
            bidirectional=bidirectional,
        )

        for name, params in self.rnn.named_parameters():
            nn.init.uniform_(params, -0.1, 0.1)

    def forward(self, x):
        """Forward pass.

        Args:
            x: tensor of shape (T, N, input_size)

        Returns:
            tensor of shape (T, N, hidden_size * (2 if bidirectional else 1))
        """
        x, _ = self.rnn(x)
        return x
