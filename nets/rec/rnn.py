import torch
import torch.nn as nn
from .sequence_encoder import SequenceEncoder
from .ctc_head import CTCHead


class CRNN(nn.Module):
    def __init__(
            self,
            classes_num,
            rnn_type,
            hidden_size,
            num_layers,
            bidirectional,
            backbone,
    ):
        super(CRNN, self).__init__()

        self.backbone = self._call_backbone(backbone)
        rnn_in_channel = self.backbone.output_channel

        self.encoder = SequenceEncoder(
            input_size=rnn_in_channel,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            rnn_type=rnn_type,
        )

        rnn_out_channel = hidden_size * 2 if bidirectional else hidden_size

        self.head = CTCHead(
            in_features=rnn_out_channel,
            out_features=classes_num,
        )

    @staticmethod
    def _call_backbone(backbone):
        module_func = backbone.pop("name")
        if module_func == "rec_mobilenet_v3":
            from nets.rec.mobilenet_v3 import rec_mobilenet_v3
            module_func = eval(module_func)(**backbone)
        elif module_func in ["rec_resnet18","rec_resnet34", "rec_resnet50"]:
            from nets.rec.resnet import rec_resnet18, rec_resnet34, rec_resnet50
            module_func = eval(module_func)(**backbone)
        else:
            raise Exception("backbone {} is not found".format(module_func))
        return module_func

    def forward(self, x):
        """
        :param x: N * 3 * H * W
        :return: # N * T * Feature
        """
        x = self.backbone(x)  # N * C * 1 * W (mobilenet: N * 288 * 1 * 25)
        x = x.squeeze(axis=2)  # N * C * W
        x = x.permute(2, 0, 1)  # W * N * C
        x = self.encoder(x)
        x = self.head(x)
        return x


if __name__ == "__main__":
    input_ = torch.randn(8, 3, 32, 320)
    se = CRNN(5000, "GRU", 48, 2, True, {"name": "rec_mobilenet_v3"})
    # se.eval()
    print(se(input_).shape)
