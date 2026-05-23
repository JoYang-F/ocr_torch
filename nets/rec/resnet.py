import math
import torch
import torch.nn as nn

__all__ = ["rec_resnet18", "rec_resnet34", "rec_resnet50"]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=(1, 1), downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=(1, 1), downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class RecResNet(nn.Module):
    """ResNet adapted for text recognition.

    Key difference from standard ResNet:
        layer2-4 use stride=(2,1) instead of (2,2) to preserve
        the width (time) dimension for CRNN.

    Input:  N x 3 x 32 x W
    Output: N x output_channel x 1 x W/4
    """

    def __init__(self, block, layers, in_channels=3):
        super().__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=(2, 2),
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=(1, 1))
        self.layer2 = self._make_layer(block, 128, layers[1], stride=(2, 1))
        self.layer3 = self._make_layer(block, 256, layers[2], stride=(2, 1))
        self.layer4 = self._make_layer(block, 512, layers[3], stride=(2, 1))

        self.output_channel = 512 * block.expansion
        self._init_weights()

    def _make_layer(self, block, planes, blocks, stride=(1, 1)):
        downsample = None
        if stride != (1, 1) or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def rec_resnet18(pre_trained_dir=None):
    model = RecResNet(BasicBlock, [2, 2, 2, 2])
    if pre_trained_dir:
        state_dict = torch.load(pre_trained_dir, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    return model


def rec_resnet34(pre_trained_dir=None):
    model = RecResNet(BasicBlock, [3, 4, 6, 3])
    if pre_trained_dir:
        state_dict = torch.load(pre_trained_dir, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    return model


def rec_resnet50(pre_trained_dir=None):
    model = RecResNet(Bottleneck, [3, 4, 6, 3])
    if pre_trained_dir:
        state_dict = torch.load(pre_trained_dir, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    return model


if __name__ == "__main__":
    x = torch.randn(4, 3, 32, 320)
    for name, fn in [("rec_resnet18", rec_resnet18),("rec_resnet34", rec_resnet34), ("rec_resnet50", rec_resnet50)]:
        m = fn()
        out = m(x)
        print(f"{name}: {out.shape}")
