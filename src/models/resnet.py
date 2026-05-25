import torch.nn as nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_ch, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, planes, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes * 4)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += x if self.downsample is None else self.downsample(x)
        return self.relu(out)


class ResNet(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(128,  64, layers[0], stride=1)
        self.layer2 = self._make_layer(256, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(512, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(1024, 512, layers[3], stride=1, dilation=4)
        self._init_weights()

    def _make_layer(self, in_ch, planes, n, stride=1, dilation=1):
        ds = None
        if stride != 1 or in_ch != planes * 4:
            ds = nn.Sequential(
                nn.Conv2d(in_ch, planes * 4, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * 4),
            )
        layers = [Bottleneck(in_ch, planes, stride, dilation, ds)]
        for _ in range(1, n):
            layers.append(Bottleneck(planes * 4, planes, 1, dilation))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def resnet50():
    return ResNet([3, 4, 6, 3])
