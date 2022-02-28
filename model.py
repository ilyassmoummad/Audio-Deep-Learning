import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary


class ResBlock(nn.Module):
    def __init__(self, input_channels, num_channels, kernel_size=3, stride=1):

        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels,
            num_channels,
            kernel_size=kernel_size,
            padding=1,
            stride=stride,
        )
        self.conv2 = nn.Conv2d(
            num_channels,
            num_channels,
            kernel_size=kernel_size,
            padding=1,
            stride=stride,
        )
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

        self.map_input = nn.Sequential()

        if input_channels != num_channels:
            self.map_input = nn.Sequential(
                nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(num_channels),
            )

    def forward(self, X):

        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        Y += self.map_input(X)
        return F.relu(Y)


class ResNet8(nn.Module):
    def __init__(self, number_features=16, audio_channels=1, num_classes=10):

        super().__init__()
        self.input_channels = audio_channels
        self.number_features = number_features

        # first conv before res blocks
        self.conv1 = nn.Conv2d(
            self.input_channels, self.number_features, kernel_size=3, padding=1
        )

        # ResNet Layers
        self.layer1 = ResBlock(self.number_features, self.number_features)
        self.layer2 = ResBlock(self.number_features, self.number_features * 2)
        self.layer3 = ResBlock(self.number_features * 2, self.number_features * 4)
        self.pool = nn.AvgPool2d(kernel_size=2)

        # FCN
        self.linear1 = nn.Linear(4 * self.number_features, 4 * self.number_features)
        self.linear2 = nn.Linear(4 * self.number_features, num_classes)

    def forward(self, x):

        x = self.conv1(x)

        # 3 Residual Blocks
        x = self.layer1(x)
        x = self.pool(x)
        x = self.layer2(x)
        x = self.pool(x)
        x = self.layer3(x)

        # Global Pooling (as in PANNs)
        x = torch.mean(x, dim=3)
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2

        # FC with 1 hidden layer
        x = self.linear1(x)
        x = F.relu_(x)
        x = self.linear2(x)

        return x


class ResNet12(nn.Module):
    def __init__(self, number_features=16, audio_channels=1, num_classes=10):

        super().__init__()
        self.input_channels = audio_channels
        self.number_features = number_features

        # first conv before res blocks
        self.conv1 = nn.Conv2d(
            self.input_channels, self.number_features, kernel_size=3, padding=1
        )

        # ResNet Layers
        self.layer1 = ResBlock(self.number_features, self.number_features)
        self.layer2 = ResBlock(self.number_features, self.number_features)
        self.layer3 = ResBlock(self.number_features, self.number_features)
        self.layer4 = ResBlock(self.number_features, self.number_features)
        self.layer5 = ResBlock(self.number_features, self.number_features)
        self.pool = nn.AvgPool2d(kernel_size=2)

        # FCN
        self.linear1 = nn.Linear(self.number_features, self.number_features)
        self.linear2 = nn.Linear(self.number_features, num_classes)

    def forward(self, x):

        x = self.conv1(x)

        # 3 Residual Blocks
        x = self.layer1(x)
        x = self.pool(x)
        x = self.layer2(x)
        x = self.pool(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = self.layer4(x)
        x = self.pool(x)

        x = self.layer5(x)

        # Global Pooling (as in PANNs)
        x = torch.mean(x, dim=3)
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2

        # FC with 1 hidden layer
        x = self.linear1(x)
        x = F.relu_(x)
        x = self.linear2(x)

        return x


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # cnn = CNN3(nb_features=8)
    # summary(cnn.to(device), (1, 64, 313))
    resnet = ResNet12(number_features=32)
    # resnet = ResNet7(number_features=16)
    summary(resnet.to(device), (1, 64, 126))