from math import prod

from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()

        conv_out_features = [8, 4, 8, 2, 4, 2]

        self.conv_stack_1 = ConvStack(1, conv_out_features[0], conv_out_features[1])  # 20
        self.conv_stack_2 = ConvStack(prod(conv_out_features[:2]), conv_out_features[2], conv_out_features[3])  # 12
        self.conv_stack_3 = ConvStack(prod(conv_out_features[:4]), conv_out_features[4], conv_out_features[5])  # 4

        self.batch_norm_2d = nn.BatchNorm2d(prod(conv_out_features))

        self.gap = nn.AvgPool2d(4)

        self.linear = nn.Sequential(
            nn.Linear(prod(conv_out_features) * 4 * 4, 1024),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.conv_stack_1(x)
        x = self.conv_stack_2(x)
        x = self.conv_stack_3(x)
        x = self.batch_norm_2d(x)
        x = self.flatten(x)
        x = self.linear(x)

        return x


class ConvStack(nn.Sequential):
    def __init__(self, mult, c_out_1, c_out_2) -> None:
        super().__init__(  # total: -8
            nn.BatchNorm2d(mult),
            nn.MaxPool2d(3, 1),  # -2
            nn.BatchNorm2d(mult),
            nn.Conv2d(mult * 1, mult * c_out_1, 3, dilation=2),  # -4
            nn.ReLU(),
            nn.BatchNorm2d(mult * c_out_1),
            nn.Conv2d(mult * c_out_1, mult * c_out_1 * c_out_2, 3),  # -2
            nn.ReLU(),
        )


class Test(nn.Sequential):
    def __init__(self) -> None:
        super().__init__(nn.Linear(28 * 28, 512), nn.ReLU(), nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 10))
