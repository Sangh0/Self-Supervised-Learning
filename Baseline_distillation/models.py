import torch
import torch.nn as nn

class TeacherNet(nn.Module):

    def __init__(
        self,
        num_classes=10,
        in_dim=1,
        num_filters=16,
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_dim, num_filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(num_filters, num_filters*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(num_filters*2 * 7 * 7, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, num_classes),
        )

    def forward(self, inputs):
        x = self.features(inputs)
        x = x.view(inputs.size(0), -1)
        x = self.classifier(x)
        return x


class StudentNet(nn.Module):

    def __init__(
        self,
        num_classes=10,
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(28*28, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.model(x)
        return x