import torch.nn as nn
import torch.nn.functional as F

class TinyEEGNet(nn.Module):
    def __init__(self, n_channels=23, n_classes=2, dropout_rate=0.5):
        super(TinyEEGNet, self).__init__()
        # Block 1: Temporal convolution
        self.conv1 = nn.Conv2d(1, 16, (1, 65), padding='same')
        self.bn1 = nn.BatchNorm2d(16)
        self.dropout1 = nn.Dropout2d(dropout_rate)

        # Block 2: Spatial depthwise convolution
        self.depthwise = nn.Conv2d(16, 32, (n_channels, 1), groups=16)
        self.bn2 = nn.BatchNorm2d(32)
        self.dropout2 = nn.Dropout2d(dropout_rate)

        # Global average pooling để giảm số tham số (thay vì flatten)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # output: (batch, 32, 1, 1)

        # Classifier
        self.fc = nn.Linear(32, n_classes)

    def forward(self, x):
        # x: (batch, channels, time) = (batch, 23, 1000)
        x = x.unsqueeze(1)          # (batch, 1, 23, 1000)
        x = F.elu(self.bn1(self.conv1(x)))      # (batch, 16, 23, 1000)
        x = self.dropout1(x)

        x = F.elu(self.bn2(self.depthwise(x)))  # (batch, 32, 1, 1000)
        x = self.dropout2(x)

        x = self.global_avg_pool(x)             # (batch, 32, 1, 1)
        x = x.view(x.size(0), -1)               # (batch, 32)

        return self.fc(x)