import torch.nn as nn

class CatDogCNN(nn.Module):
    def __init__(self):
        super().__init__()

import torch.nn as nn

class CatDogCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 3 color channels in, 32 filters out
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 224 -> 112

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 112 -> 56

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 56 -> 28
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                                 # 128 * 28 * 28 = 100352
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)                            # binary output
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x