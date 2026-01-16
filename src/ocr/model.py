import torch
import torch.nn as nn

class CNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128,256,3,padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)

class CRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = CNNBackbone()
        self.lstm = nn.LSTM(
            256, 256,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = x.mean(2)
        x = x.permute(0,2,1)
        x,_ = self.lstm(x)
        return self.fc(x)
