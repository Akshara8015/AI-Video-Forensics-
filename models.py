import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision.models import resnet50, ResNet50_Weights

#####################
#    CNN LAYER      #
#####################

class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()

        resnet152 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        modules = list(resnet152.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        for p in self.resnet.parameters():
            p.requires_grad = False

    def forward(self, x):
        """
            x shape : [batch_size, seq_len, 3, 224, 224]
            Example : [8, 16, 3, 224, 224]
        """

        batch_size, seq_len, C, H, W = x.size()

        x = x.reshape(batch_size * seq_len, C, H, W)
        features = self.resnet(x)

        # Output : [batch_size*seq_len, 2048, 1, 1]
        features = features.view(features.size(0), -1)

        # Output : [batch_size*seq_len, 2048]
        features = features.view(
            batch_size,
            seq_len,
            2048
        )
        return features
    

#####################
#   LSTM LAYER      #
#####################

class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = CNNFeatureExtractor()
        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        self.fc = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        features = self.cnn(x)
        lstm_out, (hidden, cell) = self.lstm(features)

        # hidden shape: [num_layers*directions, batch, hidden_size]
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]

        hidden_cat = torch.cat(
            [forward_hidden, backward_hidden],
            dim=1
        )

        output = self.fc(hidden_cat)
        return output
    

#########################
#    DATASET CLASS      #
#########################


class videoDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, label = self.samples[idx]
        frames = []

        for path in frame_paths:
            image = Image.open(path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            frames.append(image)

        frames = torch.stack(frames)
        return frames, torch.tensor(
            label,
            dtype=torch.float32
        )
    
