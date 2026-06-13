import os
import torch
from glob import glob
from PIL import Image
from collections import Counter
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


root_dir = r"E:\anotated_images"
classes = {
    "deepfakedetection": 1, # FAKE 
    "original": 0  # REAL
}

video_info = []
for class_name, label in classes.items():
    class_path = os.path.join(root_dir, class_name)

    for video_folder in os.listdir(class_path):
        video_path = os.path.join(class_path, video_folder)
        if os.path.isdir(video_path):
            video_info.append((video_path, label))

print("video_info", len(video_info))

train_videos, test_videos = train_test_split(
    video_info,
    test_size=0.3,
    random_state=42,
    stratify=[label for _, label in video_info]
)

SEQUENCE_LENGTH = 16
STRIDE = 5

def create_sequences(video_list):
    samples = []

    for video_path, label in video_list:
        frame_paths = sorted(glob(os.path.join(video_path, "*.jpg")))
        if len(frame_paths) < SEQUENCE_LENGTH:
            continue
        for i in range(0, len(frame_paths)-SEQUENCE_LENGTH+1, STRIDE ):
            sequence = frame_paths[i:i+SEQUENCE_LENGTH]
            samples.append((sequence, label))
    return samples

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
    

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_samples = create_sequences(train_videos)
test_samples = create_sequences(test_videos)

print("Train Sequences:", len(train_samples))
print("Test Sequences:", len(test_samples))

train_dataset = videoDataset(
    train_samples,
    transform=train_transform
)

test_dataset = videoDataset(
    test_samples,
    transform=test_transform
)

print("train_dataset: ", len(train_dataset))
print("test_dataset: ", len(test_dataset))

BATCH_SIZE = 8
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


frames, labels = next(iter(train_loader))

print("frames.shape", frames.shape)
print("labels.shape", labels.shape)

labels = [label for _, label in train_samples]
print("Counter(labels)", Counter(labels))

