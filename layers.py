import os
from glob import glob
from pathlib import Path
from collections import Counter
from torchvision import transforms
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from models import *

root_dir = r"E:\anotated_images"
classes = {
    "DeepFakeDetection": 1, # FAKE 
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


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

model = LSTM().to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)

# ===============================
# RESUME TRAINING IF CHECKPOINT EXISTS
# ===============================

SAVE_DIR = Path(__file__).resolve().parent
SAVE_DIR.mkdir(parents=True, exist_ok=True)

checkpoint_path = SAVE_DIR / "latest_checkpoint.pth"
legacy_checkpoint_path = SAVE_DIR / "deepfake_models" / "checkpoints" / "latest_checkpoint.pth"
best_model_path = SAVE_DIR / "best_deepfake_model.pth"

start_epoch = 0
best_acc = 0.0

resume_checkpoint_path = (
    checkpoint_path
    if checkpoint_path.exists()
    else legacy_checkpoint_path
)

if resume_checkpoint_path.exists():
    print(f"\nLoading checkpoint from {resume_checkpoint_path}\n")
    checkpoint = torch.load(
        resume_checkpoint_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    start_epoch = checkpoint["epoch"] + 1
    best_acc = checkpoint["best_acc"]

    print(
        f"Resuming from Epoch {start_epoch}"
    )

else:
    print("\nNo checkpoint found. Starting fresh training.\n")


# Verify Before Training
# model = LSTM()
# frames, labels = next(iter(train_loader))
# out = model(frames)
# print(out.shape)


# 5. Training Function
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for frames, labels in dataloader:
        frames = frames.to(device)
        labels = labels.float().to(device)
        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(
            outputs.squeeze(),
            labels
        )

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        probs = torch.sigmoid(outputs.squeeze())
        preds = (probs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# 6. Validation Function
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for frames, labels in dataloader:
            frames = frames.to(device)
            labels = labels.float().to(device)

            outputs = model(frames)
            loss = criterion(outputs.squeeze(),labels)

            running_loss += loss.item()
            probs = torch.sigmoid(outputs.squeeze())
            preds = (probs > 0.5).float()
            correct += ( preds == labels ).sum().item()
            total += labels.size(0)

    epoch_loss = ( running_loss / len(dataloader) )
    epoch_acc = (correct / total)
    return epoch_loss, epoch_acc


# 7. Full Training Loop
NUM_EPOCHS = 10

for epoch in range(start_epoch, NUM_EPOCHS):
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, test_loader, criterion, device)

    print(
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_acc:.4f}"
    )
    print(
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_acc:.4f}"
    )

    # ==========================
    # SAVE LATEST CHECKPOINT
    # ==========================

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict":model.state_dict(),
            "optimizer_state_dict":optimizer.state_dict(),
            "best_acc": best_acc
        },
        checkpoint_path
    )

    # ==========================
    # SAVE EPOCH CHECKPOINT
    # ==========================

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict":model.state_dict(),
            "optimizer_state_dict":optimizer.state_dict(),
            "best_acc": best_acc
        },
        SAVE_DIR /
        f"checkpoint_epoch_{epoch+1}.pth"
    )

    # ==========================
    # SAVE BEST MODEL
    # ==========================

    if val_acc > best_acc:
        best_acc = val_acc

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict":
                    model.state_dict(),
                "optimizer_state_dict":
                    optimizer.state_dict(),
                "best_acc":
                    best_acc
            },
            best_model_path
        )

        print(
            f"Best Model Saved "
            f"(Acc={best_acc:.4f})"
        )
    print("-" * 50)


# 8. Loading Best Model

best_checkpoint = torch.load(
    best_model_path,
    map_location=device
)

model.load_state_dict(
    best_checkpoint["model_state_dict"]
)

print(
    f"Loaded best model "
    f"from epoch "
    f"{best_checkpoint['epoch']+1}"
)

print("\nSaved Files:")

for root, dirs, files in os.walk(SAVE_DIR):
    for file in files:
        print(os.path.join(root, file))


