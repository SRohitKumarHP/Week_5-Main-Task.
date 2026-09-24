from pathlib import Path

import torch
import matplotlib.pyplot as plt

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset" / "classification"

TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"

IMAGE_SIZE = 224

BATCH_SIZE = 8


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# TRANSFORMS
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

])


# ============================================================
# LOAD DATASETS
# ============================================================

train_dataset = datasets.ImageFolder(
    root=TRAIN_DIR,
    transform=transform
)

val_dataset = datasets.ImageFolder(
    root=VAL_DIR,
    transform=transform
)

test_dataset = datasets.ImageFolder(
    root=TEST_DIR,
    transform=transform
)


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# DISPLAY DATASET INFORMATION
# ============================================================

print("=" * 70)
print("             DATASET VISUALIZATION")
print("=" * 70)

print()

print("Device:", DEVICE)

print()

print("Classes:")
print(train_dataset.classes)

print()

print("Class Mapping:")
print(train_dataset.class_to_idx)

print()

print("Training Images   :", len(train_dataset))
print("Validation Images :", len(val_dataset))
print("Testing Images    :", len(test_dataset))

print()

# ============================================================
# GET ONE BATCH
# ============================================================

images, labels = next(iter(train_loader))


print("=" * 70)
print("BATCH INFORMATION")
print("=" * 70)

print()

print("Image Tensor Shape :", images.shape)

print("Label Tensor Shape :", labels.shape)

print()

print(
    "Expected Image Shape:"
)

print(
    "(Batch Size, Channels, Height, Width)"
)

print()

print(
    "Actual Shape:",
    tuple(images.shape)
)

print()


# ============================================================
# DISPLAY IMAGES
# ============================================================

fig, axes = plt.subplots(
    2,
    4,
    figsize=(12, 7)
)

fig.suptitle(
    "Metal Defect Dataset Samples",
    fontsize=16
)


for index, ax in enumerate(axes.flat):

    if index >= len(images):
        ax.axis("off")
        continue

    image = images[index]

    label = labels[index].item()

    # Convert from CHW → HWC
    image = image.permute(
        1,
        2,
        0
    )

    ax.imshow(image)

    ax.set_title(
        train_dataset.classes[label]
    )

    ax.axis("off")


plt.tight_layout()

plt.show()


# ============================================================
# FINAL MESSAGE
# ============================================================

print("=" * 70)
print("Dataset loading completed successfully.")
print("=" * 70)