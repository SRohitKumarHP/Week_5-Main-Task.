from pathlib import Path

import matplotlib.pyplot as plt
import torch

from torchvision import datasets
from torchvision import transforms


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "classification"
    / "train"
)

IMAGE_SIZE = 224


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# TRAINING TRANSFORM
# ============================================================

train_transform = transforms.Compose([

    # Resize image
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    # Small rotation
    transforms.RandomRotation(
        degrees=5
    ),

    # Brightness and contrast variation
    transforms.ColorJitter(
        brightness=0.20,
        contrast=0.20
    ),

    # Slight blur
    transforms.RandomApply(
        [
            transforms.GaussianBlur(
                kernel_size=3,
                sigma=(0.1, 1.0)
            )
        ],
        p=0.20
    ),

    # Convert image to tensor
    transforms.ToTensor(),

])


# ============================================================
# LOAD DATASET
# ============================================================

dataset = datasets.ImageFolder(
    root=TRAIN_DIR,
    transform=train_transform
)


# ============================================================
# PRINT DATASET INFORMATION
# ============================================================

print("=" * 70)
print("             DATA AUGMENTATION TEST")
print("=" * 70)

print()

print("Dataset directory:")
print(TRAIN_DIR)

print()

print("Classes:")
print(dataset.classes)

print()

print("Class mapping:")
print(dataset.class_to_idx)

print()

print("Total training images:")
print(len(dataset))

print()

# ============================================================
# SELECT ONE IMAGE
# ============================================================

original_dataset = datasets.ImageFolder(
    root=TRAIN_DIR
)

image_path, label = original_dataset.samples[0]

print("Original image:")
print(image_path)

print()

print("Original label:")
print(original_dataset.classes[label])

print()


# ============================================================
# CREATE MULTIPLE AUGMENTED VERSIONS
# ============================================================

original_image = original_dataset.loader(
    image_path
)

augmented_images = []

number_of_images = 6


for _ in range(number_of_images):

    augmented_image = train_transform(
        original_image
    )

    augmented_images.append(
        augmented_image
    )


# ============================================================
# DISPLAY AUGMENTED IMAGES
# ============================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(12, 7)
)

fig.suptitle(
    "Metal Image Data Augmentation",
    fontsize=16
)


for index, ax in enumerate(
    axes.flat
):

    image = augmented_images[index]

    # Tensor CHW → HWC
    image = image.permute(
        1,
        2,
        0
    )

    ax.imshow(image)

    ax.set_title(
        f"Augmented Image {index + 1}"
    )

    ax.axis("off")


plt.tight_layout()

plt.show()


# ============================================================
# FINAL MESSAGE
# ============================================================

print("=" * 70)
print("Data augmentation test completed.")
print("=" * 70)