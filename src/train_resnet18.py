from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

from torchvision.models import (
    resnet18,
    ResNet18_Weights
)

import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "classification"
)

TRAIN_DIR = DATASET_DIR / "train"

VAL_DIR = DATASET_DIR / "val"

MODEL_DIR = PROJECT_ROOT / "models"

RESULTS_DIR = PROJECT_ROOT / "results"


# Create directories if they don't exist

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TRAINING PARAMETERS
# ============================================================

IMAGE_SIZE = 224

BATCH_SIZE = 8

NUM_EPOCHS = 5

LEARNING_RATE = 0.0001

NUM_CLASSES = 2


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# PRINT CONFIGURATION
# ============================================================

print("=" * 70)
print("             RESNET18 TRAINING")
print("=" * 70)

print()

print("Device        :", DEVICE)

print("Image Size    :", IMAGE_SIZE)

print("Batch Size    :", BATCH_SIZE)

print("Epochs        :", NUM_EPOCHS)

print("Learning Rate :", LEARNING_RATE)

print()


# ============================================================
# DATA AUGMENTATION
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    # Small realistic rotation
    transforms.RandomRotation(
        degrees=5
    ),

    # Lighting variation
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

    transforms.ToTensor(),

    # ImageNet normalization
    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )

])


# ============================================================
# VALIDATION TRANSFORM
# ============================================================

val_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )

])


# ============================================================
# LOAD DATASETS
# ============================================================

print("Loading datasets...")

train_dataset = datasets.ImageFolder(
    root=TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    root=VAL_DIR,
    transform=val_transform
)


print("Training images   :", len(train_dataset))

print("Validation images :", len(val_dataset))

print()


# ============================================================
# CLASS INFORMATION
# ============================================================

print("Classes:")

print(train_dataset.classes)

print()

print("Class mapping:")

print(train_dataset.class_to_idx)

print()


# ============================================================
# COUNT CLASS SAMPLES
# ============================================================

class_counts = [
    0 for _ in range(NUM_CLASSES)
]


for _, label in train_dataset.samples:

    class_counts[label] += 1


print("Training class counts:")

for index, count in enumerate(
    class_counts
):

    class_name = train_dataset.classes[index]

    print(
        f"{class_name:12} : {count}"
    )

print()


# ============================================================
# CALCULATE CLASS WEIGHTS
# ============================================================

total_samples = sum(
    class_counts
)

class_weights = []

for count in class_counts:

    weight = (
        total_samples
        / (
            NUM_CLASSES * count
        )
    )

    class_weights.append(weight)


print("Class weights:")

for index, weight in enumerate(
    class_weights
):

    class_name = train_dataset.classes[index]

    print(
        f"{class_name:12} : "
        f"{weight:.4f}"
    )

print()


# Convert weights to tensor

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(DEVICE)


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


# ============================================================
# LOAD PRETRAINED RESNET18
# ============================================================

print("=" * 70)

print("Loading pretrained ResNet18...")

print("=" * 70)

print()


weights = ResNet18_Weights.DEFAULT

model = resnet18(
    weights=weights
)


# ============================================================
# REPLACE FINAL CLASSIFIER
# ============================================================

input_features = model.fc.in_features

model.fc = nn.Linear(
    input_features,
    NUM_CLASSES
)


# Move model to CPU/GPU

model = model.to(DEVICE)


print(
    "Final classifier:",
    model.fc
)

print()


# ============================================================
# LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)


# ============================================================
# TRAINING HISTORY
# ============================================================

train_losses = []

val_losses = []

train_accuracies = []

val_accuracies = []


# ============================================================
# BEST MODEL TRACKING
# ============================================================

best_val_loss = float("inf")

best_model_path = (
    MODEL_DIR
    / "resnet18_classifier.pth"
)


# ============================================================
# TRAINING LOOP
# ============================================================

print("=" * 70)

print("STARTING TRAINING")

print("=" * 70)

print()


for epoch in range(NUM_EPOCHS):

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0


    for images, labels in train_loader:

        images = images.to(DEVICE)

        labels = labels.to(DEVICE)


        # Clear gradients

        optimizer.zero_grad()


        # Forward pass

        outputs = model(
            images
        )


        # Calculate loss

        loss = criterion(
            outputs,
            labels
        )


        # Backpropagation

        loss.backward()


        # Update weights

        optimizer.step()


        # Statistics

        running_loss += (
            loss.item()
            * images.size(0)
        )


        _, predicted = torch.max(
            outputs,
            1
        )


        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


    train_loss = (
        running_loss
        / len(train_dataset)
    )


    train_accuracy = (
        correct
        / total
    ) * 100


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_running_loss = 0.0

    val_correct = 0

    val_total = 0


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)


            outputs = model(
                images
            )


            loss = criterion(
                outputs,
                labels
            )


            val_running_loss += (
                loss.item()
                * images.size(0)
            )


            _, predicted = torch.max(
                outputs,
                1
            )


            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()


    val_loss = (
        val_running_loss
        / len(val_dataset)
    )


    val_accuracy = (
        val_correct
        / val_total
    ) * 100


    # --------------------------------------------------------
    # UPDATE LEARNING RATE
    # --------------------------------------------------------

    scheduler.step(
        val_loss
    )


    # --------------------------------------------------------
    # SAVE HISTORY
    # --------------------------------------------------------

    train_losses.append(
        train_loss
    )

    val_losses.append(
        val_loss
    )

    train_accuracies.append(
        train_accuracy
    )

    val_accuracies.append(
        val_accuracy
    )


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print(
        f"Epoch [{epoch + 1:02d}/{NUM_EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_accuracy:.2f}% "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_accuracy:.2f}%"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            best_model_path
        )

        print(
            "   ✓ Best model saved."
        )

    print()


# ============================================================
# TRAINING COMPLETE
# ============================================================

print("=" * 70)

print("TRAINING COMPLETED")

print("=" * 70)

print()

print(
    "Best model saved to:"
)

print(best_model_path)

print()


# ============================================================
# PLOT LOSS
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    train_losses,
    label="Training Loss"
)

plt.plot(
    val_losses,
    label="Validation Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "ResNet18 Training and Validation Loss"
)

plt.legend()

plt.grid()

loss_path = (
    RESULTS_DIR
    / "training_loss.png"
)

plt.savefig(
    loss_path
)

plt.close()


# ============================================================
# PLOT ACCURACY
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    train_accuracies,
    label="Training Accuracy"
)

plt.plot(
    val_accuracies,
    label="Validation Accuracy"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy (%)"
)

plt.title(
    "ResNet18 Training and Validation Accuracy"
)

plt.legend()

plt.grid()

accuracy_path = (
    RESULTS_DIR
    / "training_accuracy.png"
)

plt.savefig(
    accuracy_path
)

plt.close()


# ============================================================
# FINAL INFORMATION
# ============================================================

print(
    "Training loss graph:"
)

print(loss_path)

print()

print(
    "Training accuracy graph:"
)

print(accuracy_path)

print()

print("=" * 70)