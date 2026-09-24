import os
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torchvision.models import (
    mobilenet_v2,
    MobileNet_V2_Weights
)

from torch.utils.data import DataLoader


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_DIR = "dataset/classification/train"
VAL_DIR = "dataset/classification/val"

MODEL_PATH = "models/mobilenetv2_classifier.pth"

RESULTS_DIR = "results"

IMAGE_SIZE = 224

BATCH_SIZE = 8

NUM_EPOCHS = 5

LEARNING_RATE = 0.0001


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print("=" * 70)
print("MOBILENETV2 METAL DEFECT CLASSIFICATION TRAINING")
print("=" * 70)

print()
print("Device:", device)

print("Image Size:", IMAGE_SIZE)

print("Batch Size:", BATCH_SIZE)

print("Epochs:", NUM_EPOCHS)

print("Learning Rate:", LEARNING_RATE)


# ============================================================
# DATA TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomRotation(
        5
    ),

    transforms.ColorJitter(
        brightness=0.20,
        contrast=0.20
    ),

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
# LOAD DATASET
# ============================================================

print()
print("Loading datasets...")


train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)


val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_transform
)


print()
print("Training images:", len(train_dataset))

print("Validation images:", len(val_dataset))

print()
print("Classes:", train_dataset.classes)

print(
    "Class mapping:",
    train_dataset.class_to_idx
)


# ============================================================
# CALCULATE CLASS COUNTS
# ============================================================

class_counts = torch.bincount(
    torch.tensor(
        train_dataset.targets
    )
)


print()
print("Training class counts:")

for class_index, class_name in enumerate(
    train_dataset.classes
):

    print(
        f"{class_name}: "
        f"{class_counts[class_index].item()}"
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================

total_samples = len(
    train_dataset
)

num_classes = len(
    train_dataset.classes
)


class_weights = total_samples / (
    num_classes * class_counts.float()
)


class_weights = class_weights.to(
    device
)


print()
print("Class weights:")

for class_index, class_name in enumerate(
    train_dataset.classes
):

    print(
        f"{class_name}: "
        f"{class_weights[class_index].item():.4f}"
    )


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# LOAD PRETRAINED MOBILENETV2
# ============================================================

print()
print("Loading pretrained MobileNetV2...")


weights = MobileNet_V2_Weights.DEFAULT


model = mobilenet_v2(
    weights=weights
)


# ============================================================
# REPLACE CLASSIFIER
# ============================================================

classifier_input_features = (
    model.classifier[1].in_features
)


model.classifier = nn.Sequential(

    nn.Dropout(
        p=0.2
    ),

    nn.Linear(
        classifier_input_features,
        num_classes
    )
)


model = model.to(
    device
)


print(
    "MobileNetV2 loaded successfully."
)


print()
print(
    "Final classifier:"
)

print(
    model.classifier
)


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


best_val_loss = float(
    "inf"
)


# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(
    NUM_EPOCHS
):

    # ========================================================
    # TRAIN
    # ========================================================

    model.train()


    running_loss = 0.0

    correct = 0

    total = 0


    for images, labels in train_loader:

        images = images.to(
            device
        )

        labels = labels.to(
            device
        )


        optimizer.zero_grad()


        outputs = model(
            images
        )


        loss = criterion(
            outputs,
            labels
        )


        loss.backward()


        optimizer.step()


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
        / total
    )


    train_accuracy = (
        correct
        / total
        * 100
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()


    val_running_loss = 0.0

    val_correct = 0

    val_total = 0


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                device
            )

            labels = labels.to(
                device
            )


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
        / val_total
    )


    val_accuracy = (
        val_correct
        / val_total
        * 100
    )


    # ========================================================
    # UPDATE SCHEDULER
    # ========================================================

    scheduler.step(
        val_loss
    )


    # ========================================================
    # SAVE HISTORY
    # ========================================================

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


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss


        checkpoint = {

            "model_state_dict":
                model.state_dict(),

            "class_names":
                train_dataset.classes,

            "image_size":
                IMAGE_SIZE
        }


        os.makedirs(
            os.path.dirname(
                MODEL_PATH
            ),
            exist_ok=True
        )


        torch.save(
            checkpoint,
            MODEL_PATH
        )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()

    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS}"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Train Acc : {train_accuracy:.2f}%"
    )

    print(
        f"Val Loss  : {val_loss:.4f}"
    )

    print(
        f"Val Acc   : {val_accuracy:.2f}%"
    )


# ============================================================
# SAVE TRAINING GRAPHS
# ============================================================

import matplotlib.pyplot as plt


os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


epochs = range(
    1,
    NUM_EPOCHS + 1
)


# ============================================================
# LOSS GRAPH
# ============================================================

plt.figure()

plt.plot(
    epochs,
    train_losses,
    label="Training Loss"
)

plt.plot(
    epochs,
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
    "MobileNetV2 Training and Validation Loss"
)

plt.legend()

plt.grid(
    True
)

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "mobilenetv2_training_loss.png"
    )
)

plt.close()


# ============================================================
# ACCURACY GRAPH
# ============================================================

plt.figure()

plt.plot(
    epochs,
    train_accuracies,
    label="Training Accuracy"
)

plt.plot(
    epochs,
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
    "MobileNetV2 Training and Validation Accuracy"
)

plt.legend()

plt.grid(
    True
)

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "mobilenetv2_training_accuracy.png"
    )
)

plt.close()


# ============================================================
# FINAL INFORMATION
# ============================================================

print()
print("=" * 70)

print(
    "TRAINING COMPLETED"
)

print("=" * 70)

print()

print(
    "Best Validation Loss:",
    f"{best_val_loss:.4f}"
)

print()

print(
    "Model saved to:"
)

print(
    os.path.abspath(
        MODEL_PATH
    )
)

print()

print(
    "Training graph:"
)

print(
    os.path.abspath(
        os.path.join(
            RESULTS_DIR,
            "mobilenetv2_training_loss.png"
        )
    )
)

print()

print(
    "Accuracy graph:"
)

print(
    os.path.abspath(
        os.path.join(
            RESULTS_DIR,
            "mobilenetv2_training_accuracy.png"
        )
    )
)

print()
print(
    "STATUS: MOBILENETV2 TRAINING COMPLETED"
)

print("=" * 70)