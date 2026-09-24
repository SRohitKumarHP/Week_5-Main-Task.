import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


# ==========================================================
# CONFIGURATION
# ==========================================================

IMAGE_SIZE = 224
BATCH_SIZE = 8
NUM_EPOCHS = 10
LEARNING_RATE = 0.0001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_ROOT = "dataset/detection/images"
LABEL_ROOT = "dataset/detection/labels"

MODEL_PATH = "models/mobilenetv2_spatial_localization.pth"

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


CLASS_NAMES = [
    "defective",
    "good"
]

DEFECT_NAMES = [
    "scratch",
    "dent",
    "rust",
    "crack",
    "hole"
]


# ==========================================================
# DATASET
# ==========================================================

class MetalDefectDataset(Dataset):

    def __init__(self, split):

        self.image_dir = os.path.join(
            IMAGE_ROOT,
            split
        )

        self.label_dir = os.path.join(
            LABEL_ROOT,
            split
        )

        self.image_files = []

        valid_extensions = (
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp"
        )

        for filename in sorted(os.listdir(self.image_dir)):

            if filename.lower().endswith(valid_extensions):

                self.image_files.append(filename)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),

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


    def __len__(self):

        return len(self.image_files)


    def __getitem__(self, index):

        filename = self.image_files[index]

        image_path = os.path.join(
            self.image_dir,
            filename
        )

        label_filename = os.path.splitext(
            filename
        )[0] + ".txt"

        label_path = os.path.join(
            self.label_dir,
            label_filename
        )

        # --------------------------------------------------
        # Read image
        # --------------------------------------------------

        image = cv2.imread(image_path)

        if image is None:

            raise RuntimeError(
                f"Unable to read image: {image_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # --------------------------------------------------
        # Default values
        #
        # Good image:
        # class = 1
        # defect class = 0
        # bbox = 0,0,0,0
        # --------------------------------------------------

        classification_label = 1

        defect_label = 0

        bbox = np.array(
            [0.0, 0.0, 0.0, 0.0],
            dtype=np.float32
        )

        # --------------------------------------------------
        # Read YOLO annotation
        # --------------------------------------------------

        if os.path.exists(label_path):

            with open(
                label_path,
                "r"
            ) as file:

                lines = [
                    line.strip()
                    for line in file
                    if line.strip()
                ]

            if len(lines) > 0:

                # This dataset contains one defect
                # annotation per image.

                parts = lines[0].split()

                if len(parts) == 5:

                    defect_class = int(parts[0])

                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    classification_label = 0

                    defect_label = defect_class

                    bbox = np.array(
                        [
                            x_center,
                            y_center,
                            width,
                            height
                        ],
                        dtype=np.float32
                    )

        # --------------------------------------------------
        # Transform image
        # --------------------------------------------------

        image = self.transform(image)

        return (
            image,
            torch.tensor(
                classification_label,
                dtype=torch.long
            ),
            torch.tensor(
                defect_label,
                dtype=torch.long
            ),
            torch.tensor(
                bbox,
                dtype=torch.float32
            )
        )


# ==========================================================
# MODEL
# ==========================================================

class MobileNetV2SpatialLocalization(nn.Module):

    def __init__(
        self,
        num_classes=2,
        num_defect_types=5
    ):

        super().__init__()

        weights = MobileNet_V2_Weights.DEFAULT

        backbone = mobilenet_v2(
            weights=weights
        )

        self.features = backbone.features

        feature_channels = 1280

        # --------------------------------------------------
        # Classification Head
        # --------------------------------------------------

        self.classification_head = nn.Sequential(

            nn.AdaptiveAvgPool2d((1, 1)),

            nn.Flatten(),

            nn.Linear(
                feature_channels,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                128,
                num_classes
            )
        )

        # --------------------------------------------------
        # Defect Type Head
        # --------------------------------------------------

        self.defect_head = nn.Sequential(

            nn.AdaptiveAvgPool2d((1, 1)),

            nn.Flatten(),

            nn.Linear(
                feature_channels,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                128,
                num_defect_types
            )
        )

        # --------------------------------------------------
        # Bounding Box Head
        # --------------------------------------------------

        self.localization_head = nn.Sequential(

            nn.Conv2d(
                feature_channels,
                256,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                256,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.AdaptiveAvgPool2d(
                (4, 4)
            ),

            nn.Flatten(),

            nn.Linear(
                128 * 4 * 4,
                256
            ),

            nn.ReLU(),

            nn.Linear(
                256,
                4
            ),

            nn.Sigmoid()
        )


    def forward(self, x):

        features = self.features(x)

        class_output = self.classification_head(
            features
        )

        defect_output = self.defect_head(
            features
        )

        bbox_output = self.localization_head(
            features
        )

        return (
            class_output,
            defect_output,
            bbox_output
        )


# ==========================================================
# CREATE DATASETS
# ==========================================================

print("=" * 70)
print("MOBILENETV2 SPATIAL LOCALIZATION TRAINING")
print("=" * 70)

print()
print("Device:", DEVICE)
print("Image Size:", IMAGE_SIZE)
print("Batch Size:", BATCH_SIZE)
print("Epochs:", NUM_EPOCHS)
print("Learning Rate:", LEARNING_RATE)

print()

train_dataset = MetalDefectDataset(
    "train"
)

val_dataset = MetalDefectDataset(
    "val"
)

print(
    "Training images:",
    len(train_dataset)
)

print(
    "Validation images:",
    len(val_dataset)
)

# ----------------------------------------------------------
# Data loaders
# ----------------------------------------------------------

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


# ==========================================================
# CREATE MODEL
# ==========================================================

model = MobileNetV2SpatialLocalization(
    num_classes=2,
    num_defect_types=5
)

model = model.to(DEVICE)

print()
print("MobileNetV2 localization model created.")


# ==========================================================
# LOSS FUNCTIONS
# ==========================================================

classification_loss_function = nn.CrossEntropyLoss()

defect_loss_function = nn.CrossEntropyLoss()

bbox_loss_function = nn.SmoothL1Loss()


# ==========================================================
# OPTIMIZER
# ==========================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)


# ==========================================================
# TRAINING HISTORY
# ==========================================================

train_losses = []
val_losses = []

train_class_acc = []
val_class_acc = []

train_defect_acc = []
val_defect_acc = []

train_bbox_loss = []
val_bbox_loss = []


best_val_loss = float("inf")


# ==========================================================
# TRAINING LOOP
# ==========================================================

for epoch in range(NUM_EPOCHS):

    # ------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------

    model.train()

    running_loss = 0.0

    correct_class = 0
    total_class = 0

    correct_defect = 0
    total_defect = 0

    running_bbox_loss = 0.0

    for (
        images,
        class_labels,
        defect_labels,
        bbox_targets
    ) in train_loader:

        images = images.to(DEVICE)

        class_labels = class_labels.to(
            DEVICE
        )

        defect_labels = defect_labels.to(
            DEVICE
        )

        bbox_targets = bbox_targets.to(
            DEVICE
        )

        optimizer.zero_grad()

        (
            class_output,
            defect_output,
            bbox_output
        ) = model(images)

        # --------------------------------------------------
        # Classification loss
        # --------------------------------------------------

        class_loss = classification_loss_function(
            class_output,
            class_labels
        )

        # --------------------------------------------------
        # Defect type loss
        #
        # Only defective images contain a valid defect type.
        # --------------------------------------------------

        defective_mask = (
            class_labels == 0
        )

        if defective_mask.any():

            defect_loss = defect_loss_function(
                defect_output[defective_mask],
                defect_labels[defective_mask]
            )

            bbox_loss = bbox_loss_function(
                bbox_output[defective_mask],
                bbox_targets[defective_mask]
            )

        else:

            defect_loss = torch.tensor(
                0.0,
                device=DEVICE
            )

            bbox_loss = torch.tensor(
                0.0,
                device=DEVICE
            )

        # --------------------------------------------------
        # Total loss
        # --------------------------------------------------

        total_loss = (
            class_loss
            + defect_loss
            + bbox_loss
        )

        total_loss.backward()

        optimizer.step()

        # --------------------------------------------------
        # Statistics
        # --------------------------------------------------

        running_loss += (
            total_loss.item()
            * images.size(0)
        )

        class_predictions = torch.argmax(
            class_output,
            dim=1
        )

        correct_class += (
            class_predictions == class_labels
        ).sum().item()

        total_class += images.size(0)

        # Defect accuracy
        if defective_mask.any():

            defect_predictions = torch.argmax(
                defect_output[defective_mask],
                dim=1
            )

            correct_defect += (
                defect_predictions
                == defect_labels[defective_mask]
            ).sum().item()

            total_defect += (
                defective_mask.sum().item()
            )

        running_bbox_loss += (
            bbox_loss.item()
            * images.size(0)
        )

    # ------------------------------------------------------
    # Training metrics
    # ------------------------------------------------------

    epoch_train_loss = (
        running_loss
        / len(train_dataset)
    )

    epoch_train_class_acc = (
        correct_class
        / total_class
        * 100
    )

    if total_defect > 0:

        epoch_train_defect_acc = (
            correct_defect
            / total_defect
            * 100
        )

    else:

        epoch_train_defect_acc = 0.0

    epoch_train_bbox_loss = (
        running_bbox_loss
        / len(train_dataset)
    )


    # ======================================================
    # VALIDATION
    # ======================================================

    model.eval()

    validation_loss = 0.0

    val_correct_class = 0
    val_total_class = 0

    val_correct_defect = 0
    val_total_defect = 0

    validation_bbox_loss = 0.0


    with torch.no_grad():

        for (
            images,
            class_labels,
            defect_labels,
            bbox_targets
        ) in val_loader:

            images = images.to(DEVICE)

            class_labels = class_labels.to(
                DEVICE
            )

            defect_labels = defect_labels.to(
                DEVICE
            )

            bbox_targets = bbox_targets.to(
                DEVICE
            )

            (
                class_output,
                defect_output,
                bbox_output
            ) = model(images)

            class_loss = classification_loss_function(
                class_output,
                class_labels
            )

            defective_mask = (
                class_labels == 0
            )

            if defective_mask.any():

                defect_loss = defect_loss_function(
                    defect_output[defective_mask],
                    defect_labels[defective_mask]
                )

                bbox_loss = bbox_loss_function(
                    bbox_output[defective_mask],
                    bbox_targets[defective_mask]
                )

            else:

                defect_loss = torch.tensor(
                    0.0,
                    device=DEVICE
                )

                bbox_loss = torch.tensor(
                    0.0,
                    device=DEVICE
                )

            total_loss = (
                class_loss
                + defect_loss
                + bbox_loss
            )

            validation_loss += (
                total_loss.item()
                * images.size(0)
            )

            validation_bbox_loss += (
                bbox_loss.item()
                * images.size(0)
            )

            # Classification
            class_predictions = torch.argmax(
                class_output,
                dim=1
            )

            val_correct_class += (
                class_predictions
                == class_labels
            ).sum().item()

            val_total_class += (
                images.size(0)
            )

            # Defect type
            if defective_mask.any():

                defect_predictions = torch.argmax(
                    defect_output[defective_mask],
                    dim=1
                )

                val_correct_defect += (
                    defect_predictions
                    == defect_labels[defective_mask]
                ).sum().item()

                val_total_defect += (
                    defective_mask.sum().item()
                )


    # ------------------------------------------------------
    # Validation metrics
    # ------------------------------------------------------

    epoch_val_loss = (
        validation_loss
        / len(val_dataset)
    )

    epoch_val_class_acc = (
        val_correct_class
        / val_total_class
        * 100
    )

    if val_total_defect > 0:

        epoch_val_defect_acc = (
            val_correct_defect
            / val_total_defect
            * 100
        )

    else:

        epoch_val_defect_acc = 0.0

    epoch_val_bbox_loss = (
        validation_bbox_loss
        / len(val_dataset)
    )


    scheduler.step(
        epoch_val_loss
    )


    # ------------------------------------------------------
    # Save history
    # ------------------------------------------------------

    train_losses.append(
        epoch_train_loss
    )

    val_losses.append(
        epoch_val_loss
    )

    train_class_acc.append(
        epoch_train_class_acc
    )

    val_class_acc.append(
        epoch_val_class_acc
    )

    train_defect_acc.append(
        epoch_train_defect_acc
    )

    val_defect_acc.append(
        epoch_val_defect_acc
    )

    train_bbox_loss.append(
        epoch_train_bbox_loss
    )

    val_bbox_loss.append(
        epoch_val_bbox_loss
    )


    # ------------------------------------------------------
    # Print epoch results
    # ------------------------------------------------------

    print()
    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS}"
    )

    print(
        f"Train Loss: {epoch_train_loss:.4f}"
    )

    print(
        f"Val Loss: {epoch_val_loss:.4f}"
    )

    print(
        f"Train Class Acc: "
        f"{epoch_train_class_acc:.2f}%"
    )

    print(
        f"Val Class Acc: "
        f"{epoch_val_class_acc:.2f}%"
    )

    print(
        f"Train Defect Acc: "
        f"{epoch_train_defect_acc:.2f}%"
    )

    print(
        f"Val Defect Acc: "
        f"{epoch_val_defect_acc:.2f}%"
    )

    print(
        f"Train BBox Loss: "
        f"{epoch_train_bbox_loss:.4f}"
    )

    print(
        f"Val BBox Loss: "
        f"{epoch_val_bbox_loss:.4f}"
    )


    # ------------------------------------------------------
    # Save best model
    # ------------------------------------------------------

    if epoch_val_loss < best_val_loss:

        best_val_loss = epoch_val_loss

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "class_names":
                    CLASS_NAMES,

                "defect_names":
                    DEFECT_NAMES,

                "image_size":
                    IMAGE_SIZE
            },
            MODEL_PATH
        )

        print(
            "Best model saved."
        )


# ==========================================================
# SAVE TRAINING HISTORY
# ==========================================================

history_path = (
    "results/"
    "mobilenetv2_spatial_training.txt"
)

with open(
    history_path,
    "w"
) as file:

    file.write(
        "MOBILENETV2 SPATIAL LOCALIZATION "
        "TRAINING RESULTS\n"
    )

    file.write(
        "=" * 70
        + "\n\n"
    )

    file.write(
        f"Best Validation Loss: "
        f"{best_val_loss:.4f}\n\n"
    )

    for i in range(NUM_EPOCHS):

        file.write(
            f"Epoch {i + 1}\n"
        )

        file.write(
            f"Train Loss: "
            f"{train_losses[i]:.4f}\n"
        )

        file.write(
            f"Val Loss: "
            f"{val_losses[i]:.4f}\n"
        )

        file.write(
            f"Train Classification Accuracy: "
            f"{train_class_acc[i]:.2f}%\n"
        )

        file.write(
            f"Val Classification Accuracy: "
            f"{val_class_acc[i]:.2f}%\n"
        )

        file.write(
            f"Train Defect Accuracy: "
            f"{train_defect_acc[i]:.2f}%\n"
        )

        file.write(
            f"Val Defect Accuracy: "
            f"{val_defect_acc[i]:.2f}%\n"
        )

        file.write(
            f"Train BBox Loss: "
            f"{train_bbox_loss[i]:.4f}\n"
        )

        file.write(
            f"Val BBox Loss: "
            f"{val_bbox_loss[i]:.4f}\n"
        )

        file.write(
            "\n"
        )


print()
print("=" * 70)
print("TRAINING COMPLETED")
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
    os.path.abspath(MODEL_PATH)
)

print()
print(
    "Training report saved to:"
)

print(
    os.path.abspath(history_path)
)

print()
print(
    "STATUS: MOBILENETV2 SPATIAL "
    "LOCALIZATION TRAINING COMPLETED"
)

print("=" * 70)