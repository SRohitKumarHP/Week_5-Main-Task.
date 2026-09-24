import os
import cv2
import yaml
from collections import Counter
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(BASE_DIR, "dataset", "detection")
DATA_YAML = os.path.join(DATASET_DIR, "data.yaml")


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "scratch",
    "dent",
    "rust",
    "crack",
    "hole"
]


# ============================================================
# LOAD YAML
# ============================================================

print("=" * 60)
print("DETECTION DATASET VERIFICATION")
print("=" * 60)

if os.path.exists(DATA_YAML):

    with open(DATA_YAML, "r") as file:
        data = yaml.safe_load(file)

    print("\nDataset YAML loaded successfully.")

    if "names" in data:
        print("Classes from YAML:", data["names"])

else:
    print("\nWARNING: data.yaml not found.")
    print("Using predefined class names.")


# ============================================================
# DATASET SPLITS
# ============================================================

splits = ["train", "val", "test"]

total_images = 0
total_labels = 0

overall_class_count = Counter()

invalid_annotations = []
missing_labels = []
missing_images = []


# ============================================================
# CHECK EACH SPLIT
# ============================================================

for split in splits:

    print("\n" + "-" * 60)
    print(f"{split.upper()} DATASET")
    print("-" * 60)

    image_dir = os.path.join(DATASET_DIR, "images", split)
    label_dir = os.path.join(DATASET_DIR, "labels", split)

    if not os.path.exists(image_dir):

        print("Image directory not found:")
        print(image_dir)

        continue

    if not os.path.exists(label_dir):

        print("Label directory not found:")
        print(label_dir)

        continue

    image_files = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        )
    ]

    label_files = [
        f for f in os.listdir(label_dir)
        if f.lower().endswith(".txt")
    ]

    print("Images:", len(image_files))
    print("Labels:", len(label_files))

    split_class_count = Counter()

    split_images = 0
    split_annotations = 0

    # --------------------------------------------------------
    # CHECK IMAGES
    # --------------------------------------------------------

    for image_file in image_files:

        image_path = os.path.join(image_dir, image_file)

        base_name = os.path.splitext(image_file)[0]

        label_file = base_name + ".txt"
        label_path = os.path.join(label_dir, label_file)

        split_images += 1
        total_images += 1

        # ----------------------------------------------------
        # Check corresponding label
        # ----------------------------------------------------

        if not os.path.exists(label_path):

            missing_labels.append(
                f"{split}: {image_file}"
            )

            continue

        # ----------------------------------------------------
        # Read label file
        # ----------------------------------------------------

        with open(label_path, "r") as file:

            lines = [
                line.strip()
                for line in file
                if line.strip()
            ]

        for line_number, line in enumerate(lines, start=1):

            values = line.split()

            # YOLO format should contain 5 values
            if len(values) != 5:

                invalid_annotations.append(
                    f"{split}/{label_file} "
                    f"line {line_number}: "
                    f"Expected 5 values, got {len(values)}"
                )

                continue

            try:

                class_id = int(values[0])

                x_center = float(values[1])
                y_center = float(values[2])
                width = float(values[3])
                height = float(values[4])

            except ValueError:

                invalid_annotations.append(
                    f"{split}/{label_file} "
                    f"line {line_number}: "
                    f"Non-numeric value"
                )

                continue

            # ------------------------------------------------
            # Check class ID
            # ------------------------------------------------

            if class_id < 0 or class_id >= len(CLASS_NAMES):

                invalid_annotations.append(
                    f"{split}/{label_file} "
                    f"line {line_number}: "
                    f"Invalid class ID {class_id}"
                )

                continue

            # ------------------------------------------------
            # Check normalized coordinates
            # ------------------------------------------------

            values_to_check = [
                x_center,
                y_center,
                width,
                height
            ]

            if not all(0 <= value <= 1 for value in values_to_check):

                invalid_annotations.append(
                    f"{split}/{label_file} "
                    f"line {line_number}: "
                    f"Coordinates outside [0,1]"
                )

                continue

            # ------------------------------------------------
            # Count class
            # ------------------------------------------------

            split_class_count[class_id] += 1
            overall_class_count[class_id] += 1

            split_annotations += 1
            total_labels += 1

    # ========================================================
    # SPLIT RESULTS
    # ========================================================

    print("\nDefect distribution:")

    for class_id, class_name in enumerate(CLASS_NAMES):

        print(
            f"  {class_name:<10}: "
            f"{split_class_count[class_id]}"
        )

    print("\nValid annotations:", split_annotations)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL DATASET SUMMARY")
print("=" * 60)

print("Total images:", total_images)
print("Total valid defect annotations:", total_labels)

print("\nOverall defect distribution:")

for class_id, class_name in enumerate(CLASS_NAMES):

    print(
        f"{class_name:<10}: "
        f"{overall_class_count[class_id]}"
    )


# ============================================================
# ERRORS
# ============================================================

print("\n" + "-" * 60)
print("DATASET ERRORS")
print("-" * 60)

print("Missing label files:", len(missing_labels))
print("Invalid annotations:", len(invalid_annotations))

if missing_labels:

    print("\nMissing labels:")

    for item in missing_labels[:10]:
        print(" ", item)

if invalid_annotations:

    print("\nInvalid annotations:")

    for item in invalid_annotations[:10]:
        print(" ", item)


# ============================================================
# VISUALIZE SAMPLE IMAGES
# ============================================================

print("\n" + "=" * 60)
print("VISUALIZING SAMPLE BOUNDING BOXES")
print("=" * 60)


train_image_dir = os.path.join(
    DATASET_DIR,
    "images",
    "train"
)

train_label_dir = os.path.join(
    DATASET_DIR,
    "labels",
    "train"
)


image_files = [
    f for f in os.listdir(train_image_dir)
    if f.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    )
]


# Take up to 6 samples
sample_files = image_files[:6]


fig, axes = plt.subplots(
    2,
    3,
    figsize=(15, 9)
)

axes = axes.flatten()


for index, image_file in enumerate(sample_files):

    image_path = os.path.join(
        train_image_dir,
        image_file
    )

    label_path = os.path.join(
        train_label_dir,
        os.path.splitext(image_file)[0] + ".txt"
    )

    image = cv2.imread(image_path)

    if image is None:
        continue

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image_height, image_width = image.shape[:2]

    # --------------------------------------------------------
    # Read annotations
    # --------------------------------------------------------

    if os.path.exists(label_path):

        with open(label_path, "r") as file:

            lines = [
                line.strip()
                for line in file
                if line.strip()
            ]

        for line in lines:

            values = line.split()

            if len(values) != 5:
                continue

            class_id = int(values[0])

            x_center = float(values[1])
            y_center = float(values[2])
            width = float(values[3])
            height = float(values[4])

            # Convert normalized YOLO coordinates
            # into pixel coordinates

            x_center_pixel = x_center * image_width
            y_center_pixel = y_center * image_height

            box_width = width * image_width
            box_height = height * image_height

            x1 = int(
                x_center_pixel - box_width / 2
            )

            y1 = int(
                y_center_pixel - box_height / 2
            )

            x2 = int(
                x_center_pixel + box_width / 2
            )

            y2 = int(
                y_center_pixel + box_height / 2
            )

            # Keep coordinates inside image
            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(image_width - 1, x2)
            y2 = min(image_height - 1, y2)

            # Draw bounding box

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            # Class label

            label_text = CLASS_NAMES[class_id]

            cv2.putText(
                image,
                label_text,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    axes[index].imshow(image)

    axes[index].set_title(
        image_file,
        fontsize=9
    )

    axes[index].axis("off")


# Hide unused axes

for index in range(len(sample_files), len(axes)):

    axes[index].axis("off")


plt.tight_layout()

plt.show()


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 60)

if (
    len(missing_labels) == 0
    and len(invalid_annotations) == 0
):

    print("STATUS: DETECTION DATASET READY")

else:

    print("STATUS: DATASET HAS ERRORS")

print("=" * 60)