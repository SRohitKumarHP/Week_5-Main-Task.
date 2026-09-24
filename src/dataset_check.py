from pathlib import Path
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLASSIFICATION_DIR = PROJECT_ROOT / "dataset" / "classification"

SPLITS = ["train", "val", "test"]

CLASSES = ["good", "defective"]


# ============================================================
# IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# HELPER FUNCTION
# ============================================================

def get_images(folder):

    if not folder.exists():
        return []

    return [
        file
        for file in folder.iterdir()
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    ]


# ============================================================
# CHECK IMAGE
# ============================================================

def check_image(image_path):

    try:

        with Image.open(image_path) as image:

            image.verify()

        return True

    except Exception:

        return False


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("             METAL DEFECT DATASET CHECK")
    print("=" * 70)

    print()

    print("Project Directory:")
    print(PROJECT_ROOT)

    print()

    print("Dataset Directory:")
    print(CLASSIFICATION_DIR)

    print()

    if not CLASSIFICATION_DIR.exists():

        print("ERROR: Classification dataset directory not found.")

        return

    total_images = 0
    total_good = 0
    total_defective = 0

    corrupted_images = []

    print("=" * 70)
    print("DATASET INFORMATION")
    print("=" * 70)

    print()

    # --------------------------------------------------------
    # CHECK EACH SPLIT
    # --------------------------------------------------------

    for split in SPLITS:

        print("-" * 70)

        print(f"{split.upper()} DATASET")

        print("-" * 70)

        split_total = 0

        for class_name in CLASSES:

            class_dir = CLASSIFICATION_DIR / split / class_name

            images = get_images(class_dir)

            image_count = len(images)

            print(
                f"{class_name.capitalize():12} : "
                f"{image_count:4} images"
            )

            split_total += image_count

            # Count overall classes

            if class_name == "good":

                total_good += image_count

            else:

                total_defective += image_count

            # Check every image

            for image_path in images:

                if not check_image(image_path):

                    corrupted_images.append(image_path)

        print()

        print(
            f"Total {split.capitalize():8} : "
            f"{split_total:4} images"
        )

        print()

        total_images += split_total

    # ========================================================
    # OVERALL DATASET
    # ========================================================

    print("=" * 70)
    print("OVERALL DATASET")
    print("=" * 70)

    print()

    print(f"Total Images     : {total_images}")

    print(f"Good Images      : {total_good}")

    print(f"Defective Images : {total_defective}")

    print()

    # ========================================================
    # CLASS BALANCE
    # ========================================================

    if total_images > 0:

        good_percentage = (
            total_good / total_images
        ) * 100

        defective_percentage = (
            total_defective / total_images
        ) * 100

        print("CLASS DISTRIBUTION")

        print("-" * 70)

        print(
            f"Good       : "
            f"{good_percentage:.2f}%"
        )

        print(
            f"Defective  : "
            f"{defective_percentage:.2f}%"
        )

        print()

    # ========================================================
    # CORRUPTED IMAGES
    # ========================================================

    print("=" * 70)
    print("IMAGE VALIDATION")
    print("=" * 70)

    print()

    if len(corrupted_images) == 0:

        print("All images are valid.")

    else:

        print(
            f"Corrupted images found: "
            f"{len(corrupted_images)}"
        )

        print()

        for image in corrupted_images:

            print(image)

    print()

    # ========================================================
    # FINAL STATUS
    # ========================================================

    print("=" * 70)
    print("FINAL DATASET STATUS")
    print("=" * 70)

    print()

    if (
        total_images > 0
        and total_good > 0
        and total_defective > 0
        and len(corrupted_images) == 0
    ):

        print("STATUS: DATASET READY")

        print()

        print(
            "The dataset can be used for "
            "ResNet18 training."
        )

    else:

        print("STATUS: DATASET NEEDS ATTENTION")

        print()

        print(
            "Please check the dataset folders "
            "and image files."

        )

    print()

    print("=" * 70)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()