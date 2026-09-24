import os
import cv2
import torch
from PIL import Image
from torchvision import transforms

from resnet18_spatial_localization import ResNet18SpatialLocalization


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIRECTORY = "input"

MODEL_PATH = "models/resnet18_spatial_localization.pth"

OUTPUT_DIRECTORY = "output"


IMAGE_SIZE = 224


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


SUPPORTED_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 70)
print("RESNET18 MULTI-IMAGE METAL DEFECT INSPECTION")
print("=" * 70)

print()
print("Device:", device)


# ============================================================
# CHECK DIRECTORIES
# ============================================================

if not os.path.exists(INPUT_DIRECTORY):

    print()
    print("ERROR: Input directory not found.")
    print(
        "Expected:",
        os.path.abspath(INPUT_DIRECTORY)
    )

    exit()


if not os.path.exists(MODEL_PATH):

    print()
    print("ERROR: Trained model not found.")
    print(
        "Expected:",
        os.path.abspath(MODEL_PATH)
    )

    exit()


# Create output directory

os.makedirs(
    OUTPUT_DIRECTORY,
    exist_ok=True
)


# ============================================================
# FIND ALL INPUT IMAGES
# ============================================================

image_files = []


for filename in os.listdir(INPUT_DIRECTORY):

    file_path = os.path.join(
        INPUT_DIRECTORY,
        filename
    )


    if os.path.isfile(file_path):

        if filename.lower().endswith(
            SUPPORTED_EXTENSIONS
        ):

            image_files.append(
                file_path
            )


# Sort images alphabetically

image_files.sort()


# ============================================================
# CHECK IMAGE COUNT
# ============================================================

if len(image_files) == 0:

    print()
    print("ERROR: No images found in input folder.")

    print()
    print("Supported formats:")

    print(
        "JPG, JPEG, PNG, BMP, WEBP"
    )

    exit()


print()
print("=" * 70)
print("INPUT DATASET")
print("=" * 70)

print()
print(
    "Images found:",
    len(image_files)
)


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING TRAINED RESNET18 MODEL")
print("=" * 70)


print()
print("Creating model...")


model = ResNet18SpatialLocalization()


model.to(device)


print(
    "Loading checkpoint..."
)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):

    print(
        "Checkpoint format detected."
    )


    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )


    # Load metadata if available

    if "class_names" in checkpoint:

        CLASS_NAMES = checkpoint[
            "class_names"
        ]


    if "defect_names" in checkpoint:

        DEFECT_NAMES = checkpoint[
            "defect_names"
        ]


    if "image_size" in checkpoint:

        IMAGE_SIZE = checkpoint[
            "image_size"
        ]


else:

    print(
        "Raw state_dict format detected."
    )


    model.load_state_dict(
        checkpoint
    )


model.eval()


print()
print(
    "Model loaded successfully."
)


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([

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
# SUMMARY VARIABLES
# ============================================================

total_images = 0

good_count = 0

defective_count = 0


defect_counts = {

    "scratch": 0,

    "dent": 0,

    "rust": 0,

    "crack": 0,

    "hole": 0
}


failed_images = []


# ============================================================
# PROCESS EVERY IMAGE
# ============================================================

for image_index, image_path in enumerate(
    image_files,
    start=1
):


    print()
    print("=" * 70)

    print(
        f"IMAGE {image_index} / {len(image_files)}"
    )

    print("=" * 70)


    filename = os.path.basename(
        image_path
    )


    print()
    print(
        "Image:",
        filename
    )


    # ========================================================
    # READ IMAGE
    # ========================================================

    image = cv2.imread(
        image_path
    )


    if image is None:

        print(
            "ERROR: Could not read image."
        )

        failed_images.append(
            filename
        )

        continue


    total_images += 1


    original_image = image.copy()


    original_height, original_width = (
        image.shape[:2]
    )


    print(
        "Size:",
        f"{original_width} x {original_height}"
    )


    # ========================================================
    # PREPROCESS
    # ========================================================

    rgb_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


    pil_image = Image.fromarray(
        rgb_image
    )


    input_tensor = transform(
        pil_image
    )


    input_tensor = input_tensor.unsqueeze(
        0
    )


    input_tensor = input_tensor.to(
        device
    )


    # ========================================================
    # MODEL INFERENCE
    # ========================================================

    with torch.no_grad():

        (
            classification_output,
            defect_output,
            bbox_output
        ) = model(
            input_tensor
        )


    # ========================================================
    # CLASSIFICATION
    # ========================================================

    class_probabilities = torch.softmax(
        classification_output,
        dim=1
    )


    class_confidence, class_prediction = (
        torch.max(
            class_probabilities,
            dim=1
        )
    )


    class_prediction = (
        class_prediction.item()
    )


    class_confidence = (
        class_confidence.item()
    )


    predicted_class = CLASS_NAMES[
        class_prediction
    ]


    print()
    print(
        "Classification:",
        predicted_class.upper()
    )


    print(
        "Confidence:",
        f"{class_confidence * 100:.2f}%"
    )


    # ========================================================
    # GOOD IMAGE
    # ========================================================

    if predicted_class == "good":

        good_count += 1


        print()
        print(
            "Result: GOOD"
        )


        # ----------------------------------------------------
        # DRAW GOOD RESULT
        # ----------------------------------------------------

        cv2.putText(

            original_image,

            "GOOD",

            (30, 50),

            cv2.FONT_HERSHEY_SIMPLEX,

            1.2,

            (0, 255, 0),

            3
        )


        cv2.putText(

            original_image,

            (
                f"Confidence: "
                f"{class_confidence * 100:.2f}%"
            ),

            (30, 90),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (0, 255, 0),

            2
        )


    # ========================================================
    # DEFECTIVE IMAGE
    # ========================================================

    else:

        defective_count += 1


        # ====================================================
        # DEFECT TYPE
        # ====================================================

        defect_probabilities = torch.softmax(

            defect_output,

            dim=1
        )


        defect_confidence, defect_prediction = (
            torch.max(
                defect_probabilities,
                dim=1
            )
        )


        defect_prediction = (
            defect_prediction.item()
        )


        defect_confidence = (
            defect_confidence.item()
        )


        predicted_defect = DEFECT_NAMES[
            defect_prediction
        ]


        defect_counts[
            predicted_defect
        ] += 1


        print()
        print(
            "Result: DEFECTIVE"
        )


        print(
            "Defect:",
            predicted_defect.upper()
        )


        print(
            "Defect Confidence:",
            f"{defect_confidence * 100:.2f}%"
        )


        # ====================================================
        # BOUNDING BOX
        # ====================================================

        bbox = (
            bbox_output[0]
            .cpu()
            .numpy()
        )


        x_center = float(
            bbox[0]
        )


        y_center = float(
            bbox[1]
        )


        box_width = float(
            bbox[2]
        )


        box_height = float(
            bbox[3]
        )


        # ====================================================
        # CONVERT NORMALIZED BOX
        # TO PIXEL COORDINATES
        # ====================================================

        x1 = int(

            (
                x_center
                - box_width / 2
            )
            * original_width
        )


        y1 = int(

            (
                y_center
                - box_height / 2
            )
            * original_height
        )


        x2 = int(

            (
                x_center
                + box_width / 2
            )
            * original_width
        )


        y2 = int(

            (
                y_center
                + box_height / 2
            )
            * original_height
        )


        # ====================================================
        # CLAMP BOX
        # ====================================================

        x1 = max(
            0,
            min(
                x1,
                original_width - 1
            )
        )


        y1 = max(
            0,
            min(
                y1,
                original_height - 1
            )
        )


        x2 = max(
            0,
            min(
                x2,
                original_width - 1
            )
        )


        y2 = max(
            0,
            min(
                y2,
                original_height - 1
            )
        )


        print()
        print(
            "Bounding Box:"
        )


        print(
            f"x1={x1}, "
            f"y1={y1}, "
            f"x2={x2}, "
            f"y2={y2}"
        )


        # ====================================================
        # DRAW BOUNDING BOX
        # ====================================================

        cv2.rectangle(

            original_image,

            (x1, y1),

            (x2, y2),

            (0, 0, 255),

            3
        )


        # ====================================================
        # DRAW DEFECT LABEL
        # ====================================================

        label = (

            f"{predicted_defect.upper()} "

            f"{defect_confidence * 100:.1f}%"
        )


        label_y = max(
            30,
            y1 - 10
        )


        cv2.putText(

            original_image,

            label,

            (x1, label_y),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (0, 0, 255),

            2
        )


        # ====================================================
        # DRAW GENERAL STATUS
        # ====================================================

        cv2.putText(

            original_image,

            "DEFECTIVE",

            (30, 50),

            cv2.FONT_HERSHEY_SIMPLEX,

            1.2,

            (0, 0, 255),

            3
        )


        cv2.putText(

            original_image,

            f"Defect: {predicted_defect}",

            (30, 90),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (0, 0, 255),

            2
        )


    # ========================================================
    # CREATE OUTPUT FILE NAME
    # ========================================================

    base_name = os.path.splitext(
        filename
    )[0]


    output_filename = (
        base_name
        + "_result.jpg"
    )


    output_path = os.path.join(

        OUTPUT_DIRECTORY,

        output_filename
    )


    # ========================================================
    # SAVE RESULT
    # ========================================================

    success = cv2.imwrite(

        output_path,

        original_image
    )


    if success:

        print()
        print(
            "Saved:",
            output_path
        )

    else:

        print()
        print(
            "ERROR: Could not save result."
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print()
print("=" * 70)
print("FINAL INSPECTION SUMMARY")
print("=" * 70)


print()
print(
    "Total Images Processed:",
    total_images
)


print()
print(
    "Good Parts:",
    good_count
)


print(
    "Defective Parts:",
    defective_count
)


print()
print(
    "Defect Type Distribution:"
)


print(
    "Scratch:",
    defect_counts["scratch"]
)


print(
    "Dent    :",
    defect_counts["dent"]
)


print(
    "Rust    :",
    defect_counts["rust"]
)


print(
    "Crack   :",
    defect_counts["crack"]
)


print(
    "Hole    :",
    defect_counts["hole"]
)


# ============================================================
# FAILED IMAGES
# ============================================================

if len(failed_images) > 0:

    print()
    print(
        "Images that could not be processed:"
    )


    for filename in failed_images:

        print(
            "-",
            filename
        )


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

print()
print(
    "Results saved to:"
)


print(
    os.path.abspath(
        OUTPUT_DIRECTORY
    )
)


# ============================================================
# STATUS
# ============================================================

print()
print("=" * 70)

print(
    "STATUS: MULTI-IMAGE INSPECTION COMPLETED"
)

print("=" * 70)