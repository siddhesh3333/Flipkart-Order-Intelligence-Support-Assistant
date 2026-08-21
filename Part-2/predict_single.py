from pathlib import Path

import torch
from PIL import Image

from model import build_model
from preprocessing import get_preprocessing_transform


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT_DIR / "models" / "product_classifier.pt"
)

SAMPLE_IMAGE_DIR = (
    ROOT_DIR / "data" / "sample_images"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    model = build_model()

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(
        state_dict
    )

    model = model.to(DEVICE)

    model.eval()

    return model


# ============================================================
# PREDICT SINGLE IMAGE
# ============================================================

def predict_image(
    model,
    image_path
):

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = Image.open(
        image_path
    ).convert("L")

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    transform = get_preprocessing_transform()

    image_tensor = transform(image)

    # --------------------------------------------------------
    # Add batch dimension
    # --------------------------------------------------------

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(DEVICE)

    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    with torch.no_grad():

        logits = model(
            image_tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[
            0,
            predicted_class
        ].item()

    return (
        predicted_class,
        confidence
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("FINAL PART-2 MODEL INTEGRATION TEST")
    print("=" * 70)

    print(
        "\nDevice:",
        DEVICE
    )

    print(
        "\nModel:"
    )

    print(
        MODEL_PATH
    )

    # --------------------------------------------------------
    # Verify model exists
    # --------------------------------------------------------

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    print(
        "\n✓ Final model file exists"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    print(
        "✓ Final model loaded successfully"
    )

    # --------------------------------------------------------
    # Find sample images
    # --------------------------------------------------------

    sample_images = sorted(
        SAMPLE_IMAGE_DIR.glob("*.png")
    )

    print(
        f"\nSample images found: "
        f"{len(sample_images)}"
    )

    if len(sample_images) < 5:

        raise RuntimeError(
            "Acceptance criterion requires "
            "at least 5 PNG sample images."
        )

    # --------------------------------------------------------
    # Test every sample image
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SAMPLE IMAGE PREDICTIONS")
    print("=" * 70)

    passed = 0

    for image_path in sample_images:

        # ----------------------------------------------------
        # Extract expected class from filename
        #
        # Example:
        # 07_sneaker.png
        #       ↓
        # expected class = 7
        # ----------------------------------------------------

        expected_class = int(
            image_path.stem.split("_")[0]
        )

        predicted_class, confidence = predict_image(
            model,
            image_path
        )

        is_correct = (
            predicted_class == expected_class
        )

        if is_correct:

            status = "PASS"

            passed += 1

        else:

            status = "FAIL"

        print(
            f"\nFile       : {image_path.name}"
        )

        print(
            f"Expected   : "
            f"{expected_class} - "
            f"{CLASS_NAMES[expected_class]}"
        )

        print(
            f"Predicted  : "
            f"{predicted_class} - "
            f"{CLASS_NAMES[predicted_class]}"
        )

        print(
            f"Confidence : "
            f"{confidence * 100:.2f}%"
        )

        print(
            f"Status     : {status}"
        )

    # --------------------------------------------------------
    # Final verification
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL INTEGRATION RESULT")
    print("=" * 70)

    print(
        f"\nPassed: "
        f"{passed}/{len(sample_images)}"
    )

    print(
        f"Failed: "
        f"{len(sample_images) - passed}"
    )

    if passed == len(sample_images):

        print(
            "\n✓ ALL SAMPLE IMAGE PREDICTIONS PASSED"
        )

        print(
            "✓ FINAL MODEL INTEGRATION PASSED"
        )

    else:

        print(
            "\n⚠ SOME SAMPLE PREDICTIONS FAILED"
        )