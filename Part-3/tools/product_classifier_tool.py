"""
Part 3 - Product Image Classification Tool

Loads the exact Part 2 saved model:
    models/product_classifier.pt

and classifies real PNG images from:
    data/sample_images/

The model architecture and preprocessing intentionally match Part 2.
"""

from pathlib import Path
import sys

import torch
from PIL import Image


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

PART2_DIR = ROOT_DIR / "Part-2"

MODEL_PATH = ROOT_DIR / "models" / "product_classifier.pt"

SAMPLE_IMAGE_DIR = ROOT_DIR / "data" / "sample_images"


# ============================================================
# MAKE PART-2 MODULES IMPORTABLE
# ============================================================

if str(PART2_DIR) not in sys.path:
    sys.path.insert(0, str(PART2_DIR))


from model import build_model
from preprocessing import get_preprocessing_transform


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# FASHION-MNIST LABELS
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
# MODEL LOADING
# ============================================================

_MODEL = None
_TRANSFORM = None


def _load_model():
    """
    Load the exact Part 2 saved model.

    The saved artifact is a complete ResNet-18 state_dict.
    """

    global _MODEL

    if _MODEL is not None:
        return _MODEL

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Product classifier model not found:\n{MODEL_PATH}"
        )

    model = build_model()

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(state_dict)

    model = model.to(DEVICE)
    model.eval()

    _MODEL = model

    return _MODEL


# ============================================================
# PREPROCESSING
# ============================================================

def _get_transform():
    """
    Use the exact preprocessing pipeline from Part 2.
    """

    global _TRANSFORM

    if _TRANSFORM is None:
        _TRANSFORM = get_preprocessing_transform()

    return _TRANSFORM


# ============================================================
# PUBLIC TOOL
# ============================================================

def classify_product_image(image_path: str) -> dict:
    """
    Classify one real product image.

    Parameters
    ----------
    image_path:
        Path to a real PNG/JPG image.

    Returns
    -------
    dict
        {
            "image_path": str,
            "category": str,
            "class_index": int,
            "confidence": float
        }
    """

    image_path = Path(image_path)

    # --------------------------------------------------------
    # Validate image
    # --------------------------------------------------------

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    if not image_path.is_file():
        raise ValueError(
            f"Image path is not a file:\n{image_path}"
        )

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = Image.open(image_path).convert("L")

    # --------------------------------------------------------
    # Preprocess using Part 2 pipeline
    # --------------------------------------------------------

    transform = _get_transform()

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(DEVICE)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = _load_model()

    # --------------------------------------------------------
    # Real model inference
    # --------------------------------------------------------

    with torch.no_grad():

        logits = model(image_tensor)

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        confidence_tensor, prediction_tensor = (
            torch.max(
                probabilities,
                dim=1
            )
        )

    class_index = int(
        prediction_tensor.item()
    )

    confidence = float(
        confidence_tensor.item()
    )

    category = CLASS_NAMES[class_index]

    # --------------------------------------------------------
    # Return structured result
    # --------------------------------------------------------

    return {
        "image_path": str(
            image_path.resolve()
        ),
        "category": category,
        "class_index": class_index,
        "confidence": confidence,
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("PART 3 - PRODUCT IMAGE CLASSIFIER TEST")
    print("=" * 70)

    print("\nModel:")
    print(MODEL_PATH)

    print("\nDevice:")
    print(DEVICE)

    print("\nSample directory:")
    print(SAMPLE_IMAGE_DIR)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing model:\n{MODEL_PATH}"
        )

    image_paths = sorted(
        SAMPLE_IMAGE_DIR.glob("*.png")
    )

    if not image_paths:
        raise FileNotFoundError(
            f"No PNG files found in:\n{SAMPLE_IMAGE_DIR}"
        )

    print(
        f"\nFound {len(image_paths)} PNG sample images."
    )

    print("\nPredictions:")
    print("-" * 70)

    for image_path in image_paths:

        result = classify_product_image(
            str(image_path)
        )

        print(
            f"{image_path.name:25s}"
            f" -> "
            f"{result['category']:15s}"
            f" confidence={result['confidence']:.4f}"
        )

    print("\n" + "=" * 70)
    print("PRODUCT CLASSIFIER TOOL TEST: PASSED")
    print("=" * 70)