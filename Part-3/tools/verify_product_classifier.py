from pathlib import Path
import sys

import torch
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parents[2]
PART2_DIR = ROOT_DIR / "Part-2"

sys.path.insert(0, str(PART2_DIR))

from model import build_model
from preprocessing import get_preprocessing_transform

from product_classifier_tool import classify_product_image


MODEL_PATH = ROOT_DIR / "models" / "product_classifier.pt"

IMAGE_PATH = (
    ROOT_DIR
    / "data"
    / "sample_images"
    / "07_sneaker.png"
)

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

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ------------------------------------------------------------
# DIRECT MODEL
# ------------------------------------------------------------

model = build_model()

state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(state_dict)

model = model.to(DEVICE)
model.eval()

transform = get_preprocessing_transform()

image = Image.open(
    IMAGE_PATH
).convert("L")

tensor = transform(image).unsqueeze(0).to(DEVICE)

with torch.no_grad():

    logits = model(tensor)

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    confidence, prediction = torch.max(
        probabilities,
        dim=1
    )

direct_class_index = int(
    prediction.item()
)

direct_category = CLASS_NAMES[
    direct_class_index
]

direct_confidence = float(
    confidence.item()
)


# ------------------------------------------------------------
# PART 3 TOOL
# ------------------------------------------------------------

tool_result = classify_product_image(
    str(IMAGE_PATH)
)


# ------------------------------------------------------------
# COMPARISON
# ------------------------------------------------------------

print("=" * 70)
print("DIRECT MODEL VS PART 3 TOOL")
print("=" * 70)

print("\nImage:")
print(IMAGE_PATH.name)

print("\nDirect model:")
print("Category  :", direct_category)
print("Class     :", direct_class_index)
print("Confidence:", f"{direct_confidence:.10f}")

print("\nPart 3 tool:")
print("Category  :", tool_result["category"])
print("Class     :", tool_result["class_index"])
print(
    "Confidence:",
    f"{tool_result['confidence']:.10f}"
)

category_match = (
    direct_category
    == tool_result["category"]
)

class_match = (
    direct_class_index
    == tool_result["class_index"]
)

confidence_match = abs(
    direct_confidence
    - tool_result["confidence"]
) < 1e-6


print("\n" + "-" * 70)

print(
    "Category match  :",
    category_match
)

print(
    "Class match     :",
    class_match
)

print(
    "Confidence match:",
    confidence_match
)

if not (
    category_match
    and class_match
    and confidence_match
):
    raise AssertionError(
        "Direct model and Part 3 tool outputs do not match."
    )

print("\n" + "=" * 70)
print("DIRECT MODEL VS TOOL: PASSED")
print("=" * 70)