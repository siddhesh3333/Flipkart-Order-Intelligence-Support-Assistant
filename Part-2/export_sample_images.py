from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import datasets


# ============================================================
# CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data" / "fashion_mnist"

OUTPUT_DIR = ROOT_DIR / "data" / "sample_images"


CLASS_NAMES = [
    "t-shirt-top",
    "trouser",
    "pullover",
    "dress",
    "coat",
    "sandal",
    "shirt",
    "sneaker",
    "bag",
    "ankle-boot",
]


# ============================================================
# LOAD TEST DATASET
# ============================================================

def load_test_dataset():

    test_dataset = datasets.FashionMNIST(
        root=str(DATA_DIR),
        train=False,
        download=False
    )

    return test_dataset


# ============================================================
# SELECT ONE REAL IMAGE PER CLASS
# ============================================================

def select_samples(test_dataset):

    selected = {}

    for index in range(len(test_dataset)):

        image, label = test_dataset[index]

        if label not in selected:

            selected[label] = (
                index,
                image
            )

        # Stop once we have all 10 classes
        if len(selected) == 10:
            break

    return selected


# ============================================================
# EXPORT PNG FILES
# ============================================================

def export_samples(test_dataset, selected_samples):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    exported_files = []

    for label in sorted(selected_samples):

        index, image = selected_samples[label]

        # ----------------------------------------------------
        # Convert PIL image to numpy array
        # ----------------------------------------------------

        image_array = np.array(image)

        # ----------------------------------------------------
        # Convert array back into an actual image object
        # ----------------------------------------------------

        output_image = Image.fromarray(
            image_array
        )

        filename = (
            f"{label:02d}_"
            f"{CLASS_NAMES[label]}.png"
        )

        output_path = OUTPUT_DIR / filename

        output_image.save(
            output_path
        )

        exported_files.append(
            output_path
        )

        print(
            f"Class {label}: "
            f"{CLASS_NAMES[label]:15s} "
            f"| Test index: {index:5d} "
            f"| Saved: {filename}"
        )

    return exported_files


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Fashion-MNIST Test Sample Export")
    print("=" * 60)

    test_dataset = load_test_dataset()

    print(
        "\nTest dataset size:",
        len(test_dataset)
    )

    selected_samples = select_samples(
        test_dataset
    )

    print(
        "\nSelected samples:",
        len(selected_samples)
    )

    exported_files = export_samples(
        test_dataset,
        selected_samples
    )

    print("\n" + "=" * 60)
    print("EXPORT COMPLETE")
    print("=" * 60)

    print(
        "\nExported PNG files:",
        len(exported_files)
    )

    print(
        "\nOutput directory:"
    )

    print(OUTPUT_DIR)

    print("\nFiles:")

    for path in exported_files:
        print(
            " -",
            path.name
        )