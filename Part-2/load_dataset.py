from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import datasets


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data" / "fashion_mnist"

VALIDATION_SIZE = 5000


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
# LOAD DATASET
# ============================================================

def load_fashion_mnist():

    print("=" * 60)
    print("Fashion-MNIST Dataset")
    print("=" * 60)

    # Original 60,000-image training split
    train_dataset = datasets.FashionMNIST(
        root=str(DATA_DIR),
        train=True,
        download=True
    )

    # Original 10,000-image test split
    test_dataset = datasets.FashionMNIST(
        root=str(DATA_DIR),
        train=False,
        download=True
    )

    print("\nOriginal dataset:")
    print(f"Training images : {len(train_dataset)}")
    print(f"Test images     : {len(test_dataset)}")

    # --------------------------------------------------------
    # Create stratified train/validation split
    # --------------------------------------------------------

    all_train_indices = np.arange(
        len(train_dataset)
    )

    train_targets = train_dataset.targets.numpy()

    train_indices, validation_indices = train_test_split(
        all_train_indices,
        test_size=VALIDATION_SIZE,
        stratify=train_targets,
        random_state=RANDOM_STATE
    )

    print("\nFinal split:")
    print(f"Training   : {len(train_indices)}")
    print(f"Validation : {len(validation_indices)}")
    print(f"Test       : {len(test_dataset)}")

    # --------------------------------------------------------
    # Verify class distribution
    # --------------------------------------------------------

    print("\nTraining class distribution:")
    print("-" * 60)

    for class_id, class_name in enumerate(CLASS_NAMES):

        count = np.sum(
            train_targets[train_indices] == class_id
        )

        print(
            f"{class_id} - "
            f"{class_name:<15} : "
            f"{count}"
        )

    print("\nValidation class distribution:")
    print("-" * 60)

    for class_id, class_name in enumerate(CLASS_NAMES):

        count = np.sum(
            train_targets[validation_indices] == class_id
        )

        print(
            f"{class_id} - "
            f"{class_name:<15} : "
            f"{count}"
        )

    return (
        train_dataset,
        test_dataset,
        train_indices,
        validation_indices
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    load_fashion_mnist()