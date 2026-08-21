from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from load_dataset import load_fashion_mnist
from preprocessing import get_preprocessing_transform
from model import build_model


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

BATCH_SIZE = 32

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data" / "fashion_mnist"

FEATURE_DIR = ROOT_DIR / "data" / "features"


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD DATASETS
# ============================================================

def create_datasets():

    # Get the same split created in Task 1
    (
        train_dataset,
        test_dataset,
        train_indices,
        validation_indices
    ) = load_fashion_mnist()

    # IMPORTANT:
    # Use the preprocessing pipeline created in Task 2.
    transform = get_preprocessing_transform()

    # Reload Fashion-MNIST with the transform attached.
    transformed_dataset = datasets.FashionMNIST(
        root=str(DATA_DIR),
        train=True,
        transform=transform,
        download=False
    )

    # Create training and validation subsets
    train_subset = Subset(
        transformed_dataset,
        train_indices
    )

    validation_subset = Subset(
        transformed_dataset,
        validation_indices
    )

    return train_subset, validation_subset


# ============================================================
# EXTRACT FEATURES
# ============================================================

def extract_features(model, dataset):

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    all_features = []
    all_labels = []

    model.eval()

    with torch.no_grad():

        for batch_images, batch_labels in loader:

            batch_images = batch_images.to(DEVICE)

            features = model(
                batch_images
            )

            all_features.append(
                features.cpu()
            )

            all_labels.append(
                batch_labels
            )

    features = torch.cat(
        all_features,
        dim=0
    )

    labels = torch.cat(
        all_labels,
        dim=0
    )

    return features, labels


# ============================================================
# SAVE FEATURES
# ============================================================

def save_features(
    features,
    labels,
    feature_path,
    label_path
):

    feature_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        features,
        feature_path
    )

    torch.save(
        labels,
        label_path
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Fashion-MNIST Feature Extraction")
    print("=" * 60)

    print("\nDevice:", DEVICE)
    print("Batch size:", BATCH_SIZE)

    # --------------------------------------------------------
    # Create datasets
    # --------------------------------------------------------

    train_dataset, validation_dataset = create_datasets()

    print("\nDataset sizes:")
    print("Training   :", len(train_dataset))
    print("Validation :", len(validation_dataset))

    # --------------------------------------------------------
    # Build pretrained model
    # --------------------------------------------------------

    model = build_model()

    # --------------------------------------------------------
    # Remove classifier
    #
    # ResNet-18:
    #
    # 512 features → classifier → 10 classes
    #
    # We want:
    #
    # 512 features
    # --------------------------------------------------------

    model.fc = torch.nn.Identity()

    model = model.to(DEVICE)

    # --------------------------------------------------------
    # Extract training features
    # --------------------------------------------------------

    print("\nExtracting training features...")

    train_features, train_labels = extract_features(
        model,
        train_dataset
    )

    print(
        "Training feature shape:",
        train_features.shape
    )

    print(
        "Training label shape:",
        train_labels.shape
    )

    # --------------------------------------------------------
    # Extract validation features
    # --------------------------------------------------------

    print("\nExtracting validation features...")

    validation_features, validation_labels = extract_features(
        model,
        validation_dataset
    )

    print(
        "Validation feature shape:",
        validation_features.shape
    )

    print(
        "Validation label shape:",
        validation_labels.shape
    )

    # --------------------------------------------------------
    # Save training features
    # --------------------------------------------------------

    save_features(
        train_features,
        train_labels,
        FEATURE_DIR / "train_features.pt",
        FEATURE_DIR / "train_labels.pt"
    )

    # --------------------------------------------------------
    # Save validation features
    # --------------------------------------------------------

    save_features(
        validation_features,
        validation_labels,
        FEATURE_DIR / "val_features.pt",
        FEATURE_DIR / "val_labels.pt"
    )

    # --------------------------------------------------------
    # Final verification
    # --------------------------------------------------------

    print("\nFeature cache created successfully.")

    print("\nGenerated files:")

    print(
        FEATURE_DIR / "train_features.pt"
    )

    print(
        FEATURE_DIR / "train_labels.pt"
    )

    print(
        FEATURE_DIR / "val_features.pt"
    )

    print(
        FEATURE_DIR / "val_labels.pt"
    )

    print("\nTask 3 feature extraction: PASSED")