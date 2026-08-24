from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 10
FEATURE_SIZE = 512

BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 20

RANDOM_STATE = 42

ROOT_DIR = Path(__file__).resolve().parents[1]

FEATURE_DIR = ROOT_DIR / "data" / "features"

MODEL_DIR = ROOT_DIR / "models"

BEST_MODEL_PATH = MODEL_DIR / "product_classifier_head.pt"


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD CACHED FEATURES
# ============================================================

def load_cached_features():

    train_features = torch.load(
            FEATURE_DIR / "train_features.pt"
        )

    train_labels = torch.load(
            FEATURE_DIR / "train_labels.pt"
        )

    validation_features = torch.load(
        FEATURE_DIR / "val_features.pt"
    )

    validation_labels = torch.load(
        FEATURE_DIR / "val_labels.pt"
    )

    return (
        train_features,
        train_labels,
        validation_features,
        validation_labels
    )


# ============================================================
# CREATE DATA LOADERS
# ============================================================

def create_data_loaders(
    train_features,
    train_labels,
    validation_features,
    validation_labels
):

    train_dataset = TensorDataset(
        train_features,
        train_labels
    )

    validation_dataset = TensorDataset(
        validation_features,
        validation_labels
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader, validation_loader


# ============================================================
# BUILD CLASSIFIER HEAD
# ============================================================

def build_classifier():

    classifier = nn.Linear(
        FEATURE_SIZE,
        NUM_CLASSES
    )

    return classifier.to(DEVICE)


# ============================================================
# VALIDATION
# ============================================================

def evaluate(
    model,
    data_loader,
    criterion
):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for features, labels in data_loader:

            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(features)

            loss = criterion(
                logits,
                labels
            )

            total_loss += (
                loss.item() * labels.size(0)
            )

            predictions = torch.argmax(
                logits,
                dim=1
            )

            correct += (
                (predictions == labels)
                .sum()
                .item()
            )

            total += labels.size(0)

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


# ============================================================
# TRAIN
# ============================================================

def train():

    print("=" * 60)
    print("Classifier Head Training")
    print("=" * 60)

    print("\nDevice:", DEVICE)
    print("Feature size:", FEATURE_SIZE)
    print("Number of classes:", NUM_CLASSES)
    print("Batch size:", BATCH_SIZE)
    print("Learning rate:", LEARNING_RATE)
    print("Epochs:", EPOCHS)

    # --------------------------------------------------------
    # Load features
    # --------------------------------------------------------

    (
        train_features,
        train_labels,
        validation_features,
        validation_labels
    ) = load_cached_features()

    print("\nCached feature shapes:")
    print(
        "Train features:",
        train_features.shape
    )
    print(
        "Train labels:",
        train_labels.shape
    )
    print(
        "Validation features:",
        validation_features.shape
    )
    print(
        "Validation labels:",
        validation_labels.shape
    )

    # --------------------------------------------------------
    # Create DataLoaders
    # --------------------------------------------------------

    train_loader, validation_loader = create_data_loaders(
        train_features,
        train_labels,
        validation_features,
        validation_labels
    )

    # --------------------------------------------------------
    # Create classifier
    # --------------------------------------------------------

    model = build_classifier()

    print("\nClassifier:")
    print(model)

    # --------------------------------------------------------
    # Loss function
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Adam optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # Track best validation accuracy
    # --------------------------------------------------------

    best_validation_accuracy = 0.0

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for epoch in range(1, EPOCHS + 1):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for features, labels in train_loader:

            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            # Clear previous gradients
            optimizer.zero_grad()

            # Forward pass
            logits = model(features)

            # Calculate loss
            loss = criterion(
                logits,
                labels
            )

            # Backpropagation
            loss.backward()

            # Update weights
            optimizer.step()

            # Track training statistics
            running_loss += (
                loss.item() * labels.size(0)
            )

            predictions = torch.argmax(
                logits,
                dim=1
            )

            correct += (
                (predictions == labels)
                .sum()
                .item()
            )

            total += labels.size(0)

        # ----------------------------------------------------
        # Training metrics
        # ----------------------------------------------------

        train_loss = running_loss / total
        train_accuracy = correct / total

        # ----------------------------------------------------
        # Validation metrics
        # ----------------------------------------------------

        validation_loss, validation_accuracy = evaluate(
            model,
            validation_loader,
            criterion
        )

        print(
            f"\nEpoch {epoch:02d}/{EPOCHS}"
        )

        print(
            f"Train Loss      : {train_loss:.4f}"
        )

        print(
            f"Train Accuracy  : "
            f"{train_accuracy:.4f} "
            f"({train_accuracy * 100:.2f}%)"
        )

        print(
            f"Val Loss        : {validation_loss:.4f}"
        )

        print(
            f"Val Accuracy    : "
            f"{validation_accuracy:.4f} "
            f"({validation_accuracy * 100:.2f}%)"
        )

        # ----------------------------------------------------
        # Save best classifier
        # ----------------------------------------------------

        if validation_accuracy > best_validation_accuracy:

            best_validation_accuracy = (
                validation_accuracy
            )

            torch.save(
                model.state_dict(),
                BEST_MODEL_PATH
            )

            print(
                "✓ Best classifier saved"
            )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("Feature Extraction Training Complete")
    print("=" * 60)

    print(
        f"\nBest validation accuracy: "
        f"{best_validation_accuracy:.4f} "
        f"({best_validation_accuracy * 100:.2f}%)"
    )

    print(
        "\nSaved model:"
    )

    print(BEST_MODEL_PATH)

    # --------------------------------------------------------
    # Task 4 decision
    # --------------------------------------------------------

    if best_validation_accuracy >= 0.80:

        print(
            "\nFeature extraction alone "
            "was sufficient."
        )

    else:

        print(
            "\nFeature extraction alone "
            "did NOT reach 80%."
        )

        print(
            "Fine-tuning of late ResNet "
            "layers will be required."
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train()