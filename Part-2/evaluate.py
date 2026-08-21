from pathlib import Path

import torch
from torchvision import datasets

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

from model import build_model
from preprocessing import get_preprocessing_transform


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data" / "fashion_mnist"

MODEL_PATH = ROOT_DIR / "models" / "product_classifier.pt"

REPORT_DIR = ROOT_DIR / "reports"

CONFUSION_MATRIX_PATH = (
    REPORT_DIR / "confusion_matrix.csv"
)

METRICS_PATH = (
    REPORT_DIR / "test_metrics.txt"
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
# LOAD TEST DATASET
# ============================================================

def load_test_dataset():

    transform = get_preprocessing_transform()

    test_dataset = datasets.FashionMNIST(
        root=str(DATA_DIR),
        train=False,
        transform=transform,
        download=False
    )

    return test_dataset


# ============================================================
# LOAD FINAL MODEL
# ============================================================

def load_final_model():

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
# EVALUATE ALL TEST IMAGES
# ============================================================

def evaluate_model(model, test_dataset):

    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    print("\nStarting prediction on test set...")

    # --------------------------------------------------------
    # Process every test image
    # --------------------------------------------------------

    for index in range(len(test_dataset)):

        image, label = test_dataset[index]

        # ----------------------------------------------------
        # Add batch dimension
        #
        # [3, 224, 224]
        #        ↓
        # [1, 3, 224, 224]
        # ----------------------------------------------------

        image = image.unsqueeze(0)

        image = image.to(DEVICE)

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        with torch.no_grad():

            logits = model(image)

            prediction = torch.argmax(
                logits,
                dim=1
            ).item()

        # ----------------------------------------------------
        # Store labels
        # ----------------------------------------------------

        all_labels.append(label)

        all_predictions.append(prediction)

        # ----------------------------------------------------
        # Accuracy
        # ----------------------------------------------------

        if prediction == label:

            correct += 1

        total += 1

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if total % 1000 == 0:

            current_accuracy = (
                correct / total
            )

            print(
                f"Processed: {total:5d}/10000 "
                f"| Current accuracy: "
                f"{current_accuracy:.4f} "
                f"({current_accuracy * 100:.2f}%)"
            )

    # --------------------------------------------------------
    # Final accuracy
    # --------------------------------------------------------

    accuracy = correct / total

    return (
        accuracy,
        all_labels,
        all_predictions
    )


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(cm):

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        CONFUSION_MATRIX_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        # Header
        file.write(
            "Actual/Predicted,"
            + ",".join(CLASS_NAMES)
            + "\n"
        )

        # Rows
        for index, class_name in enumerate(CLASS_NAMES):

            row = ",".join(
                str(value)
                for value in cm[index]
            )

            file.write(
                f"{class_name},{row}\n"
            )

    print(
        f"\nConfusion matrix saved to:"
    )

    print(
        CONFUSION_MATRIX_PATH
    )


# ============================================================
# SAVE METRICS REPORT
# ============================================================

def save_metrics_report(
    accuracy,
    labels,
    predictions,
    cm
):

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    report = classification_report(
        labels,
        predictions,
        labels=list(range(10)),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )

    correct_predictions = int(
        accuracy * len(labels)
    )

    incorrect_predictions = (
        len(labels) - correct_predictions
    )

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "Fashion-MNIST Final Test Evaluation\n"
        )

        file.write(
            "=" * 60
            + "\n\n"
        )

        file.write(
            "Dataset\n"
        )

        file.write(
            f"Test images: {len(labels)}\n"
        )

        file.write(
            f"Correct predictions: "
            f"{correct_predictions}\n"
        )

        file.write(
            f"Incorrect predictions: "
            f"{incorrect_predictions}\n"
        )

        file.write(
            f"Test accuracy: "
            f"{accuracy:.4f} "
            f"({accuracy * 100:.2f}%)\n\n"
        )

        file.write(
            "Classification Report\n"
        )

        file.write(
            "-" * 60
            + "\n"
        )

        file.write(
            report
        )

        file.write(
            "\n\nConfusion Matrix\n"
        )

        file.write(
            "-" * 60
            + "\n\n"
        )

        file.write(
            "Rows = Actual class\n"
        )

        file.write(
            "Columns = Predicted class\n\n"
        )

        # Header
        file.write(
            f"{'Actual':<15}"
        )

        for class_name in CLASS_NAMES:

            file.write(
                f"{class_name[:10]:>12}"
            )

        file.write("\n")

        # Matrix
        for index, class_name in enumerate(CLASS_NAMES):

            file.write(
                f"{class_name:<15}"
            )

            for value in cm[index]:

                file.write(
                    f"{value:>12}"
                )

            file.write("\n")

    print(
        f"Metrics report saved to:"
    )

    print(
        METRICS_PATH
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Final Test Evaluation")
    print("=" * 60)

    print(
        "\nDevice:",
        DEVICE
    )

    # --------------------------------------------------------
    # Verify model
    # --------------------------------------------------------

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"\nFinal model not found:\n{MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Load test dataset
    # --------------------------------------------------------

    test_dataset = load_test_dataset()

    print(
        "\nTest dataset size:",
        len(test_dataset)
    )

    # --------------------------------------------------------
    # Load final model
    # --------------------------------------------------------

    model = load_final_model()

    print(
        "Final model loaded successfully."
    )

    # --------------------------------------------------------
    # Verify first test sample
    # --------------------------------------------------------

    image, label = test_dataset[0]

    print(
        "\nFirst test sample:"
    )

    print(
        "Image shape:",
        image.shape
    )

    print(
        "True label:",
        label,
        "-",
        CLASS_NAMES[label]
    )

    # --------------------------------------------------------
    # Evaluate test set
    # --------------------------------------------------------

    accuracy, labels, predictions = evaluate_model(
        model,
        test_dataset
    )

    # --------------------------------------------------------
    # Generate confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        labels,
        predictions,
        labels=list(range(10))
    )

    # --------------------------------------------------------
    # Display confusion matrix
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("10 x 10 CONFUSION MATRIX")
    print("=" * 60)

    print(
        "\nRows = Actual class"
    )

    print(
        "Columns = Predicted class\n"
    )

    print(
        f"{'Actual':<15}",
        end=""
    )

    for class_name in CLASS_NAMES:

        print(
            f"{class_name[:10]:>12}",
            end=""
        )

    print()

    for index, class_name in enumerate(CLASS_NAMES):

        print(
            f"{class_name:<15}",
            end=""
        )

        for value in cm[index]:

            print(
                f"{value:>12}",
                end=""
            )

        print()

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report = classification_report(
        labels,
        predictions,
        labels=list(range(10)),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )

    print("\n" + "=" * 60)
    print("PER-CLASS PRECISION / RECALL")
    print("=" * 60)

    print(report)

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    correct_predictions = int(
        accuracy * len(test_dataset)
    )

    incorrect_predictions = (
        len(test_dataset)
        - correct_predictions
    )

    print("\n" + "=" * 60)
    print("FINAL TEST RESULT")
    print("=" * 60)

    print(
        f"\nCorrect predictions: "
        f"{correct_predictions}"
    )

    print(
        f"Total test images: "
        f"{len(test_dataset)}"
    )

    print(
        f"Incorrect predictions: "
        f"{incorrect_predictions}"
    )

    print(
        f"\nTest Accuracy: "
        f"{accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    # --------------------------------------------------------
    # Save reports
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("SAVING EVALUATION REPORTS")
    print("=" * 60)

    save_confusion_matrix(cm)

    save_metrics_report(
        accuracy,
        labels,
        predictions,
        cm
    )

    print("\n" + "=" * 60)
    print("FINAL EVALUATION: PASSED")
    print("=" * 60)