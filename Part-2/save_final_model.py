from pathlib import Path

import torch

from model import build_model


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

HEAD_PATH = ROOT_DIR / "models" / "product_classifier_head.pt"

FINAL_MODEL_PATH = ROOT_DIR / "models" / "product_classifier.pt"


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# BUILD COMPLETE MODEL
# ============================================================

def build_final_model():

    print("=" * 60)
    print("Building Final Product Classifier")
    print("=" * 60)

    print("\nDevice:", DEVICE)

    # --------------------------------------------------------
    # Build the same ResNet-18 architecture used in Task 3
    # --------------------------------------------------------

    model = build_model()

    # --------------------------------------------------------
    # Load the trained classifier head
    # --------------------------------------------------------

    head_state_dict = torch.load(
            HEAD_PATH,
            map_location=DEVICE
        )

    # --------------------------------------------------------
    # Put the trained head into ResNet-18
    # --------------------------------------------------------

    model.fc.load_state_dict(
        head_state_dict
    )

    model = model.to(DEVICE)

    # --------------------------------------------------------
    # Evaluation mode
    # --------------------------------------------------------

    model.eval()

    return model


# ============================================================
# VERIFY MODEL
# ============================================================

def verify_model(model):

    print("\nVerifying complete model...")

    # Dummy input representing:
    #
    # batch size = 1
    # channels   = 3
    # height     = 224
    # width      = 224

    dummy_input = torch.randn(
        1,
        3,
        224,
        224
    ).to(DEVICE)

    with torch.no_grad():

        output = model(
            dummy_input
        )

    print(
        "Input shape :",
        dummy_input.shape
    )

    print(
        "Output shape:",
        output.shape
    )

    # We expect:
    #
    # [1, 10]

    if output.shape != (1, 10):

        raise RuntimeError(
            f"Unexpected output shape: {output.shape}"
        )

    print(
        "Forward-pass verification: PASSED"
    )


# ============================================================
# SAVE FINAL MODEL
# ============================================================

def save_final_model(model):

    FINAL_MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        model.state_dict(),
        FINAL_MODEL_PATH
    )

    print(
        "\nFinal model saved to:"
    )

    print(
        FINAL_MODEL_PATH
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # Check that the trained head exists
    if not HEAD_PATH.exists():

        raise FileNotFoundError(
            f"Trained classifier head not found:\n{HEAD_PATH}"
        )

    model = build_final_model()

    verify_model(model)

    save_final_model(model)

    print("\n" + "=" * 60)
    print("FINAL MODEL CREATION: PASSED")
    print("=" * 60)