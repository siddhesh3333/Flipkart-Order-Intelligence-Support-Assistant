import torch
import torch.nn as nn
from torchvision import models


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 10

# ============================================================
# BUILD TRANSFER-LEARNING MODEL
# ============================================================

def build_model():

    # --------------------------------------------------------
    # Load pretrained ResNet-18
    # --------------------------------------------------------

    weights = models.ResNet18_Weights.DEFAULT

    backbone = models.resnet18(
        weights=weights
    )

    # --------------------------------------------------------
    # Freeze the complete pretrained backbone
    # --------------------------------------------------------

    for parameter in backbone.parameters():
        parameter.requires_grad = False

    # --------------------------------------------------------
    # Get number of input features of original classifier
    # --------------------------------------------------------

    num_features = backbone.fc.in_features

    # --------------------------------------------------------
    # Replace ImageNet classifier with Fashion-MNIST head
    # --------------------------------------------------------

    backbone.fc = nn.Linear(
        num_features,
        NUM_CLASSES
    )

    return backbone


# ============================================================
# MODEL INSPECTION
# ============================================================

if __name__ == "__main__":

    model = build_model()

    print("=" * 60)
    print("Transfer Learning Model")
    print("=" * 60)

    print("\nBackbone: ResNet-18")
    print("Pretrained: ImageNet")
    print("Number of classes:", NUM_CLASSES)

    print("\nClassifier:")
    print(model.fc)

    # --------------------------------------------------------
    # Count parameters
    # --------------------------------------------------------

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    frozen_parameters = (
        total_parameters - trainable_parameters
    )

    print("\nParameter summary:")
    print("Total parameters     :", total_parameters)
    print("Trainable parameters :", trainable_parameters)
    print("Frozen parameters    :", frozen_parameters)

    # --------------------------------------------------------
    # Verify only classifier is trainable
    # --------------------------------------------------------

    trainable_names = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]

    print("\nTrainable parameters:")
    for name in trainable_names:
        print(" -", name)

    # --------------------------------------------------------
    # Test forward pass
    # --------------------------------------------------------

    dummy_input = torch.randn(
        2,
        3,
        224,
        224
    )

    output = model(dummy_input)

    print("\nForward-pass test:")
    print("Input shape :", dummy_input.shape)
    print("Output shape:", output.shape)

    assert output.shape == (
        2,
        NUM_CLASSES
    )

    print("\nTask 3 model construction: PASSED")