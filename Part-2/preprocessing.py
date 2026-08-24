import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
print("torch:", torch.__version__)

import torchvision
print("torchvision:", torchvision.__version__)

from torchvision import transforms

print("torchvision import successful")



# ============================================================
# RESNET-18 PREPROCESSING CONFIGURATION
# ============================================================

IMAGE_SIZE = 224

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225
]


# ============================================================
# BUILD TRANSFORM
# ============================================================

def get_preprocessing_transform():

    transform = transforms.Compose([

        # Fashion-MNIST: 28x28
        # ResNet-18: 224x224
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        # Convert:
        # 1-channel grayscale
        # →
        # 3-channel image
        transforms.Grayscale(
            num_output_channels=3
        ),

        # Convert PIL image into PyTorch tensor
        # and scale pixels from [0,255] to [0,1]
        transforms.ToTensor(),

        # Normalize using ImageNet statistics
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
    ])

    return transform


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    transform = get_preprocessing_transform()

    print("Preprocessing pipeline:")
    print(transform)