# Part 2 — Product Image Classification

## 1. Overview

Part 2 builds the product-image classification component of the **Flipkart Order Intelligence & Support Assistant**.

The objective is to classify product images into the 10 Fashion-MNIST categories:

```text
T-shirt/top
Trouser
Pullover
Dress
Coat
Sandal
Shirt
Sneaker
Bag
Ankle boot
```

The project uses **Fashion-MNIST** and transfer learning with a pretrained **ResNet-18** backbone.

The final trained ResNet-18 state dictionary is saved as:

```text
models/product_classifier.pt
```

Part 3 loads this exact artifact for image classification.

---

## 2. Folder Contents

```text
Part-2/
├── evaluate.py
├── export_sample_images.py
├── extract_features.py
├── load_dataset.py
├── model.py
├── predict_single.py
├── preprocessing.py
├── save_final_model.py
└── train_head.py
```

Generated artifacts include:

```text
data/
├── fashion_mnist/
├── features/
└── sample_images/

models/
├── product_classifier_head.pt
└── product_classifier.pt

reports/
└── test_metrics.txt
```

---

## 3. Dataset

The project uses the standard Fashion-MNIST dataset.

It contains:

```text
60,000 training images
10,000 test images
10 classes
28 × 28 grayscale images
```

A validation subset is carved out from the training data while keeping the official test set untouched until final evaluation. This follows the assessment requirement.

---

## 4. Download / Load Fashion-MNIST

The dataset is loaded through:

```text
Part-2/load_dataset.py
```

The first training/evaluation run can download the dataset automatically if it is not already present.

After downloading, subsequent executions reuse the local dataset.

---

## 5. Image Preprocessing

The original Fashion-MNIST images are:

```text
28 × 28
1 channel
grayscale
```

ResNet-18 expects:

```text
224 × 224
3 channels
```

The preprocessing pipeline therefore:

1. Resizes the image to `224 × 224`.
2. Converts grayscale to 3 channels.
3. Converts the image to a tensor.
4. Normalizes using ImageNet mean/std.

The implementation is in:

```text
Part-2/preprocessing.py
```

The actual preprocessing is reused by Part 3, so the inference pipeline remains consistent with training.

---

## 6. Model Architecture

The model is defined in:

```text
Part-2/model.py
```

The project uses:

```text
Pretrained ResNet-18
        ↓
Frozen backbone
        ↓
New 10-class classifier head
```

The feature-extraction workflow is implemented in:

```text
Part-2/extract_features.py
```

Feature caches can be generated under:

```text
data/features/
```

The generated files include:

```text
train_features.pt
train_labels.pt
val_features.pt
val_labels.pt
```

The feature-extraction code explicitly removes the classifier layer and extracts the ResNet feature representation before training the new classifier.

---

## 7. Train the Classifier Head

From the repository root:

```powershell
python "Part-2/train_head.py"
```

This trains the new classification head.

The best classifier-head weights are saved as:

```text
models/product_classifier_head.pt
```

---

## 8. Build the Final Model

Run:

```powershell
python "Part-2/save_final_model.py"
```

This script:

1. Builds the same ResNet-18 architecture.
2. Loads the trained classifier head.
3. Attaches the head to ResNet-18.
4. Performs a forward-pass verification.
5. Saves the complete model state dict.

The final output is:

```text
models/product_classifier.pt
```

The verification expects an output tensor with:

```text
[1, 10]
```

because the classifier contains exactly 10 Fashion-MNIST classes.

---

## 9. Evaluate the Model

Run:

```powershell
python "Part-2/evaluate.py"
```

The evaluation should report:

* test accuracy
* per-class performance
* confusion matrix
* classification metrics

The project result achieved approximately:

```text
Test Accuracy: 88.61%
Macro F1:       0.8871
```

The strongest categories included classes such as:

```text
Trouser
Bag
Sneaker
Ankle boot
```

The more difficult classes are visually similar apparel categories such as:

```text
T-shirt/top
Pullover
Dress
Coat
Shirt
```

The assessment requires the confusion matrix to come from real model predictions and requires specific confusion pairs to be explained from the actual matrix.

---

## 10. Export Real Sample Images

Run:

```powershell
python "Part-2/export_sample_images.py"
```

This creates actual PNG files under:

```text
data/sample_images/
```

For example:

```text
data/sample_images/03_sneaker.png
```

These are not simulated images and are not raw IDX files.

They are real images extracted from the Fashion-MNIST test split.

At least five real test images are required for Part 3 integration.

---

## 11. Single Image Prediction

Run:

```powershell
python "Part-2/predict_single.py"
```

The script loads:

```text
models/product_classifier.pt
```

and applies the same preprocessing used during training.

A prediction conceptually looks like:

```text
Input PNG
   ↓
PIL Image
   ↓
224 × 224 resize
   ↓
3-channel conversion
   ↓
ImageNet normalization
   ↓
ResNet-18
   ↓
10 logits
   ↓
Softmax
   ↓
Predicted category + confidence
```

---

## 12. Example: User Provides an Image

Suppose a user asks:

```text
Classify this catalog image:
data/sample_images/03_sneaker.png
```

The system identifies that the request requires image classification.

Part 3 calls:

```text
classify_product_image(
    "data/sample_images/03_sneaker.png"
)
```

The tool loads:

```text
models/product_classifier.pt
```

and uses the same preprocessing pipeline from Part 2.

For example, the output could be:

```json
{
  "image_path": "data/sample_images/03_sneaker.png",
  "category": "Sneaker",
  "confidence": 0.94
}
```

The exact confidence must come from the actual model at runtime.

The Part 3 image tool explicitly loads the complete saved ResNet-18 state dict rather than returning a hardcoded category.

---

## 13. What Happens If the Image Does Not Exist?

If the user provides:

```text
data/sample_images/not_available.png
```

the tool validates the path before inference.

If the file does not exist, it raises an image-not-found error rather than inventing a classification.

Likewise, a directory path is rejected because the classifier requires an actual image file.

---

## 14. What If the User Provides Text Instead of an Image?

Example:

```text
What is the return policy for electronics?
```

This is not an image-classification request.

Part 3 routes it to the policy/RAG path instead of trying to classify text as an image.

Similarly:

```text
Check the return risk for my order.
```

is routed to the Part 1 return-risk tool.

Therefore the image classifier is only invoked when the request is actually about product-image classification.

---

## 15. Complete Part 2 Training Workflow

From the repository root:

```powershell
python "Part-2/train_head.py"

python "Part-2/save_final_model.py"

python "Part-2/evaluate.py"

python "Part-2/export_sample_images.py"

python "Part-2/predict_single.py"
```

The important dependency order is:

```text
Fashion-MNIST
     ↓
Preprocessing
     ↓
ResNet-18
     ↓
Classifier-head training
     ↓
product_classifier_head.pt
     ↓
Complete model creation
     ↓
product_classifier.pt
     ↓
Evaluation / inference
     ↓
Part 3 image tool
```

---

## 16. Final Part 2 Findings

Part 2 demonstrates that:

* Fashion-MNIST can be adapted to a product-catalog classification problem.
* Transfer learning avoids training a CNN from scratch.
* ResNet-18 provides useful visual representations.
* The final classifier achieves approximately 88.61% test accuracy.
* Visually similar apparel classes remain the main source of confusion.
* The complete trained model is persisted in `models/product_classifier.pt`.
* Real test images are exported to `data/sample_images/`.
* Part 3 reuses the exact saved model and preprocessing pipeline.

Part 2 therefore provides the **visual intelligence component** of the support assistant.
