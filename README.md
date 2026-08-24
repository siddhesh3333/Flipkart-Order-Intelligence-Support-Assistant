# Flipkart Order Intelligence & Support Assistant

This project is organized into three parts:

1. Part 1: Return-risk prediction for order-level return probability
2. Part 2: Product image classification using transfer learning
3. Part 3: Support agent orchestration with grounded policy answers, return-risk checks, and product-image classification

The repository combines machine-learning modeling, policy retrieval, and a final support workflow that connects all three capabilities.

## Project Structure

- `Part-1/` - return-risk model training, evaluation, and dataset validation
- `Part-2/` - image classification model, preprocessing, training, and evaluation
- `Part-3/` - RAG-based policy answerer, model integration tools, and support agent
- `data/` - datasets and sample images used by the project
- `models/` - trained model artifacts and saved thresholds
- `reports/` - evaluation metrics and performance summaries
- `transcripts/` - compact conversation and project summary records

## Prerequisites

- Python 3.10+
- pip
- A working internet connection for the first dependency install
- CUDA is optional; CPU execution is supported

Install the required packages:

```bash
python -m pip install --upgrade pip
python -m pip install numpy pandas scikit-learn joblib pillow torch torchvision sentence-transformers faiss-cpu
```

If your environment already contains the relevant ML libraries, you can skip the installation command.

## Part 1: Return-Risk Prediction

### Objective
Train and validate a return-risk model using Flipkart-style order data and save the final model for downstream use.

### Run Part 1
From the project root:

```bash
python "Part-1/random_forest_model.py"
```

This script loads the order dataset, trains the Random Forest pipeline, performs grid search, evaluates the held-out set, and saves the model and threshold metadata to:

- `models/return_risk_model.pkl`
- `models/return_risk_threshold.json`

### Useful Part 1 checks

```bash
python "Part-1/verify_dataset.py"
python "Part-1/check_logistic.py"
python "Part-1/check_feature_importance.py"
python "Part-1/check_subgroups.py"
```

### Part 1 findings

- Dataset size: 6000 rows and 13 columns
- Overall return rate: about 22.75%
- Missing values were present in `rating_given`, and the missingness pattern was related to payment-method behavior
- Baseline dummy F1 for the positive class was 0.0
- Logistic regression improved recall but was still weaker than the Random Forest approach
- Best Random Forest threshold selected from the saved metadata was `t_rf = 0.5`
- Final model parameters selected: `max_depth = 6` and `n_estimators = 200`
- The Random Forest was selected as the final artifact for Part 3 integration because it had the strongest practical risk-detection signal

## Part 2: Product Image Classification

### Objective
Build a transfer-learning image classifier using a pretrained ResNet-18 backbone and a Fashion-MNIST-style product taxonomy.

### Run Part 2
From the project root:

```bash
python "Part-2/train_head.py"
python "Part-2/save_final_model.py"
python "Part-2/evaluate.py"
```

If you want to run single-image inference:

```bash
python "Part-2/predict_single.py"
```

### Notes on Part 2 workflow

- Model architecture is defined in `Part-2/model.py`
- Preprocessing is defined in `Part-2/preprocessing.py`
- Feature extraction is handled in `Part-2/extract_features.py`
- The final saved classifier is stored as:
  - `models/product_classifier.pt`
  - `models/product_classifier_head.pt`

### Part 2 findings

- The final model achieved a test accuracy of 88.61%
- Macro F1 score was 0.8871
- Best-performing classes included `Trouser`, `Bag`, `Sneaker`, and `Ankle boot`
- Some confusion remained in visually similar categories such as `T-shirt/top`, `Pullover`, `Dress`, `Coat`, and `Shirt`
- This classifier is the model reused by the Part 3 product-image tool without retraining

## Part 3: Support Agent and Policy Grounding

### Objective
Create an orchestration layer that combines:

- policy-grounded question answering
- return-risk evaluation using the real Part 1 model
- product-image classification using the real Part 2 model

### Run Part 3
From the project root:

```bash
cd "Part-3"
python evaluation/evaluate_agent.py
python evaluation/evaluate_return_risk.py
python evaluation/evaluate_answerer.py
python evaluation/evaluate_retrieval.py
```

To run the support agent directly:

```bash
cd "Part-3"
python -c "from agent.support_agent import SupportAgent, POLICY, RETURN_RISK, PRODUCT_IMAGE; agent = SupportAgent(); print(agent.handle(POLICY, query='Can I return a product after the return window expires?')); print(agent.handle(RETURN_RISK, order_features={'price_inr':2499.0,'discount_pct':20.0,'customer_tenure_days':180,'num_previous_orders':8,'num_previous_returns':1,'delivery_distance_km':12.5,'delivery_days':4,'is_weekend_order':0,'rating_given':4.0,'product_category':'Electronics','payment_method':'UPI'}));"
```

### Part 3 workflow details

- `Part-3/rag/chunker.py` creates text chunks from the policy corpus
- `Part-3/rag/embedder.py` generates sentence-transformer embeddings
- `Part-3/rag/vector_store.py` builds the FAISS index
- `Part-3/rag/retriever.py` retrieves relevant policy chunks
- `Part-3/rag/answerer.py` decides whether the answer is supported and grounded
- `Part-3/tools/return_risk_tool.py` loads the Part 1 Random Forest model and threshold
- `Part-3/tools/product_classifier_tool.py` loads the Part 2 saved classifier
- `Part-3/agent/support_agent.py` routes the request to the appropriate component

### Part 3 findings

- The policy route is able to answer supported questions with grounded evidence
- Unsupported policy questions are rejected when no sufficiently relevant policy chunks are found
- The return-risk route loads the real saved Random Forest model and threshold file rather than hardcoding a value
- The product-image route loads the actual saved Part 2 model without retraining or changing inference logic
- The integration layer is designed to preserve the original Part 1 and Part 2 artifacts while exposing a clean support interface

## End-to-End Execution Summary

A typical end-to-end flow is:

```bash
python "Part-1/random_forest_model.py"
python "Part-2/save_final_model.py"
cd "Part-3"
python evaluation/evaluate_agent.py
```

This ensures the saved artifacts exist and the support agent can validate the integration path.

## Important Output Locations

- Return-risk model: `models/return_risk_model.pkl`
- Return-risk threshold: `models/return_risk_threshold.json`
- Product classifier: `models/product_classifier.pt`
- Part 2 evaluation report: `reports/test_metrics.txt`
- Part 1/Part 2 summary records: `transcripts/`

## Final Notes

- Part 1 and Part 2 are the core underlying ML artifacts
- Part 3 is the integration layer that consumes them with the policy RAG system
- The repository is structured so the original modeling work remains intact while the final support agent orchestrates the results in a controlled way
