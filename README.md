Liver Biopsy Image Classification (Research Only)
Overview

This project implements a research-oriented machine learning pipeline for liver biopsy image classification using handcrafted and deep features with classical ML models.
The goal is to study feature separability, model behavior, and explainability, not clinical deployment.

⚠️ Not intended for clinical or diagnostic use.

Problem Statement

Histopathological liver biopsy images show visual patterns that correlate with conditions such as:

Healthy tissue

Inflammation

Steatosis

Ballooning

Fibrosis

Manual assessment is subjective and time-consuming. This project explores whether feature-based ML models can learn discriminative patterns from biopsy images in a controlled research setting.

Dataset Structure

The dataset is expected to be organized as:

Liver Biopsies/
├── Healthy/
├── Inflammation/
├── Steatosis/
├── Ballooning/
└── Fibrosis/


Each folder contains biopsy images belonging to one class.

Dataset is accessed via Google Drive (Colab environment).

Methodology
1. Data Loading

Images are loaded class-wise from Google Drive

Labels are inferred from folder names

Stratified splits are used to preserve class balance

2. Feature Extraction

The pipeline works on pre-extracted numerical features, including:

Deep CNN feature vectors (e.g., DenseNet embeddings)

Flattened high-dimensional representations

These features are treated as fixed descriptors (no end-to-end CNN training).

3. Model Training

A Random Forest classifier is used as the primary model:

Handles high-dimensional features well

Robust to overfitting

Interpretable via feature importance

Training uses Stratified K-Fold Cross Validation.

4. Evaluation Metrics

Training vs testing accuracy

Confusion matrix

Class-wise performance visualization

Misclassification analysis

5. Explainability & Analysis

To understand model behavior, the notebook includes:

Feature importance ranking (Random Forest)

t-SNE visualization for feature space separability

SHAP analysis for local and global explanations

Visualization of misclassified samples

Visual Outputs

The notebook generates:

Accuracy trends across folds

Confusion matrices

Feature importance bar plots

2D t-SNE embeddings

SHAP summary plots

These outputs are meant for analysis and interpretation, not reporting clinical performance.

Environment

Designed for Google Colab

Key Libraries

Python 3

NumPy, Pandas

scikit-learn

Matplotlib

SHAP

OpenCV / PIL (image handling)

Google Colab Drive API

How to Run

Open the notebook in Google Colab

Mount Google Drive when prompted

Update dataset path if needed

Run cells sequentially

No training configuration is hard-coded for production use.

Research Scope

✔ Method comparison
✔ Feature space analysis
✔ Model explainability
✔ Educational and experimental use

❌ Clinical decision support
❌ Real-world diagnosis
❌ Regulatory compliance
