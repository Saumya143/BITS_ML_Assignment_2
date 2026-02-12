# Classification Model Deployment

## Problem Statement
The goal of this project is to predict whether a breast cancer tumor is **Malignant (0)** or **Benign (1)** based on digitized image features of a fine needle aspirate (FNA) of a breast mass. We implement and compare 6 different Machine Learning classifiers and deploy the best solution using Streamlit.

## Dataset Description
* **Source:** Sklearn (Breast Cancer Wisconsin Diagnostic Database)
* **Features:** 30 numeric features (radius, texture, perimeter, area, smoothness, etc.)
* **Instances:** 569 samples
* **Target:** Binary Class (Malignant vs Benign)

## Models Used & Comparison
The following 6 models were trained and evaluated:

| ML Model Name                | Accuracy | AUC    | Precision | Recall | F1 Score | MCC    |
|------------------------------|----------|--------|-----------|--------|----------|--------|
| Logistic Regression          | 0.9825   | 0.9980 | 0.9861    | 0.9861 | 0.9861   | 0.9624 |
| Decision Tree                | 0.9386   | 0.9325 | 0.9577    | 0.9444 | 0.9510   | 0.8687 |
| KNN                          | 0.9649   | 0.9901 | 0.9722    | 0.9722 | 0.9722   | 0.9248 |
| Naive Bayes                  | 0.9474   | 0.9911 | 0.9583    | 0.9583 | 0.9583   | 0.8869 |
| Random Forest (Ensemble)     | 0.9649   | 0.9950 | 0.9722    | 0.9722 | 0.9722   | 0.9248 |
| XGBoost (Ensemble)           | 0.9737   | 0.9940 | 0.9859    | 0.9722 | 0.9790   | 0.9431 |

*(Note: Replace the values above with the actual output from your train_models.py execution)*

### Observations
* **Logistic Regression** performed exceptionally well due to the linear separability of the high-dimensional feature space.
* **XGBoost** provided robust results, slightly outperforming the standalone Decision Tree.
* **Naive Bayes** showed high recall, which is crucial in medical diagnosis to avoid false negatives.

## How to Run
1.  Install dependencies: `pip install -r requirements.txt`
2.  Train models: `python train_models.py` (This generates the `model/` folder)
3.  Run App: `streamlit run app.py`