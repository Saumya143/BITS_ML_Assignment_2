import pandas as pd
import numpy as np
import os
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# 1. Setup & Data Loading
# ---------------------------------------------------------
if not os.path.exists('model'):
    os.makedirs('model')

print("Loading Breast Cancer Wisconsin Dataset...")
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scaling (Important for KNN and Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler to use in the app later
joblib.dump(scaler, 'model/scaler.pkl')
# Save X_test and y_test to a CSV for you to upload to the app for testing
test_data = X_test.copy()
test_data['target'] = y_test
test_data.to_csv('test_data.csv', index=False)
print("Test data saved as 'test_data.csv' (Use this to upload in the app)")

# 2. Model Initialization
# ---------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# 3. Training & Evaluation
# ---------------------------------------------------------
results = []

print("\nTraining Models and Calculating Metrics:\n")
print(f"{'Model':<20} | {'Acc':<8} | {'AUC':<8} | {'Prec':<8} | {'Rec':<8} | {'F1':<8} | {'MCC':<8}")
print("-" * 85)

for name, model in models.items():
    # Use scaled data for LR and KNN, raw for others (though Trees handle scaled fine too)
    if name in ["Logistic Regression", "KNN"]:
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

    # Calculate Metrics
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    mcc = matthews_corrcoef(y_test, preds)

    # Print row
    print(f"{name:<20} | {acc:.4f}   | {auc:.4f}   | {prec:.4f}   | {rec:.4f}   | {f1:.4f}   | {mcc:.4f}")

    # Save Model
    filename = f"model/{name.replace(' ', '_').lower()}.pkl"
    joblib.dump(model, filename)

print("\nAll models saved to 'model/' directory.")