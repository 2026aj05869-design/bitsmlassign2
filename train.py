# Script to trian 5 classical models for performance improvement
import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, "model")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Load Breast Cancer Wisconsin (Diagnostic) dataset
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    feature_cols = list(data.feature_names)
    target_col = 'target'

    # 2. Stratified train-test split (80/20, random_state=2026 as per README)
    X_train, X_test, y_train, y_test = train_test_split(
        df[feature_cols], df[target_col], test_size=0.2, random_state=2026, stratify=df[target_col]
    )

    # Save test dataset
    test_df = pd.concat([X_test, y_test], axis=1)
    test_df.to_csv(os.path.join(BASE_DIR, "test_data.csv"), index=False)

    # 3. Train 5 Classifiers
    models = {
        "Logistic Regression": Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(random_state=2026))]),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=2026),
        "KNN": Pipeline([('scaler', StandardScaler()), ('clf', KNeighborsClassifier(n_neighbors=5))]),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": Pipeline([('scaler', StandardScaler()), ('clf', SVC(probability=True, random_state=2026))])
    }

    meta_model_files = {}

    print("Training models and saving joblib files...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        fname = f"{name.lower().replace(' ', '_')}.joblib"
        joblib.dump(model, os.path.join(MODEL_DIR, fname))
        meta_model_files[name] = fname

        # Verify test metrics
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f" -> {name}: Accuracy = {acc:.4f}")

    # 4. Save meta.json
    meta_data = {
        "dataset_name": "Breast Cancer Wisconsin (Diagnostic) - UCI",
        "feature_names": feature_cols,
        "target_col": target_col,
        "model_files": meta_model_files
    }

    with open(os.path.join(MODEL_DIR, "meta.json"), "w") as f:
        json.dump(meta_data, f, indent=4)

    print("\nAll models and metadata serialized successfully into model/ directory!")

if __name__ == "__main__":
    main()
