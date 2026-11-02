# app.py
# Streamlit app: evaluates saved classification models on a test CSV.
# Models are loaded from the model/ folder and are NOT retrained here.

# importing required libraries
import json
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# importing required evaluation metric fuctions required for different models
from sklearn.metrics import (accuracy_score, classification_report,
                             ConfusionMatrixDisplay, f1_score,
                             matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)

# folder of this file (falls back to the working folder if __file__ is missing)
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()
MODEL_DIR = os.path.join(BASE_DIR, "model")
SAMPLE_CSV = os.path.join(BASE_DIR, "test_data.csv")

st.set_page_config(page_title="Classification Model Evaluator", layout="wide")


@st.cache_data
def load_meta():
    # feature names, target name and model file names
    with open(os.path.join(MODEL_DIR, "meta.json"), "r") as f:
        return json.load(f)


@st.cache_resource
def load_model(file_name):
    # load a saved model (no training here)
    return joblib.load(os.path.join(MODEL_DIR, file_name))


def evaluate_model(model, X_eval, y_eval):
    # same metric calculation as in the notebook
    classes = model.classes_
    y_pred = model.predict(X_eval)
    y_proba = model.predict_proba(X_eval)
    if len(classes) == 2:
        average, pos_label = "binary", classes[1]
    else:
        average, pos_label = "weighted", 1
    try:
        if len(classes) == 2:
            auc = roc_auc_score(y_eval, y_proba[:, 1])
        else:
            auc = roc_auc_score(y_eval, y_proba, multi_class="ovr",
                                average="weighted", labels=classes)
    except ValueError:
        auc = float("nan")
    metrics = {
        "Accuracy": accuracy_score(y_eval, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_eval, y_pred, average=average, pos_label=pos_label, zero_division=0),
        "Recall": recall_score(y_eval, y_pred, average=average, pos_label=pos_label, zero_division=0),
        "F1": f1_score(y_eval, y_pred, average=average, pos_label=pos_label, zero_division=0),
        "MCC": matthews_corrcoef(y_eval, y_pred),
    }
    return metrics, y_pred


meta = load_meta()
st.title("Classification Model Evaluator")
st.write("Dataset used for training: " + meta["dataset_name"])

# sidebar for model dropdown and CSV upload
st.sidebar.header("Controls")
model_name = st.sidebar.selectbox("Select a model", list(meta["model_files"].keys()))
uploaded = st.sidebar.file_uploader("Upload test data (CSV)", type=["csv"])

# use the uploaded file, otherwise the bundled test_data.csv
if uploaded is not None:
    test_df = pd.read_csv(uploaded)
    st.caption("Using the uploaded file: " + uploaded.name)
else:
    test_df = pd.read_csv(SAMPLE_CSV)
    st.caption("No file uploaded, so the bundled test_data.csv is used.")

# validate that the csv must have every feature and the target column
needed = meta["feature_names"] + [meta["target_col"]]
missing = [c for c in needed if c not in test_df.columns]
if missing:
    st.error("The CSV is missing these columns: " + ", ".join(missing))
    st.stop()

test_df = test_df.dropna(subset=needed)
X_eval = test_df[meta["feature_names"]]
y_true = test_df[meta["target_col"]]

model = load_model(meta["model_files"][model_name])

# target values must be classes the model knows
unknown = set(y_true.unique()) - set(model.classes_)
if unknown:
    st.error("The target column has values the model has not seen: " + str(unknown))
    st.stop()

metrics, y_pred = evaluate_model(model, X_eval, y_true)
