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
