**Classification Models Evaluation Using Streamlit**

**1. Problem Statement**

The objective of this project is to implement and compare five classical machine learning classification models using the same dataset.

The following classification models are implemented:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors Classifier
4. Gaussian Naive Bayes Classifier
5. Support Vector Machine

The performance of each model is evaluated using the following metrics:
1. Accuracy
2. AUC Score
3. Precision
4. Recall
5. F1 Score
6. Matthews Correlation Coefficient (MCC)

An interactive Streamlit application is also developed to evaluate the saved models using uploaded test data. The application loads the already trained models and does not retrain them on the uploaded test data.

**2. Dataset Description**

The **Breast Cancer Wisconsin (Diagnostic) dataset** is used for this classification project and the dataset is available through the scikit-learn library and was originally obtained from the UCI Machine Learning Repository.

**Dataset Details**

## Dataset Details

| Property | Description |
|---|---|
| Dataset Name | Breast Cancer Wisconsin (Diagnostic) |
| Dataset Source | UCI Machine Learning Repository through scikit-learn |
| Problem Type | Binary Classification |
| Number of Instances | 569 |
| Number of Features | 30 |
| Target Column | `target` |
| Target Classes | Malignant and Benign |
| Problem Statement | Binary Classification |
| Train-Test Split | 80% Training and 20% Testing |

The input features are calculated from digitized images of breast mass samples. They describe characteristics of the cell nuclei present in the images.

Some of the important features are:

Radius
Texture
Perimeter
Area
Smoothness
Compactness
Concavity
Concave points
Symmetry
Fractal dimension
The target variable contains two classes:

Target value	Class
0	Malignant
1	Benign
The dataset satisfies the assignment requirement of having at least 12 features and 500 instances.

## 3. Project Links

| Item | Link |
|---|---|
| GitHub Repository | [2026aj05869-Git](https://github.com/2026aj05869-design/bitsmlassign2/) |
| Streamlit Application | [2026aj05869-Streamlit](https://bitsmlassign2-jqvmjv2ts4af4atbkmrs4d.streamlit.app/) |


## 4. Models used and their Performance Metrics Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9737 | 0.9927 | 0.9726 | 0.9861 | 0.0.9793 | 0.9433 |
| Decision Tree | 0.9298 | 0.9395 | 0.9848 | 0.9028 | 0.9420 | 0.8588 |
| KNN | 0.9649 | 0.9926 | 0.9595 | 0.9726 | 0.9861 | 0.9245 |
| Naive Bayes | 0.9386 | 0.9838 | 0.9577 | 0.9444 | 0.9510 | 0.8689 |
| Support Vector Machine | 0.9649 | 0.9954 | 0.9857 | 0.9583 | 0.9718 | 0.9260 |

## 5. Model Performance Observations

| ML Model Name | Observation |
|---|---|
| Logistic Regression | ... |
| Decision Tree | ... |
| KNN | ... |
| Naive Bayes | ... |
| Support Vector Machine | ... |

## 6. Streamlit Application
| Streamlit Application | [2026aj05869-Streamlit](https://bitsmlassign2-jqvmjv2ts4af4atbkmrs4d.streamlit.app/) |

## 7. Installation and Setup

...

## 8. Running the Training Script

...

## 9. Running the Streamlit Application

...


   

