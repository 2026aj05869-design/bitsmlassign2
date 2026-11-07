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
| Logistic Regression | Logistic Regression performed very well with the highest Accuracy of 0.9737 and Recall of 0.9861. It also achieved an AUC of 0.9927 and MCC of 0.9433, showing strong and balanced performance on the test dataset. |
| Decision Tree | Decision Tree achieved an Accuracy of 0.9298, which is the lowest among the five models. It provided the second-highest Precision of 0.9848, but its lower Recall of 0.9028 and MCC of 0.8588 indicate that its overall performance was lower compared to the other models. |
| KNN | KNN performed well with an Accuracy of 0.9649 and AUC of 0.9926. It achieved the highest F1 Score of 0.9861, showing a good balance between Precision and Recall. Feature scaling was applied because KNN uses distance between data points for classification. |
| Naive Bayes | Naive Bayes achieved an Accuracy of 0.9386 and AUC of 0.9838. Its performance was better than Decision Tree in Accuracy, Recall, F1 Score and MCC, but lower than Logistic Regression, KNN and Support Vector Machine in overall performance. |
| Support Vector Machine | Support Vector Machine performed very well with an Accuracy of 0.9649. It achieved the highest AUC of 0.9954 and highest Precision of 0.9857 among all five models, showing very good class separation and accurate positive predictions. |


### Overall Observation

Logistic Regression provided the best overall performance with the highest Accuracy of 0.9737, highest Recall of 0.9861 and highest MCC of 0.9433.

Support Vector Machine also performed very well and achieved the highest AUC of 0.9954 and highest Precision of 0.9857.

KNN achieved the highest F1 Score of 0.9861 and also showed good overall performance.

**Based on the overall comparison of all six evaluation metrics, Logistic Regression was selected as the best-performing model for this test dataset.**

## 6. Streamlit Application
| Streamlit Application | [2026aj05869-Streamlit](https://bitsmlassign2-jqvmjv2ts4af4atbkmrs4d.streamlit.app/) |

## 7. Installation and Setup

...

## 8. Running the Training Script

...

## 9. Running the Streamlit Application

...


   

