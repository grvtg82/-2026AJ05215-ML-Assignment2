# Machine Learning Assignment 2

## 1. Problem Statement

The objective of this assignment is to compare the performance of five Machine Learning classification models using the same dataset.

The trained models are tested using unseen data and evaluated using six performance metrics.

A Streamlit application is used to display the results.

## 2. Dataset Description

**Dataset:** Spambase

**Source:** UCI Machine Learning Repository

**Dataset URL:** https://archive.ics.uci.edu/dataset/94/spambase

**Original number of records:** 4,601

**Number of input features:** 57

**Target column:** spam

**Classification:** Binary classification

- 0 — Not Spam
- 1 — Spam

This dataset was selected because it satisfies the assignment requirement of at least 500 records and 12 input features.

The dataset contains numerical features describing the frequencies of words, characters and capital letters in emails.

The data was checked for missing values, duplicate records and feature values. Exact duplicate records were removed. A 70% training and 30% testing split was used. Normalisation was applied where required.

## 3. GitHub Repository Link
https://github.com/grvtg82/-2026AJ05215-ML-Assignment2
## 4. Models Used and Evaluation Results

The following five models were trained and tested:

- Logistic Regression
- Decision Tree Classifier
- K-Nearest Neighbour Classifier
- Gaussian Naive Bayes Classifier
- Support Vector Machine

### Model Comparison

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8931 | 0.9529 | 0.9261 | 0.7956 | 0.8559 | 0.7774 |
| Decision Tree | 0.9010 | 0.9247 | 0.8844 | 0.8651 | 0.8746 | 0.7930 |
| KNN | 0.8876 | 0.9477 | 0.8851 | 0.8254 | 0.8542 | 0.7641 |
| Naive Bayes | 0.8321 | 0.9477 | 0.7219 | 0.9425 | 0.8176 | 0.6875 |
| Support Vector Machine | 0.9272 | 0.9662 | 0.9310 | 0.8829 | 0.9063 | 0.8476 |

## 5. Model Performance Observations

The following are points to consider when writing the final observations in the student's own words.

| Model | Observation |
|---|---|
| Logistic Regression | High precision of 0.9261, but recall is lower at 0.7956. Some spam emails were not detected. |
| Decision Tree | Accuracy of 0.9010. Precision and recall are reasonably balanced. |
| KNN | Accuracy of 0.8876. Performance is lower than Decision Tree and SVM on this test dataset. |
| Naive Bayes | Highest recall of 0.9425, but lowest precision of 0.7219. More spam emails were detected, but more legitimate emails were incorrectly classified as spam. |
| Support Vector Machine | Highest accuracy, AUC, precision, F1 and MCC among the five models. |

### Overall Winning Model

**Support Vector Machine (SVM)**

SVM achieved the highest accuracy of 92.72%, with an AUC of 0.9662 and an F1 score of 0.9063. It performed best overall on this test dataset.

Naive Bayes achieved higher recall and may be preferable when detecting as many spam emails as possible is the main priority.
