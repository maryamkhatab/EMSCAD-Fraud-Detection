# Fraudulent Online Job Posting Detection Using Machine Learning

## Project Overview

This project investigates the detection of fraudulent online job postings using machine learning techniques and the Employment Scam Aegean Dataset (EMSCAD).

## Dataset

The project uses EMSCAD, containing 17,645 job postings after duplicate removal. Each posting is classified as legitimate or fraudulent.

## Methodology

1. Dataset inspection and duplicate removal
2. Text cleaning and lemmatization
3. TF-IDF feature extraction
4. One-hot encoding of categorical features
5. Binary feature processing and missing-value indicators
6. Hybrid feature construction
7. SMOTE for handling class imbalance
8. Model training and evaluation
9. Out-of-fold stacking ensemble
10. SHAP-based model explainability

## Machine Learning Models

- Multinomial Naive Bayes
- Logistic Regression
- Random Forest
- Linear SVM
- XGBoost
- LightGBM
- Stacking Ensemble

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Average Precision

## Technologies

Python, Jupyter Notebook, Pandas, NumPy, Scikit-learn, NLTK, Imbalanced-learn, XGBoost, LightGBM, SHAP, SciPy, and Matplotlib.

## Main Notebook

`EMSCAD_Fraud_Detection.ipynb`

## Project Status

The experimental implementation, model training, evaluation, and explainability analysis have been completed.
