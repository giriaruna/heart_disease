# Heart Disease Classification

Predicting heart disease using the UCI dataset with Logistic Regression, KNN, and Random Forest. Models are evaluated with accuracy, precision, recall, F1, and ROC-AUC. A voting ensemble is tested to improve performance, and a Chi-Squared test identifies key predictive features.

## Project Overview

This project implements a comprehensive machine learning solution for heart disease classification using classical ML models. The application is built with Streamlit and provides an interactive interface for data exploration, model training, evaluation, and analysis.

## Features

- **Data Overview**: Explore the UCI Heart Disease dataset with visualizations
- **Data Preprocessing**: Automatic data cleaning and preprocessing
- **Model Training**: Train three classical ML models:
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Random Forest Classifier
- **Model Evaluation**: Comprehensive evaluation with:
  - Accuracy, Precision, Recall, F1-Score, ROC-AUC
  - Confusion matrices
  - ROC curves
  - Classification reports
- **Ensemble Analysis**: Voting ensemble (soft and hard voting) combining all three models
- **Feature Selection**: Chi-Squared test to identify most important features

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit application:
```bash
streamlit run streamlit_app.py
```

The application will open in your default web browser. Navigate through the different pages using the sidebar:

1. **Data Overview**: View dataset statistics and visualizations
2. **Data Preprocessing**: See preprocessing steps applied
3. **Model Training**: Train the three ML models
4. **Model Evaluation**: Compare model performance and view detailed metrics
5. **Ensemble Analysis**: Test voting ensemble performance
6. **Feature Selection**: Run Chi-Squared feature selection test

## Dataset

The application uses the UCI Heart Disease Dataset, which contains:
- 14 attributes
- Diagnostic outcomes for over 300 individuals
- Features include age, cholesterol, blood pressure, heart rate, etc.

## Team

- İlayda Dilek
- Aruna Giri

## References

1. Dua, D., & Graff, C. (2019). UCI Machine Learning Repository: Heart Disease Data Set. https://archive.ics.uci.edu/ml/datasets/Heart+Disease
2. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. JMLR, 12, 2825–2830.
3. James, G. et al. (2023). An Introduction to Statistical Learning. Springer.
