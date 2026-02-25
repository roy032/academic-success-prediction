# Academic Success Prediction

A machine learning project that predicts student academic outcomes using supervised and unsupervised learning techniques. Includes full EDA, feature engineering, model training, and evaluation with multiple classifiers.

## Overview

This project analyzes an academic success dataset to classify students into outcome categories. It applies a complete ML pipeline — from data exploration and preprocessing to model comparison and clustering.

## Pipeline

1. **Exploratory Data Analysis** — Feature distributions, variance, skewness, missing values, class balance
2. **Data Preprocessing** — Imputation, encoding, feature selection (mutual information), standardization
3. **Train/Val/Test Split** — 64/16/20 stratified split
4. **Model Training** — Logistic Regression, Naive Bayes, Neural Network (MLP)
5. **Unsupervised Learning** — KMeans clustering with elbow method, PCA visualization
6. **Evaluation** — Accuracy comparison, classification reports, confusion matrices, ROC curves

## Models

| Model | Approach |
|-------|----------|
| Logistic Regression | Multinomial with L-BFGS solver |
| Naive Bayes | Gaussian |
| Neural Network | MLPClassifier (128, 64) with early stopping |
| KMeans | Unsupervised clustering (k=3) with PCA |

## Tech Stack

- Python, Jupyter Notebook
- pandas, NumPy, Matplotlib, Seaborn
- scikit-learn (classification, clustering, preprocessing, evaluation)

## Project Structure

```
├── README.md
└── academic_success_prediction.ipynb    # Full analysis notebook
```

> **Note:** The dataset (`academic_success_dataset.csv`) is loaded in the notebook. Place it in the same directory before running.

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook academic_success_prediction.ipynb
```
