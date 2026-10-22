# Academic Success Prediction

Predicts whether a student will **drop out, stay enrolled, or graduate** from enrollment, socio-economic and macro-economic features (4,424 students, 24 features). Compares three classifiers and a clustering analysis.

## Pipeline (`train.py`)

1. **Split first**: stratified 64 / 16 / 20 train / validation / test.
2. **Fit on train only**: median imputation, z-score outlier removal (|z| > 4, training rows only) and standardization are fit on the training split and applied unchanged to validation and test. Nothing sees the test set before the final evaluation.
3. **Models**: Logistic Regression (plain and class-balanced), Gaussian Naive Bayes, MLP (128, 64) with early stopping.
4. **Model selection** by validation macro-F1; test metrics are reported once.
5. **Clustering**: KMeans on train features with elbow and silhouette analysis, PCA visualization, and a cluster-vs-target crosstab.

Accuracy alone is misleading here: the *Enrolled* class is small (about 18%) and hard to separate, so the script reports macro-F1 and per-class F1 as well.

## Why `train.py` replaces the notebook's pipeline

The original notebook (`academic_success_prediction.ipynb`) imputed, removed outliers and standardized the **whole dataset before splitting**, so test-set statistics leaked into training. It also removed outlier rows from the test set. `train.py` fixes both. The notebook is kept for its EDA and as a record of the first pass; its modeling numbers should not be compared with `train.py` output.

Original notebook results (with leakage), test accuracy: Logistic Regression 0.636; Naive Bayes 0.258; both with near-zero recall on *Enrolled*.

## Results

Run on the public UCI *Predict Students' Dropout and Academic Success* file (4,424 students). **Caveat:** this is the clean UCI release, not the course variant used in the original notebook, which had roughly 10% of values randomly blanked out. The imputation step is therefore mostly a no-op on this data, and absolute numbers are not directly comparable to the notebook. By default the 'Curricular units' semester columns are dropped so the features match the notebook's 24 (enrollment-time information only).

**Enrollment-time features only (24 features, default):**

| Model | Val acc | Val macro-F1 | Test acc | Test macro-F1 |
|---|---|---|---|---|
| Logistic Regression | 0.621 | 0.465 | 0.614 | 0.447 |
| Logistic Regression (balanced) | 0.558 | 0.520 | 0.516 | 0.486 |
| Naive Bayes | 0.564 | 0.480 | 0.559 | 0.460 |
| Neural Network (MLP) | 0.609 | 0.444 | 0.621 | 0.461 |

**With semester performance (`--include-curricular`):**

| Model | Val acc | Val macro-F1 | Test acc | Test macro-F1 |
|---|---|---|---|---|
| Logistic Regression | 0.743 | 0.643 | 0.758 | 0.672 |
| Logistic Regression (balanced) | 0.730 | 0.679 | 0.721 | 0.680 |
| Naive Bayes | 0.671 | 0.623 | 0.634 | 0.581 |
| Neural Network (MLP) | 0.754 | 0.657 | 0.732 | 0.653 |

Takeaways:
- At enrollment time the *Enrolled* class is essentially unpredictable: the plain models reach about 0.6 accuracy by almost never predicting it (F1 of 0.02 to 0.06). Class balancing trades accuracy for a much better macro-F1 (0.486 vs 0.447), which is why models are selected on macro-F1.
- Adding first and second semester results lifts the best test macro-F1 from 0.49 to 0.68, so early-warning use needs those features.
- Results are from a single split with seed 42; differences of a few points between models are within noise.

## Run it

```bash
pip install -r requirements.txt
python train.py --data academic_success_dataset.csv   # real data
python train.py --data data.csv --include-curricular  # UCI file with semester columns
python train.py --synthetic                           # smoke test, random data
```

The dataset is not included. Download the UCI "Predict Students' Dropout and Academic Success" data (https://archive.ics.uci.edu/dataset/697) and pass its CSV with `--data`; both the semicolon-delimited UCI file and the comma-delimited course variant are accepted.

## Project structure

```
├── README.md
├── requirements.txt
├── train.py                           # leakage-free pipeline, writes results/
└── academic_success_prediction.ipynb  # original EDA + first-pass models
```
