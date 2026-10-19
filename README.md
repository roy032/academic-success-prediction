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

Run `python train.py --data academic_success_dataset.csv` and the table below is written to `results/results.md` (metrics in `results/metrics.json`, plots in `results/clustering.png`).

> Results from the real dataset have not been added yet; the dataset is not in this repository. Paste `results/results.md` here after running.

## Run it

```bash
pip install -r requirements.txt
python train.py --data academic_success_dataset.csv   # real data
python train.py --synthetic                           # smoke test, random data
```

The dataset (`academic_success_dataset.csv`) is not included. It is a version of the UCI "Predict Students' Dropout and Academic Success" data with some values removed. Place it next to `train.py`.

## Project structure

```
├── README.md
├── requirements.txt
├── train.py                           # leakage-free pipeline, writes results/
└── academic_success_prediction.ipynb  # original EDA + first-pass models
```
