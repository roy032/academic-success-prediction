# Model card: academic success classifier

## Model details

- **Task**: three-class classification of a student's eventual outcome: Dropout, Enrolled (still studying at the end of the normal course duration), Graduate.
- **Model**: selected automatically by 5-fold cross-validated macro-F1 among logistic regression (plain and class-balanced), Gaussian naive Bayes, an MLP and class-balanced gradient boosting. On the reference runs the class-balanced logistic regression is selected for enrollment-time features.
- **Artifact**: `model.joblib` holds a scikit-learn `Pipeline` (imputation, scaling, one-hot encoding, classifier) plus metadata: feature list, training config, library versions, timestamp.

## Intended use

- Early-warning support for academic advisors: rank students for proactive outreach (tutoring, financial counselling).
- **Not** for admissions, scholarship or any decision that removes opportunities from a student. Predictions are probabilistic and frequently wrong for individuals.

## Data

UCI *Predict Students' Dropout and Academic Success* (Realinho et al., 2022): 4,424 students from a Portuguese higher-education institution, enrolled 2008/09 to 2018/19. Features cover demographics, prior qualification, parents' education and occupation, tuition and scholarship status, and national unemployment, inflation and GDP at enrollment. Semester results are excluded by default.

## Evaluation

- Stratified 80/20 split; the test set is used once. 95% bootstrap CIs are reported for test accuracy and macro-F1.
- Real data, enrollment-time features: macro-F1 about 0.49, accuracy about 0.52 (earlier single-split run).
- With first- and second-semester results: macro-F1 about 0.68, accuracy about 0.72.
- The *Enrolled* class is the hardest (F1 around 0.25 at enrollment time). Class weighting is used so it is predicted at all.

## Limitations

- One institution in one country; macro-economic features take only a handful of distinct values, so the model cannot be expected to transfer to other systems or periods without re-training and re-validation.
- Outcomes depend on interventions: once used, the model changes the behaviour it predicts. Monitor for drift and re-train.
- Probabilities are not calibrated; use them for ranking, or calibrate on local data before reading them as risks.

## Fairness considerations

`Gender`, `Age at enrollment`, `Nacionality`, `Displaced`, `Educational special needs`, `Debtor` and parental occupation are sensitive or socio-economic attributes, and several are among the most important features. Before any deployment:

1. Report error rates (false negative rate for Dropout in particular) per group.
2. Decide with stakeholders whether to exclude sensitive attributes, and measure the cost in accuracy and the effect on group error rates.
3. Keep a human in the loop; predictions should trigger support, not penalties.

## Reproducing

```bash
pip install -e .
asp train --data data.csv --out artifacts   # writes metrics.json, results.md, figures/
```
