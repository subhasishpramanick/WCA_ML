# Stroke Prediction using Boosting (AdaBoost & XGBoost)

**Machine Learning Mini-Project - Group 10**
Topic: *Boosting - AdaBoost / XGBoost* | Dataset: Kaggle *Stroke Prediction Dataset*

**Project question:** Can a patient's risk of stroke be predicted using health and demographic factors?

Team members: *(add names / roll numbers)*

---

## 1. Files

| File | Description |
|---|---|
| `healthcare-dataset-stroke-data.csv` | Original Kaggle dataset (5,110 patients, 12 columns) |
| `stroke_prediction_boosting.ipynb` | **Colab / Jupyter notebook** - full project with code, explanations, plots and saved outputs |
| `stroke_prediction_boosting.py` | Same code as a plain Python script |
| `requirements.txt` | Python dependencies |
| `outputs/cleaned_stroke_data.csv` | Dataset after cleaning (id and the one `gender = Other` row removed, binary encoding, `bmi_missing` flag) |
| `outputs/test_metrics.csv`, `cv_comparison.csv`, `descriptive_statistics.csv`, `outlier_summary.csv`, `feature_importance_builtin.csv`, `permutation_importance.csv`, `best_hyperparameters.json` | Result tables |
| `outputs/xgboost_pipeline.joblib`, `adaboost_pipeline.joblib` | Trained models (preprocessing + classifier) |
| `figures/` | All 16 plots as PNG |

## 2. How to run

**Google Colab (recommended)**
1. Open <https://colab.research.google.com>, choose *File -> Upload notebook*, and select `stroke_prediction_boosting.ipynb`.
   (To use your own existing Colab file instead, copy the cells from this notebook into it.)
2. *Runtime -> Run all*.
3. When the data-loading cell asks, upload `healthcare-dataset-stroke-data.csv` (or place it in the Colab file panel beforehand).

**Locally**
```bash
pip install -r requirements.txt
python stroke_prediction_boosting.py        # or: jupyter notebook stroke_prediction_boosting.ipynb
```
Keep the CSV in the same folder. Runtime is about 1-2 minutes. All random seeds are fixed (`SEED = 42`), so results are reproducible.

## 3. What the project does (mapped to the assignment)

| Assignment section | Where / what |
|---|---|
| **A. Data loading** | Rows/columns, names, dtypes, head/tail, `info()`, `describe()` |
| **B. Preprocessing** | Missing values (201 missing BMI -> median + missing flag), duplicates (none), inconsistent values (one `Other` gender row dropped, `id` dropped), numeric vs categorical identification, binary + one-hot encoding, standard scaling - each choice justified in a table. Everything is fitted inside a scikit-learn `Pipeline` on training data only (no leakage) |
| **C. EDA** | Target bar/pie chart, histograms + KDE, boxplots, stroke rate per category, scatter plots, correlation heatmap, pair plot - each with written observations |
| **D. Statistics** | Range, IQR, MAD, variance, standard deviation, Pearson correlation (with p-values), outliers by IQR rule and Z-score |
| **E. Model training** | Stratified 80/20 split; **AdaBoost** and **XGBoost** explained; default baselines; `RandomizedSearchCV` (5-fold, ROC-AUC) for hyper-parameters; class imbalance handled with class weights / `scale_pos_weight` |
| **F. Evaluation** | Accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrices, ROC and precision-recall curves, classification reports, cross-validation comparison, decision-threshold selection from out-of-fold predictions |
| **G. Curves** | XGBoost train-vs-validation log loss and AUC per boosting round; AdaBoost staged AUC / log loss and weak-learner error; learning curves for both models |
| **H. Interpretation** | Built-in and permutation feature importance, risk-quintile (lift) analysis, example patient profiles, written answers to all seven interpretation questions |

## 4. Key results (test set: 1,022 patients, 50 strokes)

| Model | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| AdaBoost (tuned) | 0.50 | 0.568 | 0.088 | 0.840 | 0.160 | 0.781 | 0.148 |
| AdaBoost (tuned) | 0.86 (tuned) | 0.837 | 0.180 | 0.660 | 0.283 | 0.781 | 0.148 |
| XGBoost (tuned) | 0.50 | 0.676 | 0.115 | 0.840 | 0.202 | **0.839** | **0.240** |
| XGBoost (tuned) | 0.73 (tuned) | 0.915 | 0.239 | 0.340 | 0.281 | **0.839** | **0.240** |

5-fold cross-validated ROC-AUC on the training data: **XGBoost 0.852 +/- 0.030**, AdaBoost 0.820 +/- 0.026.

**Main findings**
* Only 4.87% of patients had a stroke, so **accuracy is misleading** (always predicting "no stroke" gives 95.1%). Default models reach that accuracy but detect only 1-2 of the 50 strokes.
* **Age is by far the most important predictor** (stroke rate 0.2% under 18 vs 17.8% at 70+); average glucose is a distant second. Hypertension, heart disease, marital status, etc. add little once age is known.
* **XGBoost ranks risk better than AdaBoost** (AUC 0.839 vs 0.781, also in cross-validation).
* Ranking is useful: the **top 20% of patients by XGBoost score contain 74% of all strokes** (stroke rate 18.1%, 3.7x average).
* Patients with a **missing BMI** had a much higher stroke rate (19.9% vs 4.3%), so a missing-value flag was added.

## 5. Limitations (short version)
Only 249 stroke cases (50 in the test set), so all metrics are uncertain; precision is moderate (~3 false alarms per detected stroke at the best F1); class weighting makes the scores a *ranking*, not calibrated probabilities; the data is a single observational snapshot without clinical variables such as blood pressure or cholesterol. The model is a screening/triage aid for educational purposes, **not a medical diagnostic tool**.

## 6. Dataset source
Kaggle - *Stroke Prediction Dataset* (fedesoriano): <https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset>
