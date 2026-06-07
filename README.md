# Credit Risk Classifier — German Credit Dataset

Binary classification project predicting whether a bank loan applicant will default using Logistic Regression, Random Forest, and XGBoost.

---
## Results

| Model | Accuracy | ROC AUC | F1 (Bad/Default) | Recall (Bad) |
|-------|----------|---------|------------------|--------------|
| Logistic Regression | 0.8050 | 0.8181 | 0.64 | 0.59 |
| Random Forest | 0.8000 | 0.8086 | 0.58 | 0.47 |
| XGBoost | 0.7750 | 0.8205 | 0.57 | 0.51 |

**Winner: Logistic Regression** — highest accuracy and best recall for bad loans (defaulters).
**Best ROC AUC: XGBoost (0.8205)** — best at separating good vs bad applicants overall.

---


## Project Structure

```
credit_risk/
├── data/
│   └── credit_customers.csv     ← German Credit dataset
├── notebooks/
│   └── analysis.ipynb           ← EDA, feature engineering, model exploration
├── plots/
│   └── feature_importance.png
├── src/
│   ├── train.py                 ← load, features, transform, train
│   └── evaluate.py              ← evaluate all 3 models
├── main.py                      ← pipeline orchestrator
├── requirements.txt
└── README.md
```

---

## Dataset

- **Source:** German Credit Risk dataset (ppb00x/credit-risk-customers on Kaggle)
- **Size:** 1,000 rows × 21 columns → 200 test samples
- **Target:** `class` — good=0 (no default), bad=1 (defaulted)
- **Class distribution:** 700 good (70%), 300 bad (30%) — imbalanced

---

## Pipeline Flow

```
data_load()         → read credit_customers.csv
features()          → encode target + get_dummies on all categorical columns
transform_scaler()  → split 80/20 + StandardScaler
credit_model()      → fit LogisticRegression, RandomForest, XGBoost
evaluate_models()   → accuracy, confusion matrix, classification report, ROC AUC
```

---
