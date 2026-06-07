# Credit Risk Classifier — German Credit Dataset

Binary classification project predicting whether a bank loan applicant will default using Logistic Regression, Random Forest, and XGBoost.

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