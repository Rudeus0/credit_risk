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

# What's Different From credit_risk to titanic

|  Titanic |  Credit Risk |
|------------|----------------|
| 2 categorical features encoded | 13 categorical features encoded |
| Clean dataset, simple features | Mixed types, more complex encoding |
| Target already in CSV | Target needed `.map()` conversion |
| get_dummies on 2 columns | get_dummies on all categorical columns at once |

---

## Feature Engineering

```python
# Encode target from text to number
df['class'] = df['class'].map({'good': 0, 'bad': 1})

# Find all text columns automatically
categorical_cols = df.select_dtypes(include=['object', 'str']).columns.tolist()
if 'class' in categorical_cols:
    categorical_cols.remove('class')

# One-hot encode all categorical columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
# Result: 8 numeric columns → 49 columns after encoding

```

**Why encode target before get_dummies:**
If you run `get_dummies` before `map()`, pandas treats `class` as a categorical column and creates `class_good` and `class_bad` dummy columns — destroying the target. Always encode the target first.

---
## Feature Importance — Top 10

| Rank | Feature | Why It Matters |
|------|---------|----------------|
| 1 | `credit_amount` | Larger loans = higher default risk |
| 2 | `age` | Younger applicants = less financial stability |
| 3 | `duration` | Longer loan = more risk over time |
| 4 | `checking_status_no checking` | No account = financial instability |
| 5 | `installment_commitment` | High % of income to loans = stressed |
| 6 | `checking_status_<0` | Negative balance = already struggling |
| 7 | `residence_since` | Short residence = unstable living |
| 8 | `credit_history_critical` | Past problems predict future ones |
| 9 | `purpose_new car` | Car loans have specific risk profile |
| 10 | `other_payment_plans_none` | No other commitments |

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
## Confusion Matrix Explained

```
Logistic Regression:         Random Forest:
[[126  15]                   [[132   9]
 [ 24  35]]                   [ 31  28]]

LR caught 35 defaulters.     RF caught only 28 defaulters.
LR missed 24 defaulters.     RF missed 31 defaulters.
```

For credit risk, missing a defaulter (false negative) is more costly than flagging a good customer (false positive). Logistic Regression is better here.

---

## How to Run

```bash
git clone https://github.com/Rudeus0/credit_risk.git
cd credit_risk

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
python main.py
```

---