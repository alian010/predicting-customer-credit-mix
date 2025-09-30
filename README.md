# Predicting Customer Credit Mix

**Dataset (public CSV):**
`https://raw.githubusercontent.com/rashakil-ds/Public-Datasets/refs/heads/main/Bank%20Data.csv`

---

## Table of Contents

* [Project Overview](#project-overview)
* [Repo Structure](#repo-structure)
* [Environment & Setup](#environment--setup)
* [Data](#data)
* [Workflow](#workflow)
* [Quickstart (Notebook)](#quickstart-notebook)
* [Evaluation](#evaluation)
* [Feature Importance & Insights](#feature-importance--insights)
* [Class Imbalance](#class-imbalance)
* [Reproducibility](#reproducibility)
* [Troubleshooting](#troubleshooting)
* [Save/Load Model](#saveload-model)
* [License & Use](#license--use)

---

## Project Overview

* **Goal:** Multiclass classification of `Credit_Mix` and **per-customer** recommendations (e.g., reduce revolving balances, avoid late payments, keep old accounts open).
* **Why:** Support risk assessment and coach customers toward healthier credit behavior.

---

## Repo Structure

```
.
├─ notebooks/
│  └─ credit_mix_end_to_end.ipynb         # Step-by-step, commented notebook
├─ src/                                   # (optional) scripts for production
│  ├─ train.py
│  └─ serve_api.py                        # (optional) FastAPI endpoint
├─ models/                                # Saved models (.joblib/.pkl)
├─ data/                                  # (optional) local cache
├─ requirements.txt
├─ .gitignore
└─ README.md
```

> If submitting coursework, it’s fine to keep everything in `notebooks/`.

---

## Environment & Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
```

**Recommended versions:** `pandas >= 2.0`, `numpy >= 1.24`, `scikit-learn >= 1.3`, `matplotlib >= 3.7`, `seaborn >= 0.12`
**Optional:** `imbalanced-learn`, `optuna`, `xgboost`, `lightgbm`, `catboost`

---

## Data

* **Target:** `Credit_Mix` (categorical)
* **Typical features (if available):** income, number of accounts/loans/cards, outstanding debt, delayed/late payments, interest rate, monthly in-hand salary, credit history age, payment behavior, etc.
* The notebook handles **missing values** and **categorical encoding** automatically.

---

## Workflow

1. **Load & Quick EDA** – head, info, missing summary, target distribution
2. **Preprocessing** – imputers, robust scaling, **sparse** one-hot encoding (memory-safe)
3. **Modeling** – baseline Logistic Regression (sparse-friendly); optional tree/GBM models
4. **Evaluation** – Accuracy, weighted Precision/Recall/F1, classification report, confusion matrix
5. **(Optional) Tuning** – GridSearchCV/Optuna with StratifiedKFold
6. **Retrain & Save** – persist the best model
7. **Interpretability & Insights** – permutation importance, rule-based tips per customer

---

## Quickstart (Notebook)

Minimal training block (uses **sparse OHE** + `saga`):

```python
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

URL = "https://raw.githubusercontent.com/rashakil-ds/Public-Datasets/refs/heads/main/Bank%20Data.csv"
df = pd.read_csv(URL)

TARGET = "Credit_Mix"
df = df.dropna(subset=[TARGET]).copy()
y = df[TARGET].astype(str)
X = df.drop(columns=[TARGET])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

numeric_preprocess = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", RobustScaler())
])

try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32, min_frequency=0.005)
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=True, dtype=np.float32)

categorical_preprocess = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", ohe)
])

preprocessor = ColumnTransformer([
    ("num", numeric_preprocess, num_cols),
    ("cat", categorical_preprocess, cat_cols)
])

pipe = Pipeline([
    ("prep", preprocessor),
    ("clf", LogisticRegression(max_iter=1000, solver="saga", n_jobs=-1))
])

pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)

acc = accuracy_score(y_test, pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_test, pred, average="weighted", zero_division=0)
print(f"Test -> Acc:{acc:.4f} Prec:{prec:.4f} Rec:{rec:.4f} F1:{f1:.4f}")
print(classification_report(y_test, pred, zero_division=0))
```

---

## Evaluation

* **Report:** Accuracy + weighted Precision/Recall/F1 + per-class metrics
* **Confusion Matrix:**

```python
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(y_test, pred)
plt.title("Confusion Matrix"); plt.show()
```

* **(Optional) CV:**

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_weighted", n_jobs=-1)
print("CV F1_weighted:", scores.round(4), "Mean±Std:", round(scores.mean(),4), "±", round(scores.std(),4))
```

---

## Feature Importance & Insights

**Permutation Importance — choose one approach**

* **A) Pipeline / original feature space (fast):**

```python
from sklearn.inspection import permutation_importance
r = permutation_importance(pipe, X_test, y_test, n_repeats=10, random_state=42, scoring="f1_weighted")
imp = pd.Series(r.importances_mean, index=X_test.columns).sort_values(ascending=False).head(20)
display(imp)
```

* **B) Transformed (OHE) space / classifier only (detailed per dummy):**

```python
prep = pipe.named_steps['prep']
Xt = prep.transform(X_test)
est = pipe.named_steps['clf']
r2 = permutation_importance(est, Xt, y_test, n_repeats=10, random_state=42, scoring="f1_weighted")

num_names = list(prep.transformers_[0][2])
ohe = prep.named_transformers_['cat'].named_steps['ohe']
cat_base = prep.transformers_[1][2]
try:
    cat_names = list(ohe.get_feature_names_out(cat_base))
except AttributeError:
    cat_names = list(ohe.get_feature_names(cat_base))
feat_names = num_names + cat_names

assert Xt.shape[1] == len(feat_names)
imp2 = pd.Series(r2.importances_mean, index=feat_names).sort_values(ascending=False).head(20)
display(imp2)
```

**Actionable Recommendations (rule-based example)**

```python
def predict_one(d):
    row = pd.DataFrame([{c: np.nan for c in X.columns}])
    for k, v in d.items():
        if k in row.columns: row.loc[0, k] = v
    label = pipe.predict(row)[0]
    return label, row

num_stats = {c: {"p25": np.nanpercentile(X_train[c].astype(float), 25),
                 "p75": np.nanpercentile(X_train[c].astype(float), 75)}
             for c in num_cols if X_train[c].dtype != object}

def insights(row):
    tips = []
    def has(c): return c in row.columns
    def val(c):
        v = row[c].iloc[0] if has(c) else None
        return float(v) if v is not None and pd.notna(v) else None

    if has("Num_of_Delayed_Payment") and val("Num_of_Delayed_Payment") is not None:
        if val("Num_of_Delayed_Payment") > num_stats.get("Num_of_Delayed_Payment", {}).get("p75", float("inf")):
            tips.append("Enable autopay/alerts to eliminate late payments.")
    if has("Outstanding_Debt") and has("Monthly_Inhand_Salary"):
        v, s = val("Outstanding_Debt"), val("Monthly_Inhand_Salary")
        if v is not None and s is not None and v > 6*s:
            tips.append("Reduce outstanding debt; keep balances modest vs. income.")
    if has("Num_Credit_Card") and val("Num_Credit_Card") is not None:
        if val("Num_Credit_Card") > num_stats.get("Num_Credit_Card", {}).get("p75", float("inf")):
            tips.append("Limit new credit cards; keep a few seasoned accounts.")
    if has("Credit_History_Age") and val("Credit_History_Age") is not None:
        if val("Credit_History_Age") < num_stats.get("Credit_History_Age", {}).get("p25", 0):
            tips.append("Keep old accounts open to lengthen credit history.")
    if not tips:
        tips.append("Pay on time, keep utilization low, avoid unnecessary new credit.")
    return tips
```

---

## Class Imbalance

* Quick fix: `class_weight="balanced"`.
* Advanced: `SMOTE/SMOTENC` **inside** the pipeline (and **only** on training folds) to avoid leakage.

---

## Reproducibility

* Fixed `random_state` for splits/models
* Deterministic CV (`StratifiedKFold`)
* Environment pinned via `requirements.txt`

---

## Troubleshooting

* **MemoryError (large OHE):** Keep OHE **sparse**

  ```python
  OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32, min_frequency=0.005)
  ```

  Use `sparse=True` if `sparse_output` isn’t available in your sklearn version.

* **NameError: TruncatedSVD:**
  `from sklearn.decomposition import TruncatedSVD` (or remove the SVD step).

* **Permutation importance length mismatch:**
  Use **either** pipeline/original **or** transformed/OHE method and match feature names accordingly.

---

## Save/Load Model

```python
import joblib
joblib.dump(pipe, "models/credit_mix_model.joblib")
# later
pipe = joblib.load("models/credit_mix_model.joblib")
```

---

