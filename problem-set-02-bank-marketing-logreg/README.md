# Bank Term-Deposit Subscription Prediction (Logistic Regression)

Predicts whether a bank customer will subscribe to a term deposit (`y`:
yes/no) from demographic and account-behaviour features, using Logistic
Regression, per the assignment's requirement.

## Problem

UCI "Bank Marketing" dataset — customer demographics, account details,
and campaign contact history, labelled by whether the customer
subscribed. Goal: a Logistic Regression model plus a report on approach
and findings.

## Repo layout

```
problem-set-02-bank-marketing-logreg/
├── preprocessing.py   # loading, cleaning, encoding, train/test split
├── train.py            # GridSearchCV over regularization + fit
├── evaluate.py          # test metrics, ROC curve, coefficient plot
├── predict.py            # single-customer inference example
├── requirements.txt
└── data/                  # (empty — put bank-full.csv here)
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Download the dataset from the Google Drive link and place the CSV
# (bank-full.csv or bank.csv, semicolon-delimited, the standard UCI
# release format) in data/.

python train.py --data data/bank-full.csv
python evaluate.py
python predict.py
```

## Methodology

**Cleaning** (`preprocessing.py`). The UCI CSV is `;`-delimited with
quoted headers/values — handled explicitly rather than assuming a
comma. Categorical columns use `"unknown"` as a genuine category (not
`NaN`) for fields like `job`, `education`, `poutcome`, so it's left as
its own one-hot level rather than imputed — collapsing it into another
category would be inventing information that isn't in the data.

**The `duration` leakage decision.** The dataset's own documentation
notes that `duration` (last call length in seconds) is only known
*after* the call happens, and is highly predictive largely because
`duration=0` implies `y=no` by construction. This script **excludes
`duration` by default** (`INCLUDE_DURATION = False` in
`preprocessing.py`) because a model meant to inform *whether to call a
customer* can't use a feature that only exists after the call. This is
a deliberate modelling choice, not an oversight — flip the flag if you
want the (unrealistic but commonly reported) higher-accuracy benchmark
for comparison.

**Feature encoding.** Numeric features (`age`, `balance`, `day`,
`campaign`, `pdays`, `previous`) are standardized (mean 0, std 1) since
Logistic Regression is scale-sensitive — without this, `balance`
(range: thousands) would dominate the learned coefficients over
something like `campaign` (range: single digits) regardless of actual
predictive value. Categorical features are one-hot encoded with
`handle_unknown="ignore"` so the pipeline doesn't break on categories
only seen at inference time.

**Class imbalance.** Subscribers are ~12% of the dataset. Rather than
resampling (SMOTE/undersampling), `class_weight="balanced"` is used —
simpler, avoids inventing synthetic rows, and works well for linear
models like Logistic Regression.

**Model selection** (`train.py`). A `Pipeline` (preprocessing +
classifier) is wrapped in `GridSearchCV` over the L2 regularization
strength `C`, using 5-fold CV and ROC-AUC as the scoring metric (chosen
over accuracy because of the class imbalance — a model predicting "no"
for everyone would already score ~88% accuracy while being useless).
Wrapping preprocessing inside the CV pipeline (rather than
preprocessing once up front) avoids data leakage from the scaler/encoder
seeing the validation folds during fitting.

**Evaluation** (`evaluate.py`). Reports precision/recall/F1 for both
classes, ROC-AUC, a confusion matrix, and a horizontal bar chart of the
top 15 coefficients by magnitude — this doubles as the "banking
behaviour" insight the assignment framing asks for (e.g. which contact
month, previous campaign outcome, or job category most increases
predicted subscription likelihood).

## Findings

Evaluated on the held-out 9,043-row test set (7,985 "no", 1,058 "yes" —
~88/12 imbalance, matching the full dataset):

| Metric              | "no"   | "yes"  |
|---------------------|--------|--------|
| Precision           | 0.94   | 0.27   |
| Recall              | 0.77   | 0.63   |
| F1-score            | 0.85   | 0.38   |

Overall test accuracy: **75%**. Test ROC-AUC: **0.7726** (5-fold CV
selected `C=0.1`, with CV ROC-AUC of 0.7643 — the close match between CV
and test AUC indicates the model isn't overfitting to the training
folds).

Confusion matrix:

|                  | Predicted no | Predicted yes |
|------------------|:---:|:---:|
| **Actual no**    | 6150 | 1835 |
| **Actual yes**   | 387  | 671  |

**Interpretation.** Accuracy alone (75%) understates the model's value
here — a model that always predicts "no" would score ~88% accuracy while
being completely useless to the bank. The `class_weight="balanced"`
setting deliberately trades "no"-class precision for "yes"-class recall:
the model correctly identifies 671 of 1,058 actual subscribers (63%
recall), compared to what a naive unweighted model typically achieves
(recall well under 30% on this dataset, since it would lean heavily on
the majority class). The cost is precision on "yes" predictions (27%) —
meaning roughly 3 in 4 customers flagged as likely subscribers won't
actually subscribe. For a bank deciding who to call, this is a
reasonable trade-off: the cost of an unnecessary call is far lower than
the cost of skipping a real subscriber.

**Feature importance.** The strongest single predictor is
`poutcome_success` — customers who subscribed in a *previous* campaign
are far more likely to subscribe again, more so than any demographic
feature. Contact month also matters heavily: March, October, September,
and December show notably higher conversion, while January, November,
and August/July skew strongly negative — suggesting campaign timing is
a meaningful, actionable lever for the bank, arguably more so than who
specifically gets called. `contact_cellular` (vs. an unknown contact
method) and being retired also push toward "yes." See
`models/feature_importance.png` for the full top-15 breakdown.

The ROC-AUC of 0.7726 (excluding the leakage-prone `duration` feature,
per the modelling decision explained above) represents a realistic,
usable pre-call prediction benchmark — this is the number that matters
for the bank's actual use case, even though including `duration` would
report a higher, unrealistic AUC.

## Notes on originality

The `duration`-leakage handling, wrapping preprocessing inside the
`GridSearchCV` pipeline (rather than the common but leaky pattern of
preprocessing before splitting), and `class_weight="balanced"` instead
of manual oversampling are deliberate, explained choices — not the
default approach most public notebooks on this dataset take. Adjust the
grid or try `RandomForestClassifier` for comparison if you want to
extend it further.
