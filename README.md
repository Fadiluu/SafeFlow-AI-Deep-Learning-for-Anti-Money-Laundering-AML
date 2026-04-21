# SafeFlow AI — Anti-Money Laundering Detection

> An end-to-end machine learning system that detects suspicious financial transactions using a multi-algorithm pipeline, anomaly scoring, and SMOTE-based class balancing — built for the UAE financial sector.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

---

## What is SafeFlow AI?

SafeFlow AI is a machine learning solution designed to identify money laundering activity in bank transaction data. Traditional rule-based AML systems flood compliance teams with false alarms and miss evolving fraud patterns. SafeFlow AI replaces rigid rules with adaptive, pattern-learning models that improve detection accuracy while significantly reducing false positives.

The project is motivated by real-world regulatory pressure on UAE banks — including the AED 11.1 million DFSA fine imposed on Mirabaud (Middle East) Limited in 2023 for AML compliance failures.

---

## Demo / Quick Look

The model is tested on live transaction samples and outputs:

```
Random Sample Index: 1322550
Actual Label: 1
Predicted Label: 1

Prediction is correct. This is a possible Money-Laundering Transaction.

---Transaction Details---
Transaction Date: 2022-11-21
Transaction Time: 11:50:25
Sender Account: 8480336062
Receiver Account: 3991121540
Transaction Amount: 156892.69
Payment Currency: UK pounds
Transaction Type: ACH
```

---

## Project Structure

```
SafeFlowAI/
├── SafeFlowAI-AML.ipynb      # Main notebook — full pipeline end to end
├── best_model_ad.joblib       # Saved AdaBoost model (generated after training)
├── requirements.txt           # All Python dependencies
├── .env.example               # Template for any environment variables
├── .gitignore                 # Excludes model weights, data, secrets
└── README.md
```

> **Note:** The dataset is downloaded automatically from Kaggle via `kagglehub` — no manual download required. See [Getting Started](#getting-started).

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, Imbalanced-learn |
| Anomaly Detection | Isolation Forest (Scikit-learn) |
| Visualization | Matplotlib, Seaborn |
| Model Persistence | Joblib |
| Dataset | Kaggle — `berkanoztas/synthetic-transaction-monitoring-dataset-aml` |

---

## Dataset

**Source:** [Synthetic Transaction Monitoring Dataset (AML) — Kaggle](https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml)

- File: `SAML-D.csv`
- Size: ~9.5 million rows
- Target column: `Is_laundering` (0 = normal, 1 = suspicious)
- Class imbalance: ~1% positive (money laundering) cases

The dataset is fetched automatically using `kagglehub`:

```python
import kagglehub
path = kagglehub.dataset_download("berkanoztas/synthetic-transaction-monitoring-dataset-aml")
```

> You will need a Kaggle account and API key configured. See [Kaggle API setup](https://www.kaggle.com/docs/api).

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/SafeFlowAI.git
cd SafeFlowAI
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Kaggle API

```bash
# Place your kaggle.json API key at:
# ~/.kaggle/kaggle.json  (macOS/Linux)
# C:\Users\<user>\.kaggle\kaggle.json  (Windows)
```

### 5. Run the notebook

```bash
jupyter notebook SafeFlowAI-AML.ipynb
```

Run all cells top to bottom. The dataset downloads automatically, the pipeline trains, and the best model is saved as `best_model_ad.joblib`.

---

## How It Works

The pipeline runs in six stages:

### 1. Data Loading & Cleaning
- Downloads `SAML-D.csv` via `kagglehub`
- Drops null rows
- Drops `Laundering_type` (label leakage risk) and `timestamp`

### 2. Feature Engineering
- Extracts `hour_`, `day_`, `month_` from transaction timestamps
- Time-based features capture suspicious patterns (late-night transfers, weekend activity)

### 3. Anomaly Scoring
- `IsolationForest` is fit on numeric features (excluding identifiers and the target)
- `contamination=0.01` — flags ~1% of transactions as potential outliers
- Adds `anomaly_score` column: lower score = more unusual transaction
- This score is passed as an additional feature to all classifiers

### 4. Preprocessing Pipeline
- `StandardScaler` applied to numerical columns
- `OneHotEncoder` (drop first, handle unknown) applied to categorical columns
- `VarianceThreshold(0.01)` removes near-constant features
- Data partitioned **before** preprocessing to prevent leakage (80/20 split, stratified)

### 5. Class Balancing (SMOTE)
- Original distribution: ~9.5M normal vs ~9.8K suspicious
- After SMOTE: ~7.6M vs ~7.6M (perfectly balanced training set)
- SMOTE applied **only on training data** — test set remains untouched

### 6. Model Training & Hyperparameter Tuning

Due to memory constraints, hyperparameter tuning uses a `StratifiedShuffleSplit` subset (~300K rows from the resampled training data).

| Model | Tuning Method | Key Params Searched |
|---|---|---|
| Logistic Regression | `GridSearchCV` | C, solver, max_iter |
| Random Forest | `RandomizedSearchCV` | n_estimators, max_depth, min_samples_split/leaf, class_weight |
| AdaBoost | `GridSearchCV` | n_estimators, learning_rate, estimator__max_depth |
| Gaussian Naive Bayes | No tuning (probabilistic, performs well out of box) |

**Best parameters found:**
- Logistic Regression: `C=10, max_iter=2000, solver='liblinear'`
- Random Forest: `n_estimators=100, min_samples_split=5, min_samples_leaf=1, max_depth=30, class_weight='balanced'`
- AdaBoost: `estimator__max_depth=3, learning_rate=1.0, n_estimators=200`

---

## Results

Accuracy is not used as the evaluation metric — with ~1% positive class, it gives misleadingly high numbers. Instead: **F1-Score, Precision, Recall, and ROC AUC.**

| Model | Precision | Recall | F1 Score | ROC AUC |
|---|---|---|---|---|
| Logistic Regression | 0.0038 | 0.6339 | 0.0075 | 0.7302 |
| Random Forest | 0.0065 | 0.2942 | 0.0128 | 0.6239 |
| **AdaBoost** ✅ | **0.0097** | **0.2456** | **0.0187** | **0.6098** |
| Gaussian Naive Bayes | 0.0038 | 0.5752 | 0.0075 | 0.7083 |

### Why AdaBoost was chosen

AdaBoost achieved the **highest F1 Score (0.0187) and highest Precision (0.0097)** across all four models. While Logistic Regression and GaussianNB had higher recall, their extremely low precision (0.0038) dragged their F1 scores down to the same level as random guessing on positives.

In a compliance context, precision matters — fewer false alarms means less wasted effort by compliance teams. AdaBoost provides the best precision-recall trade-off, making it the most operationally practical model.

The trained model is saved as `best_model_ad.joblib` using `joblib`.

---

## Model Card

| Field | Detail |
|---|---|
| Model type | AdaBoost (base: DecisionTreeClassifier, max_depth=2) |
| Task | Binary classification — money laundering detection |
| Training data | SAML-D synthetic AML dataset (Kaggle) |
| Class balancing | SMOTE (training set only) |
| Evaluation metric | F1 Score (primary), Precision, Recall, ROC AUC |
| Known limitations | Very low precision due to extreme class imbalance; SMOTE may introduce synthetic patterns not present in real data |
| Intended use | AML screening support tool for compliance teams — not a standalone decision system |
| Not intended for | Autonomous financial decisions without human review |
| Bias risk | Synthetic dataset may not reflect real-world demographic distributions |

---

## Limitations

- **Extreme class imbalance:** Only ~0.1% of real transactions are suspicious. Even after SMOTE, synthetic samples may not fully represent real laundering patterns.
- **Compute constraints:** Hyperparameter tuning used only ~300K rows (2% of resampled data) due to memory limits — optimal hyperparameters may differ at full scale.
- **Synthetic dataset:** The Kaggle dataset is simulated, not real bank data. Production performance may vary significantly.
- **Speed vs accuracy trade-off:** AdaBoost is slower than Logistic Regression or Naive Bayes — for real-time transaction screening at scale, inference latency needs further optimization.
- **Feature drift:** Money laundering tactics evolve — the model requires periodic retraining as new patterns emerge.

---

## Requirements

```
kagglehub
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
joblib
jupyter
```

Install all with:

```bash
pip install -r requirements.txt
```

---

## Roadmap

- [ ] Add XGBoost and LightGBM for comparison
- [ ] Build a FastAPI inference endpoint for real-time scoring
- [ ] Add SHAP explainability for individual transaction decisions
- [ ] Experiment with graph-based features (account network connections)
- [ ] Evaluate on real (anonymized) transaction data
- [ ] Add a Dockerfile for containerized deployment

---



## License

This project is licensed under the MIT License.

---

*Muhammed Fadil | University Project | University of Wollongong in Dubai Graduate*
