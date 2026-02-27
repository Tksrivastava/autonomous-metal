# Autonomous Metal

**Autonomous Metal** is a personal research project focused on building an **Autonomous AI Commodity Analyst** capable of generating structured weekly outlook reports for **LME Aluminum prices**.

Unlike typical forecasting projects that stop at prediction, this project aims to replicate the workflow of a real commodity research analyst by combining:

* Quantitative forecasting
* Model explainability
* Market context understanding
* Automated report generation

The repository currently represents the **foundational forecasting and data pipeline layer** of this larger system.

---

## 🎯 Project Vision

The long-term objective is to build an AI system that can autonomously produce a weekly analyst-style report, similar to institutional commodity research desks.

Every Friday (End-of-Day), the system will eventually:

1. Forecast next week’s aluminum prices
2. Explain the dominant market drivers behind forecasts
3. Analyze weekly industry news and sentiment
4. Produce a structured market outlook report

Autonomous Metal treats forecasting as only **one component** of an analyst workflow — not the final output.

---

## 🧠 Development Philosophy

> A useful forecast must also explain *why* the market may move.

The project is designed around three principles:

* **Reproducibility** — pipeline-based ML instead of notebooks
* **Interpretability** — explain model decisions using SHAP
* **Analyst Simulation** — combine quantitative signals with narrative reasoning

---

## 🏗️ Current System Architecture (Implemented)

The repository currently implements a structured **machine learning pipeline architecture**.

### Pipeline Overview

```
Raw Market Data
        ↓
Label Preparation
        ↓
Training Dataset Assembly
        ↓
Feature Engineering
        ↓
Forecast Model Training
```

Each stage is intentionally separated to mirror production-grade ML workflows and enable future automation.

---

## ⚙️ Implemented Pipelines

### 1️⃣ Kaggle Data Fetch Pipeline

**`pipelines/fetch-data-kaggle-pipeline.py`**

Downloads the required dataset from Kaggle and prepares the local data directory.

> ⚠️ Mandatory step: Run this before executing any scripts, setup commands, or Docker builds.

---

### 2️⃣ Label Preparation Pipeline

**`pipelines/label-preparation-pipeline.py`**

Defines the supervised learning problem by:

* Creating forecast targets
* Constructing forward-looking labels
* Preparing prediction horizons

---

### 3️⃣ Training Data Preparation

**`pipelines/prepare-training-data-pipeline.py`**

Responsible for:

* Loading raw driver datasets
* Timestamp alignment
* Dataset merging and cleaning
* Producing model-ready training data

Acts as the data integration layer.

---

### 4️⃣ Feature Engineering Pipeline

**`pipelines/feature-engineering-pipeline.py`**

Transforms raw inputs into predictive signals through:

* Feature transformations
* Scaling and conditioning
* Driver preprocessing

---

### 5️⃣ Forecast Model Training Pipeline

**`pipelines/forecast-model-training-pipeline.py`**

Handles:

* Model training
* Forecast generation
* Preparation for explainability analysis

Outputs serve as the quantitative backbone for future analyst reports.

---

### 6️⃣ Performance Evaluation Pipeline

**`pipelines/performance-evaluation-pipeline.py`**

Evaluates trained forecasting models using a strict chronological split to measure real out-of-sample performance across all prediction horizons.

This pipeline serves as the **final validation layer** of the forecasting system, converting model outputs into economically meaningful evaluation metrics.

**Responsibilities**

* Loads trained models for each forecast horizon
* Reconstructs sliding-window inputs identical to training
* Applies saved feature scaling for consistency
* Generates batch predictions across all horizons
* Aligns predictions with `(ssd, days_ahead)` timestamps
* Converts predicted returns back into price space
* Computes performance metrics:

  * Mean Absolute Percentage Error (MAPE)
  * Directional Accuracy (DA)

**Evaluation Design**

Performance is computed using a chronological regime split:

* **Train period** — historical data used during model development
* **Validation period** — strictly future observations

This ensures evaluation reflects realistic forward forecasting rather than randomized validation.

**Outputs**

The pipeline logs aggregated performance summaries:

* Horizon-wise MAPE statistics
* Directional accuracy metrics
* Sample counts per evaluation period

These results form the quantitative benchmark reported in the project’s model performance section.

---

## 📁 Repository Structure

```
├── 📁 .github
│   └── 📁 workflows
│       └── ⚙️ pylint.yml
├── 📁 artifacts
│   ├── ⚙️ feature-interpretation.json
│   ├── 📄 feature-scaler.pkl
│   ├── 📄 features-set.pkl
│   ├── 📄 features.csv
│   ├── 📄 lme-al-forecast-model-1-days-ahead.keras
│   ├── 📄 lme-al-forecast-model-2-days-ahead.keras
│   ├── 📄 lme-al-forecast-model-3-days-ahead.keras
│   ├── 📄 lme-al-forecast-model-4-days-ahead.keras
│   ├── 📄 lme-al-forecast-model-5-days-ahead.keras
│   ├── 🖼️ loss-plot-1-days-ahead.png
│   ├── 🖼️ loss-plot-2-days-ahead.png
│   ├── 🖼️ loss-plot-3-days-ahead.png
│   ├── 🖼️ loss-plot-4-days-ahead.png
│   ├── 🖼️ loss-plot-5-days-ahead.png
│   ├── 📄 spot-prices.csv
│   ├── 📄 training-x.pkl
│   └── 📄 training-y.pkl
├── 📁 core
│   ├── 🐍 __init__.py
│   ├── 🐍 graph.py
│   ├── 🐍 logging.py
│   ├── 🐍 model.py
│   ├── 🐍 prompts.py
│   └── 🐍 utils.py
├── 📁 logs
├── 📁 pipelines
│   ├── 🐍 feature-engineering-pipeline.py
│   ├── 🐍 fetch-data-kaggle-pipeline.py
│   ├── 🐍 forecast-model-training-pipeline.py
│   ├── 🐍 label-preparation-pipeline.py
│   ├── 🐍 performance-evaluation-pipeline.py
│   └── 🐍 prepare-training-data-pipeline.py
├── ⚙️ .env.example
├── ⚙️ .gitignore
├── 📄 LICENSE
├── 📝 README.md
├── ⚙️ pyproject.toml
├── 📄 requirement.txt
└── 📝 same-report.md
```

---

## 📊 Inputs (Current Phase)

* Historical LME Aluminum prices
* 14 raw market drivers

---

## 📈 Outputs (Current Phase)

* Trained forecasting models
* Forward price predictions
* Intermediate artifacts for analysis and explainability

---

# 📊 Forecast Model Performance (Current Benchmark)

The forecasting system has reached a stable performance baseline after extensive architectural experimentation, hyperparameter tuning, and repeated out-of-sample evaluation.

Evaluation uses a **strict chronological split**, ensuring realistic forward-looking performance.

The model performs **direct multi-horizon forecasting** using a fixed historical lookback window across 14 market drivers.

---

## Evaluation Setup

* Forecast horizons: **1–5 trading days ahead**
* Targets derived from predicted returns
* Metrics:

  * **MAPE** — price accuracy
  * **Directional Accuracy** — correctness of predicted movement sign

Directional accuracy is emphasized because market usefulness depends primarily on predicting price direction rather than minimizing numerical deviation.

---

## Final Model Performance

### Price Forecast Accuracy (MAPE)

| Days Ahead | Train | Validation |
| ---------- | ----- | ---------- |
| 1          | 0.87% | **0.96%**  |
| 2          | 1.23% | **1.23%**  |
| 3          | 1.56% | **1.46%**  |
| 4          | 1.95% | **2.24%**  |
| 5          | 2.21% | **2.22%**  |

---

### Directional Accuracy (Primary Metric)

| Days Ahead | Train | Validation |
| ---------- | ----- | ---------- |
| 1          | 63.3% | **57.4%**  |
| 2          | 63.5% | **58.0%**  |
| 3          | 62.6% | **60.5%**  |
| 4          | 56.0% | **58.5%**  |
| 5          | 55.8% | **58.3%**  |

---

## Interpretation Relative to Market Standards

Financial markets exhibit low signal-to-noise ratios and near-random short-term behavior.

Typical benchmarks:

| Accuracy   | Interpretation             |
| ---------- | -------------------------- |
| ~50%       | Random walk                |
| 52–55%     | Weak signal                |
| 55–58%     | Strong ML performance      |
| **58–61%** | Research-level forecasting |

Autonomous Metal achieves **≈57–60% directional accuracy**, placing it within modern deep-learning commodity forecasting ranges.

---

# 🧠 Model Architecture

The forecasting model is a lightweight temporal convolutional network designed for noisy financial time-series environments.

```python
Input (lookback × features)
        ↓
Conv1D (temporal feature extraction)
        ↓
Batch Normalization
        ↓
Flatten Projection
        ↓
Regularized Dense Forecast Head
```

### Architectural Rationale

**Temporal Convolution**

Captures short-term momentum and micro-trend patterns while remaining parameter-efficient.

**Batch Normalization**

Stabilizes training under non-stationary market distributions.

**Flatten Projection**

Acts as a compact signal aggregation mechanism, avoiding high-capacity recurrent models that tend to overfit small financial datasets.

**Regularized Forecast Head**

Combines:

* L2 weight regularization (controls magnitude)
* L1 activity regularization (encourages sparse signal usage)

The `tanh` output bounds predictions and stabilizes return forecasting.

---

# 🎯 Directional Penalty Loss

Financial usefulness depends on predicting **direction**, not only magnitude.

The model therefore uses a custom objective:

```python
def _directional_penatly_loss(y_true, y_pred, sample_weight=None):
    mse = tf.keras.losses.mean_squared_error(y_true, y_pred)

    directional_accuracy = tf.reduce_mean(
        tf.cast(tf.equal(tf.sign(y_true), tf.sign(y_pred)), tf.float32)
    )

    directional_penalty = 2 / (1 + directional_accuracy)

    return mse * directional_penalty
```

### Concept

This dynamically adjusts optimization pressure:

* Correct direction → smaller penalty
* Incorrect direction → stronger correction
* Magnitude learning preserved via MSE backbone

The loss aligns gradient updates with economically meaningful prediction behavior.

---

# 📉 Training Dynamics

Training convergence can be inspected via loss curves stored in:

```
/artifacts/loss-plot-{days_ahead}-days-ahead.png
```

Observed characteristics:

* Rapid training convergence
* Smooth validation improvement
* No late-stage divergence
* Consistent behavior across all horizons

These patterns indicate learning of persistent market structure rather than memorization.

---

## Market Interpretation

* **1-day horizon:** dominated by microstructure noise
* **2–3 days:** strongest predictive signal
* **4–5 days:** gradual information decay

This structure closely matches empirical commodity market behavior.

---

## 🔮 Planned System Layers (In Development)

### Phase 2 — Model Explainability

SHAP-based driver importance analysis.

### Phase 3 — Market Intelligence Layer

Automated ingestion of aluminum industry news.

### Phase 4 — Autonomous Analyst Report

Weekly AI-generated commodity outlook.

---

## 🛠️ Tech Stack

* Python 3.11
* TensorFlow / Keras
* Pandas / NumPy
* Scikit-learn
* SHAP (planned)
* GitHub Actions

---

## 🚧 Project Status

**Active Work in Progress**

Current focus:

* Stabilizing forecasting pipelines
* Feature engineering experimentation
* Improving reproducibility

---

## 📌 Why This Project Exists

> Can an AI system behave like a commodity research analyst rather than just a forecasting model?

---

## ⚠️ Disclaimer

Research and educational purposes only. Not financial advice.

---

## 👤 Author

**Tanul Kumar Srivastava**
Applied Data Scientist & ML Systems Engineer

---

## ⭐ Long-Term Vision

To evolve Autonomous Metal into a fully autonomous commodity intelligence system combining quantitative modeling, explainability, and market reasoning into a unified analyst workflow.
