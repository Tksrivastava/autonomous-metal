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

The long‑term objective is to build an AI system that can autonomously produce a weekly analyst-style report, similar to institutional commodity research desks.

Every Friday (End‑of‑Day), the system will eventually:

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

>⚠️ Mandatory step: Run this before executing any scripts, setup commands, or Docker builds.

### 2️⃣ Label Preparation Pipeline

**`pipelines/label-preparation-pipeline.py`**

Defines the supervised learning problem by:

* Creating forecast targets
* Constructing forward-looking labels
* Preparing prediction horizons

This stage formalizes how price forecasting is framed.

---

##Downloads data from Kaggle website.

> Mandetory to execute before running any other script/.sh/dockerfile# 3️⃣ Training Data Preparation

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

This stage captures market structure used by the forecasting model.

---

### 5️⃣ Forecast Model Training Pipeline

**`pipelines/forecast-model-training-pipeline.py`**

Handles:

* Model training
* Forecast generation
* Preparation for explainability analysis

Outputs serve as the quantitative backbone for future analyst reports.

---

## 📁 Repository Structure

```
autonomous-metal/
│
├── core/                         # Shared utilities and data logic
├── artifacts/                    # Generated outputs and intermediates
│
├── fetch-data-kaggle-pipeline.py
├── label-preparation-pipeline.py
├── prepare-training-data-pipeline.py
├── feature-engineering-pipeline.py
├── forecast-model-training-pipeline.py
│
├── .github/workflows/            # CI automation (linting, checks)
├── .pre-commit-config.yaml       # Pre-commit quality checks
├── pyproject.toml                # Project configuration
└── requirement.txt               # Dependencies
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

## 📊 Forecast Model Performance (Current Benchmark)

The current forecasting system has reached a stable performance baseline after extensive architectural experimentation, hyperparameter tuning, and repeated out-of-sample evaluation.

All reported metrics are computed on a **strict chronological validation period**, ensuring that model evaluation reflects realistic forward-looking market prediction rather than randomized cross-validation.

The model performs **direct multi-horizon forecasting** of LME Aluminum price movements using a fixed historical lookback window accross 14 market drivers.

---

### Evaluation Setup

* Training period: historical market data prior to validation window
* Validation period: strictly future observations (chronological split)
* Forecast horizons: **1–5 trading days ahead**
* Targets: price forecasts derived from predicted returns
* Metrics:

  * **MAPE (Mean Absolute Percentage Error)** — price accuracy
  * **Directional Accuracy** — correctness of predicted price movement sign

Directional accuracy is emphasized because, in financial markets, correctly predicting the **direction of movement** is often more economically meaningful than minimizing absolute error.

---

### Final Model Performance

#### Price Forecast Accuracy (MAPE)

| Days Ahead | Train MAPE | Validation MAPE |
| ---------- | ---------- | --------------- |
| 1          | 0.87%      | **0.96%**       |
| 2          | 1.23%      | **1.23%**       |
| 3          | 1.56%      | **1.46%**       |
| 4          | 1.95%      | **2.24%**       |
| 5          | 2.21%      | **2.22%**       |

**Observations**

* Error increases smoothly with forecast horizon, which is expected due to uncertainty accumulation.
* Training and validation errors remain closely aligned, indicating strong generalization.
* No instability or divergence is observed at longer horizons.

---

#### Directional Accuracy (Primary Metric)

| Days Ahead | Train | Validation |
| ---------- | ----- | ---------- |
| 1          | 63.3% | **57.4%**  |
| 2          | 63.5% | **58.0%**  |
| 3          | 62.6% | **60.5%**  |
| 4          | 56.0% | **58.5%**  |
| 5          | 55.8% | **58.3%**  |

---

### Interpretation Relative to Market Standards

Financial time-series forecasting differs fundamentally from typical machine learning tasks due to:

* High market efficiency
* Low signal-to-noise ratio
* Regime shifts and macroeconomic shocks
* Near-random short-term return behavior

As a result, even small improvements over random prediction are statistically meaningful.

#### Typical Directional Accuracy Benchmarks (Finance Literature)

| Accuracy Range | Interpretation                                    |
| -------------- | ------------------------------------------------- |
| ~50%           | Random walk baseline                              |
| 52–55%         | Weak predictive signal                            |
| 55–58%         | Strong ML forecasting performance                 |
| **58–61%**     | High-quality research-level models                |
| >62%           | Rare; often regime-specific or unrealistic setups |

The Autonomous Metal forecasting model achieves:

* **≈57–60% directional accuracy across horizons**
* Peak performance of **60.5% at 3-day horizon**

This places the system within the performance range commonly reported by modern machine-learning approaches applied to commodity markets.

---

### Market Interpretation

The performance profile exhibits economically realistic behavior:

* **Short horizon (1 day):** lower predictability due to market microstructure noise.
* **Medium horizon (2–3 days):** strongest predictive signal, consistent with delayed information propagation in commodity markets.
* **Longer horizons (4–5 days):** gradual signal decay as uncertainty increases.

This structure aligns with observed dynamics in real commodity trading environments rather than over-optimized backtests.

---

### Key Validation Indicators

The final model demonstrates:

✅ Stable convergence during training

✅ Minimal train–validation performance gap

✅ Consistent behavior across horizons

✅ No evidence of leakage or unrealistic fitting

✅ Performance maintained on unseen future data

Together, these characteristics suggest the model is learning persistent market structure rather than memorizing historical price paths.

---

### Practical Significance

In financial forecasting, predictive edge is incremental rather than dramatic.
Directional accuracy improvements of even **5–10 percentage points above random** can represent meaningful informational advantage when integrated into a broader analytical workflow.

Within the broader Autonomous Metal system, this forecasting component serves as the **quantitative backbone** that will later be combined with:

* explainability analysis,
* market news interpretation,
* and automated analyst-style reasoning.

---


## 🔮 Planned System Layers (In Development)

### Phase 2 — Model Explainability

* SHAP-based driver importance analysis
* Identification of dominant market factors
* Driver behavior tracking over prior weeks

### Phase 3 — Market Intelligence Layer

* Automated ingestion of aluminum industry news
* Weekly sentiment aggregation
* Theme extraction from global developments

### Phase 4 — Autonomous Analyst Report

* Weekly execution (Friday EoD)
* Narrative market outlook generation
* Risk and driver explanation
* Fully automated analyst-style report

---

## 🔁 Target Weekly Workflow (Future)

```
Weekly News + Forecasts + SHAP Drivers
                ↓
        Market Reasoning Layer
                ↓
     AI-Generated Analyst Report
```

---

## 🛠️ Tech Stack

* Python 3.11
* Pandas / NumPy
* Scikit-learn ecosystem
* SHAP (planned integration)
* Linux development environment
* GitHub Actions
* Pre-commit automation

---

## 🚧 Project Status

**Active Work in Progress**

Current focus:

* Stabilizing forecasting pipelines
* Feature engineering experimentation
* Improving reproducibility and automation

The repository reflects an evolving research system rather than a finished product.

---

## 📌 Why This Project Exists

Most ML finance projects end at prediction accuracy. Autonomous Metal explores a different question:

> Can an AI system behave like a commodity research analyst rather than just a forecasting model?

---

## ⚠️ Disclaimer

This project is for **research and educational purposes only** and does not constitute financial or trading advice.

---

## 👤 Author

**Tanul Kumar Srivastava**
Applied Data Scientist & ML Systems Engineer

---

## ⭐ Long-Term Vision

To evolve Autonomous Metal into a fully autonomous commodity intelligence system capable of combining quantitative modeling, explainability, and real‑world market reasoning into a single analyst workflow.
