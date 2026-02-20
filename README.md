# Autonomous Metal

## 🚧 Project Status

**This project is currently under active development.**

## Context

The price of **LME Aluminum (USD/MT)** is influenced by a constant stream of global events such as:

* Export/import policies
* Energy price fluctuations affecting smelters
* Logistics and shipping disruptions
* Geopolitical tensions
* Macro-economic indicators
* Supply–demand imbalances across regions

While traditional time-series forecasting models learn historical price behavior, they **do not incorporate real-time qualitative intelligence** present in thousands of daily news articles and market reports.

In practice, human commodity analysts manually bridge this gap by reading news, interpreting its impact, and adjusting their market expectations. This process is:

* Subjective
* Time consuming
* Difficult to scale
* Hard to reproduce consistently

There is a need for a system that can autonomously convert unstructured news into structured market intelligence and integrate it into a forecasting pipeline.

---

## Core Problem

> There is no automated system that can continuously read aluminum-related news, extract structured supply–demand signals, quantify their market impact, and combine this intelligence with ML forecasting to produce explainable weekly price projections.

---

## Project Goal

**Autonomous Metal** aims to build an Agentic AI–driven Commodity Intelligence System for **LME Aluminum** that:

1. Forecasts 1-week ahead aluminum prices using ML/DL models.
2. Curates and summarizes aluminum-relevant news from large corpora.
3. Converts unstructured articles into structured market signals.
4. Computes a weekly aggregated **News Impact Score**.
5. Adjusts the baseline forecast using this score.
6. Generates an analyst-style weekly intelligence report with explanations and what-if scenarios.

---

## Forecasting Tasks

### 1. Baseline Price Forecasting

* Train ML/DL time-series models on historical LME Aluminum prices and related variables.
* Produce a 1-week ahead price forecast.

### 2. LLM-Driven Adjustment Layer

* Extract structured signals from curated news using LLMs.
* Aggregate signals into a weekly **Impact Score**.
* Adjust the baseline forecast using this intelligence.

**Final Output:**

```
Base ML Prediction
        +
News Impact Adjustment
        =
Final Projected Price
```

---

## Intelligence Tasks

### 1. News Curation

* Remove duplicate, irrelevant, and noisy articles.
* Identify true aluminum relevance.

### 2. Structured Summarization

Transform articles into structured events such as:

* Supply shocks
* Demand changes
* Policy actions
* Energy and logistics issues
* Geopolitical risks

### 3. Weekly Impact Aggregation

* Combine all structured signals into a quantitative weekly score representing expected directional pressure on price.

---

## Final Objective — Weekly Autonomous Commodity Report

Every week, the system generates a report containing:

* Projected LME Aluminum price for the next week
* Adjustment applied due to news intelligence
* Explanation of:

  * Why the model predicted that price
  * Why the adjustment was applied
* Major market events of the week
* What-if scenario analysis (e.g., export bans, energy spikes, sanctions)

---

## Expected Outcome

A working prototype of an autonomous commodity analyst that demonstrates:

* Integration of Agentic AI and ML forecasting
* Transformation of unstructured news into quantifiable market intelligence
* Explainable and scenario-aware price projections
* Production-grade system thinking for real-world commodity markets
