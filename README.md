# 👔 Employee Flight Risk Analysis Dashboard

An end-to-end, production-ready Machine Learning system built with **Streamlit** to predict employee attrition. The application leverages an advanced **imbalance-aware Stacking Classifier** combining **CatBoost** and **Random Forest** algorithms to proactively flag flight risks within your organization.

---

## 🎯 Strategic Business Goals
* **Maximize Recall Strategy:** Configured with a risk threshold of **0.35** (instead of standard 0.50) to aggressively catch flight risks, minimizing expensive unpredicted corporate resignations.
* **Leakage-Free SMOTEENN Resampling:** Leverages `imblearn.pipeline` inside a 10-fold cross-validation scheme to handle class imbalances without historical data contamination.

---

## 🏗️ Stacking Machine Learning Pipeline Architecture

The application handles text fields, scales continuous parameters, balances the target metrics, and stacks models sequentially:

1. **Preprocessing Layer (`ColumnTransformer`):**
   * **Power Transformer (`Yeo-Johnson`):** Normalises numeric spreads dynamically.
   * **One-Hot Encoding:** Applied to `BusinessTravel` and `Department`.
   * **Ordinal Encoding:** Applied to `Gender` and `OverTime`.
   * **CatBoost Encoding:** Target-aware text vectorisation for high-cardinality values (`EducationField`, `JobRole`, `MaritalStatus`).
2. **Resampling Engine (`SMOTEENN`):** Combines synthetic minority over-sampling with Edited Nearest Neighbors data point cleaning.
3. **Dimensionality Optimization (`SelectKBest`):** Isolates the top 23 most descriptive information columns via mutual classification scoring.
4. **Stacked Classifiers:**
   * **Base Layer:** `CatBoostClassifier` (1,000 iterations, balanced class optimization weight mapping).
   * **Final Meta-Learner Layer:** `RandomForestClassifier` (1,000 estimators, max depth 10, balanced bootstrap settings).

---

## 🚀 Step-by-Step Local Deployment Guide

Follow these steps to get your environment up and running smoothly:

### 1. Clone or Open Your Project Folder
Ensure your folder structure matches this setup perfectly:
```text
HR-Employee-Attrition/
├── data/
│   └── HR-Employee-Attrition.csv
├── streamlit_app.py
├── requirements.txt
└── README.md
```

### 2. Create and Activate a Virtual Environment
Run the following commands in your terminal console to keep dependencies isolated:

* **On Windows (Command Prompt / PowerShell):**
  ```bash
  python -m venv .venv
  .venv\Scripts\activate
  ```
* **On macOS / Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 3. Install Required Dependencies
Upgrade pip and run the requirements bundle installation script:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Boot Up the Dashboard
Execute the local network pipeline server using this Streamlit call:
```bash
streamlit run streamlit_app.py
```
*Your system browser should open automatically displaying the live portal dashboard layout at `http://localhost:8501`.*

---

## 💡 Key App Operations & Workflows

### Tab 1: Individual Employee Analysis
* Set individual employee attributes using dynamic slider configurations and dropdown picklists.
* Background fallbacks cleanly inject historical average/mode fields for complex non-visible features (e.g., `DailyRate`, `EnvironmentSatisfaction`) to prevent matrix calculation errors.
* Click **Evaluate Attrition Risk** to view the calculated risk distribution probability alongside tailored HR recommendations.

### Tab 2: Batch CSV Processing
* **Get the Upload Layout:** Click the **Download Sample Template CSV** button. This automatically generates a clean file formatted with the precise input order schemas required by the underlying model pipeline.
* **Process Mass Logs:** Drop your employee data sheets straight onto the upload zone. The framework parses, reorders, evaluates, and wraps predictions into an overview layout instantly.
* **Export Actionable Data:** Click **Download Complete Risk Analysis Reports** to get your enriched data sheets back containing calculated probabilities and action indicators.
