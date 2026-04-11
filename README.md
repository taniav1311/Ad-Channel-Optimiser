

# **Marketing Mix Revenue Optimiser**

## **Overview**

This project is an interactive **marketing budget optimization system** that estimates revenue impact based on channel-level spend allocation.

It enables users to:

* Allocate a fixed marketing budget across channels
* Observe **real-time revenue predictions**
* Understand **channel-wise contribution to revenue**

The system is designed to support **data-driven budget allocation decisions**, rather than relying on static reporting.

---

## **Technical Contribution**

The core contribution is the development of a **lightweight Marketing Mix Modeling (MMM) pipeline** integrated with an interactive decision interface.

The system combines:

* **Structured marketing spend data** (multi-channel weekly aggregation)
* **Statistical regression modeling** (Ridge regression)
* **Interactive simulation layer** (Streamlit dashboard)

This enables:

* **Instant evaluation of allocation strategies**
* **Quantitative comparison of channel effectiveness**
* **Real-time decision support under budget constraints**

---

## **System Architecture**

The workflow follows a structured data pipeline:

> *(Insert architecture diagram here)*

**Flow:**
Raw Event Data → Data Cleaning & Aggregation → Star Schema → MMM Dataset → Ridge Model → Streamlit Dashboard → Live Predictions

---

## **Key Features**

### **1. Marketing Mix Modeling Engine**

* Ridge regression on weekly aggregated data
* Multi-channel input (Paid Search, Email, Display, Social, Affiliate)
* Outputs revenue as a function of spend allocation

---

### **2. Interactive Budget Optimization Interface**

* Adjustable **total budget ($10k–$100k)**
* Channel sliders with **sum constraint enforcement**
* Instant recalculation of predicted revenue

---

### **3. Channel Contribution Analysis**

* Per-channel revenue attribution
* Identification of **high-impact vs low-impact channels**
* Visualization of positive and negative contributions

---

### **4. Data Engineering Pipeline**

* Transformation of **2M+ raw events** into structured datasets
* Star schema design:

  * `dim_channel`
  * `fact_marketing`
  * `fact_revenue`
* Aggregation to **weekly modeling dataset (101 observations)**

---

### **5. Reproducible Workflow**

* Preprocessed datasets included
* Serialized model (`model.pkl`)
* End-to-end pipeline available via notebook and app

---

## **Technology Stack**

| Layer           | Technology            | Rationale                                     |
| --------------- | --------------------- | --------------------------------------------- |
| Interface       | Streamlit             | Rapid development of interactive analytics UI |
| Modeling        | Python (scikit-learn) | Efficient regression modeling                 |
| Data Processing | Pandas                | Flexible data transformation and aggregation  |
| Storage         | CSV (Star Schema)     | Lightweight and reproducible data structure   |
| Visualization   | Streamlit Charts      | Real-time feedback for user decisions         |

---

## **Model Characteristics**

* Model: **Ridge Regression (α = 1.0)**
* Input: 5-channel normalized spend vector
* Output: Weekly revenue prediction

**Performance:**

* R² (Train): 0.033
* R² (Test): -0.004

---

## **Interpretation & Limitations**

The model is intentionally simple and highlights key challenges in MMM:

* Low R² indicates **high influence of external factors**, such as:

  * Seasonality
  * Pricing changes
  * Competitive dynamics
  * Organic traffic
  * Lag/adstock effects

* Negative test R² suggests **limited generalization due to dataset size**

**Implication:**
The system is best used for **directional insights and channel comparison**, not precise forecasting.

---

## **Novelty and Differentiation**

This project stands out in three ways:

1. **End-to-End MMM Pipeline**
   Integrates data engineering, modeling, and deployment into a single workflow.

2. **Interactive Decision System**
   Moves beyond static dashboards by enabling **real-time budget experimentation**.

3. **Constraint-Aware Optimization Interface**
   Enforces realistic conditions (fixed budget allocation), reflecting practical marketing scenarios.

---

## **Impact**

This system bridges the gap between **data analysis and decision-making**:

* Enables **real-time evaluation of marketing strategies**
* Simplifies complex MMM outputs into actionable insights
* Demonstrates how data pipelines translate into business tools

---

## **Reproducibility**

To run locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Included artifacts:

* `model.pkl` – trained regression model
* `data_processed/` – cleaned datasets
* `notebook.ipynb` – full analysis pipeline
* `app.py` – interactive dashboard

---

## **Conclusion**

The Marketing Mix Revenue Optimiser is a **practical implementation of marketing analytics principles**, combining data engineering, statistical modeling, and interactive visualization.

Its primary value lies in enabling **intuitive, data-driven budget allocation decisions under real-world constraints**.

---

If you want, I can also:

* Make this **more “data science resume optimized” (with stronger metric framing)**
* Or compress it into a **1-minute recruiter skim version** (very useful for GitHub viewers)

