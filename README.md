# README.md

## E-Commerce A/B Test Analysis Pipeline

A full end-to-end pipeline for running and analyzing A/B tests in an e-commerce environment. Built with Python, PostgreSQL, statistical testing, cohort analytics, and interactive visualizations.

---

## 🎯 Project Purpose

This project demonstrates:

* ETL pipeline for loading raw/synthetic data into PostgreSQL
* Cohort analysis with retention and LTV calculation
* Statistical hypothesis testing (z-test, t-test)
* Interactive visualizations using Plotly
* Sample size and statistical power estimation
* Sequential testing
* Full analytical evaluation of A/B experiment performance

---

## 📁 Project Structure

```
ab_test_ecommerce/
├── data/
│   ├── raw/
│   │   └── ecommerce_transactions.csv
│   └── processed/
├── sql/
│   └── schema.sql
├── src/
│   ├── __init__.py
│   ├── ab_test.py
│   ├── analysis_extensions.py
│   ├── cohort_builder.py
│   ├── data_loader.py
│   └── visualizations.py
├── analysis_notebook.py
├── .env
├── .gitignore
├── README.md
└── requirements.txt 
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone <https://github.com/DMan113/ab_test_ecommerce.git>
cd ab_test_ecommerce
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure PostgreSQL

```bash
createdb ecommerce_ab_test
psql -d ecommerce_ab_test -f sql/schema.sql
```

### 4. Set up environment variables

```
DATABASE_URL=postgresql://user:password@localhost:5432/ecommerce_ab_test
```

### 5. Run the analysis notebook

```bash
jupytext --to ipynb analysis_notebook.py 

jupyter notebook      
```

---

## 📊 Datasets

The project uses synthetic data generated to resemble real-world e-commerce behavior.

* Users: 250,000
* Products: 50,000
* Transactions: 50,000
* Events (views → purchase): 1,400,000+
* Experiment groups: control vs treatment

**External data (Kaggle):** The pipeline architecture (DB schema and ETL) is fully compatible with importing real data for further analysis.

Alternatively, you can use real datasets from Kaggle:
- [E-Commerce Transactions Dataset](https://www.kaggle.com/datasets/smayanj/e-commerce-transactions-dataset)
- [E-commerce Clickstream Dataset](https://www.kaggle.com/datasets/waqi786/e-commerce-clickstream-and-transaction-dataset)

---

## 🔬 Methodology Overview

### Experimental Design

* A/B test with a 50/50 split
* Primary metric: Conversion Rate
* Secondary metrics: Revenue per User, LTV, Retention

### Cohort Analysis

* Monthly cohorts based on registration date
* Retention by active months
* Cumulative revenue
* LTV calculation

### Statistical Tests

* Z-test for conversion
* T-test (Welch) for revenue
* Effect sizes (Cohen’s d)

### Sequential Testing

* Monitoring p-values over time
* Lift progression

---

## 📈 Key Metrics

* Conversion rate
* Revenue per user
* Retention rate
* LTV
* Lift
* Statistical significance

---

## 🎨 Visualizations Included

* Conversion comparison
* Revenue comparison
* Retention curves
* Cohort heatmaps
* Cumulative revenue curves
* Conversion funnel
* Statistical power curves
* Sequential testing plots
* Full analytical dashboard

---

## 💡 Usage Examples

### Load data

```python
from src.data_loader import EcommerceDataLoader
loader = EcommerceDataLoader()
data = loader.generate_synthetic_data()
loader.load_to_postgres(data)
```

### Cohort Analysis

```python
from src.cohort_builder import CohortAnalyzer
cohorts = CohortAnalyzer()
cohort_data = cohorts.get_cohort_data()
```

### A/B Testing

```python
from src.ab_test import ABTestAnalyzer
ab = ABTestAnalyzer()
report = ab.full_analysis_report()
```

### Visualizations

```python
from src.visualizations import ABTestVisualizer
viz = ABTestVisualizer()
fig = viz.plot_conversion_rate(report["conversion_metrics"])
fig.show()
```

---

## 🛠️ Tech Stack

* Python 3.8+
* PostgreSQL
* Pandas, NumPy
* SciPy
* SQLAlchemy
* Plotly, Kaleido
* Jupyter Notebook

---

## 🤝 Contribution

1. Fork repository
2. Create feature branch
3. Commit changes
4. Open Pull Request

---

## 📄 License

This project is for educational and demonstration purposes.
