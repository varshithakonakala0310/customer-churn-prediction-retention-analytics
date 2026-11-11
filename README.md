# Customer Churn Prediction & Retention Analytics

## Business problem

Acquiring a customer costs more than keeping one. This project predicts which customers are likely to churn so retention effort can be targeted early.

## Questions

1. What is the overall churn rate?
2. How accurately can churn be predicted?
3. Which factors matter most in the model?

## Dataset

**Synthetic data only** — generated in `analysis.py` with a fixed seed (42): 3,000 customers with tenure, monthly charges, support tickets, contract type and internet service. Saved to `data/churn_data_synthetic.csv`.

## Method

Random Forest classifier (300 trees), 75/25 stratified train/test split, evaluated with accuracy, precision, recall, F1 and ROC-AUC, plus feature importance.

## Key findings (actual run results)

From `results/results_summary.txt`:

- Churn rate: **0.2977 (29.77%)**
- Random Forest: **accuracy = 0.7440, ROC-AUC = 0.7694, precision = 0.5886, recall = 0.4619, F1 = 0.5176**
- Most important features: **tenure_months (0.380), monthly_charges_gbp (0.366), support_tickets (0.157)**, then contract type (0.048)

## Tech stack

Python · pandas · NumPy · scikit-learn

## How to run

```bash
pip install -r requirements.txt
python3 analysis.py
```

## Repository structure

```
├── analysis.py
├── data/
├── results/results_summary.txt
├── requirements.txt
└── LICENSE
```

## License

MIT — see `LICENSE`. Copyright (c) 2026 Varshitha Konakala.
