# Iris Species Classification: Scratch vs. Library Implementation

This project implements and compares two machine learning classification algorithms on the extended Iris dataset in `data/iris_extended.csv`:
1. **Algorithm 1:** Custom Multi-Class Logistic Regression (implemented from scratch using NumPy with One-vs-Rest strategy).
2. **Algorithm 2:** Random Forest Classifier (implemented via `scikit-learn`).

---

## Project Structure

```text
ML-Project/
├── data/
│   └── iris_extended.csv        # Extended Iris dataset
├── outputs/
│   ├── figures/                  # Output visualization plots
│   └── report_metrics.csv        # Summary evaluation metrics table
├── src/
│   ├── preprocess.py             # Data checks, soil-type one-hot encoding, label encoding, and standardization
│   ├── logistic_scratch.py       # Custom Binary & OvR Logistic Regression algorithm
│   └── evaluate.py               # Evaluation metrics helper & plotting routines
├── notebooks/
│   ├── 01_eda_and_preprocessing.ipynb
│   ├── 02_logistic_regression_scratch.ipynb
│   └── 03_model_comparison.ipynb
├── main.py                       # One-click script to execute the complete pipeline
├── requirements.txt              # Project dependencies
└── README.md                     # Documentation