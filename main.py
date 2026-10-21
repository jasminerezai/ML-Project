import sys
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Add src folder to module search path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.preprocess import load_and_preprocess_data
from src.logistic_scratch import LogisticRegressionScratchOvR
from src.evaluate import evaluate_model, plot_confusion_matrices

def main():
    print("  IRIS CLASSIFICATION PIPELINE: SCRATCH VS LIBRARY  ")

    # 1. Load and preprocess dataset
    data_path = os.path.join("data", "Iris.csv")
    print(f"\n[1/4] Loading and preprocessing data from '{data_path}'...")
    df, X_train, X_test, y_train, y_test, encoder = load_and_preprocess_data(data_path)
    print(f"      Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    # 2. Train and evaluate Scratch Logistic Regression
    print("\n[2/4] Training Model 1: Logistic Regression (Scratch OvR)...")
    scratch_model = LogisticRegressionScratchOvR(lr=0.1, num_iter=1000)
    scratch_model.fit(X_train, y_train)
    scratch_preds = scratch_model.predict(X_test)
    scratch_metrics, scratch_cm = evaluate_model(y_test, scratch_preds, "Logistic Regression (Scratch)")

    # 3. Train and evaluate Library Random Forest
    print("\n[3/4] Training Model 2: Random Forest (Library)...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=21)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_metrics, rf_cm = evaluate_model(y_test, rf_preds, "Random Forest (Library)")

    # 4. Print & Save Comparison Results
    print("\n[4/4] Model Performance Comparison:")
    results_df = pd.DataFrame([scratch_metrics, rf_metrics])
    print("-" * 50)
    print(results_df.to_string(index=False))
    print("-" * 50)

    # Save metrics table to CSV
    metrics_path = os.path.join("outputs", "report_metrics.csv")
    results_df.to_csv(metrics_path, index=False)
    print(f"Metrics table exported to '{metrics_path}'")

    # Generate and save confusion matrices plot
    cm_plot_path = os.path.join("outputs", "figures", "confusion_matrices.png")
    plot_confusion_matrices(
        scratch_cm, rf_cm, 
        "Logistic Regression (Scratch)", "Random Forest (Library)", 
        encoder.classes_, save_path=cm_plot_path
    )

    print("\nPipeline execution complete! All plots saved in 'outputs/figures/'.\n")

if __name__ == "__main__":
    main()