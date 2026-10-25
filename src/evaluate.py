import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_model(y_true, y_pred, model_name="Model"):
    # Calculate performance metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro')
    rec = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        "Model": model_name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4)
    }
    return metrics, cm

def plot_confusion_matrices(cm1, cm2, name1, name2, class_names, save_path="../outputs/figures/confusion_matrices.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Plot Model 1 Confusion Matrix (Logistic Regression)
    sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', ax=axes[0], xticklabels=class_names, yticklabels=class_names)
    axes[0].set_title(name1)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    # Plot Model 2 Confusion Matrix (Random Forest)
    sns.heatmap(cm2, annot=True, fmt='d', cmap='Greens', ax=axes[1], xticklabels=class_names, yticklabels=class_names)
    axes[1].set_title(name2)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
