"""
Модуль оценки моделей. Акцент на PR-AUC.
"""
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, PrecisionRecallDisplay
import matplotlib.pyplot as plt


def evaluate_model(y_true, y_pred_proba, model_name: str = "Model"):
    y_pred = (y_pred_proba >= 0.5).astype(int)
    metrics = {
        'pr_auc': average_precision_score(y_true, y_pred_proba),
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'f1': f1_score(y_true, y_pred)
    }
    print(f"\n{'='*50}")
    print(f"[METRICS] {model_name}")
    print(f"{'='*50}")
    print(f"PR-AUC:   {metrics['pr_auc']:.4f}  <- ГЛАВНАЯ МЕТРИКА")
    print(f"ROC-AUC:  {metrics['roc_auc']:.4f}")
    print(f"F1-score: {metrics['f1']:.4f}")
    return metrics


def plot_pr_curves(results_dict, y_test):
    plt.figure(figsize=(10, 7))
    for name, y_proba in results_dict.items():
        PrecisionRecallDisplay.from_predictions(y_test, y_proba, name=name, plot_chance_level=False)
    plt.title("Precision-Recall Curves Comparison", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("pr_curves_comparison.png", dpi=150)
    plt.show()
