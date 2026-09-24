"""
Запуск полного пайплайна без Jupyter.
Результаты: метрики в консоли + графики в папке проекта.
"""
import sys
sys.path.append('.')

from src.preprocessing import load_and_preprocess
from src.train import train_baselines, train_lgbm_optuna
from src.evaluate import plot_pr_curves
import shap
import matplotlib.pyplot as plt

print("="*60)
print("🚀 Credit Card Fraud Detection - Full Pipeline")
print("="*60)

# 1. Загрузка данных
print("\n[1/5] Загрузка и предобработка данных...")
X_train, X_test, y_train, y_test, scaler = load_and_preprocess('data/creditcard.csv')
print(f"  Train: {len(X_train)} samples, Fraud: {y_train.mean()*100:.3f}%")
print(f"  Test:  {len(X_test)} samples, Fraud: {y_test.mean()*100:.3f}%")

# 2. Baseline модели
print("\n[2/5] Обучение Baseline моделей...")
baseline_results = train_baselines(X_train, X_test, y_train, y_test)

# 3. LightGBM + Optuna
print("\n[3/5] Подбор гиперпараметров LightGBM (Optuna, ~5-10 мин)...")
lgbm_model, lgbm_proba = train_lgbm_optuna(X_train, X_test, y_train, y_test, n_trials=50)
baseline_results['LightGBM + Optuna'] = lgbm_proba

# 4. PR-кривые
print("\n[4/5] Построение PR-кривых...")
plot_pr_curves(baseline_results, y_test)
print("  ✅ График сохранён: pr_curves_comparison.png")

# 5. SHAP
print("\n[5/5] SHAP интерпретируемость...")
explainer = shap.TreeExplainer(lgbm_model)
shap_values = explainer.shap_values(X_test)
plt.figure()
shap.summary_plot(shap_values[1], X_test, show=False, title="Feature Importance for Fraud Detection")
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=150)
print("  ✅ График сохранён: shap_summary.png")

print("\n" + "="*60)
print("🎉 ГОТОВО! Проверьте файлы:")
print("   - pr_curves_comparison.png")
print("   - shap_summary.png")
print("   - models/best_lgbm_fraud.joblib")
print("="*60)
