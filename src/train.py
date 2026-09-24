"""
Обучение моделей: Baseline + LightGBM с Optuna.
"""
import lightgbm as lgb
import optuna
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from .evaluate import evaluate_model


def train_baselines(X_train, X_test, y_train, y_test):
    results = {}
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    results['Logistic Regression'] = lr.predict_proba(X_test)[:, 1]
    evaluate_model(y_test, results['Logistic Regression'], "Logistic Regression")
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    results['Random Forest'] = rf.predict_proba(X_test)[:, 1]
    evaluate_model(y_test, results['Random Forest'], "Random Forest")
    return results


def train_lgbm_optuna(X_train, X_test, y_train, y_test, n_trials: int = 50):
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    spw = neg_count / pos_count

    def objective(trial):
        params = {
            'objective': 'binary', 'metric': 'average_precision', 'boosting_type': 'gbdt',
            'n_estimators': 1000,
            'learning_rate': trial.suggest_float('lr', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'scale_pos_weight': spw, 'random_state': 42, 'verbose': -1
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        y_proba = model.predict_proba(X_test)[:, 1]
        return average_precision_score(y_test, y_proba)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"\n[OK] Лучший PR-AUC: {study.best_value:.4f}")

    best_params = study.best_params
    best_params.update({'objective': 'binary', 'metric': 'average_precision',
                        'boosting_type': 'gbdt', 'n_estimators': 1000,
                        'scale_pos_weight': spw, 'random_state': 42, 'verbose': -1})
    final_model = lgb.LGBMClassifier(**best_params)
    final_model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
                    callbacks=[lgb.early_stopping(50, verbose=False)])
    y_proba = final_model.predict_proba(X_test)[:, 1]
    evaluate_model(y_test, y_proba, "LightGBM + Optuna")
    dump(final_model, "models/best_lgbm_fraud.joblib")
    print("[OK] Модель сохранена в models/best_lgbm_fraud.joblib")
    return final_model, y_proba
