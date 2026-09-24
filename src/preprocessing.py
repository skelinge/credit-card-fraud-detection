"""
Модуль предобработки данных для задачи обнаружения мошенничества.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


def load_and_preprocess(filepath: str, test_size: float = 0.2, random_state: int = 42):
    df = pd.read_csv(filepath)
    df['Time_Hours'] = (df['Time'] % (24 * 3600)) / 3600
    df = df.drop('Time', axis=1)
    scaler = RobustScaler()
    df['Amount_Scaled'] = scaler.fit_transform(df[['Amount']])
    df = df.drop('Amount', axis=1)
    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, scaler
