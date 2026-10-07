"""Навчання легкої churn-моделі для tool `get_customer_churn_risk`.

Чому окрема модель: `churn_model.pkl` з telco-churn-mlops-synthetic-08 важить ~190 МБ
і не проходить у GitHub (ліміт 100 МБ), а CI має запускати агента з моделлю.
Тут той самий датасет, той самий набір фіч, але компактний pipeline (<1 МБ).

Запуск:
    python scripts/train_model.py "../telco-churn-mlops-synthetic-08/data/telco_customers.csv"
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

ROOT = Path(__file__).resolve().parents[1]
CAT = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
       "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
       "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"]
NUM = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

FIRST = ["Olena", "Taras", "Iryna", "Andrii", "Maria", "Dmytro", "Sofia", "Oleh", "Anna", "Yurii"]
LAST = ["Kovalenko", "Shevchenko", "Bondarenko", "Tkachenko", "Kravchenko", "Melnyk", "Boyko", "Moroz"]


def add_synthetic_pii(df: pd.DataFrame) -> pd.DataFrame:
    """CRM-поля з ПЕРСОНАЛЬНИМИ даними (синтетичні) — щоб було що захищати guardrail-ом."""
    rng = np.random.default_rng(13)
    first = rng.choice(FIRST, len(df))
    last = rng.choice(LAST, len(df))
    df = df.copy()
    df.insert(1, "full_name", [f"{f} {l}" for f, l in zip(first, last)])
    df.insert(2, "email", [f"{f.lower()}.{l.lower()}{i}@example-mail.com"
                           for i, (f, l) in enumerate(zip(first, last))])
    df.insert(3, "phone", [f"+380 67 {rng.integers(100, 999)} {rng.integers(10, 99)} {rng.integers(10, 99)}"
                           for _ in range(len(df))])
    return df


def main(src: str) -> None:
    df = pd.read_csv(src)
    df = df.sample(n=min(25000, len(df)), random_state=42).reset_index(drop=True)
    y = (df["Churn"] == "Yes").astype(int)
    X = df[CAT + NUM]
    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X, y, df.index, test_size=0.2, random_state=42, stratify=y)

    pipe = Pipeline([
        ("prep", ColumnTransformer([
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT),
            ("num", "passthrough", NUM)])),
        ("clf", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=42)),
    ])
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    print(f"ROC-AUC={roc_auc_score(y_te, proba):.3f}  F1={f1_score(y_te, proba > 0.5):.3f}")

    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump({"pipeline": pipe, "features": CAT + NUM}, ROOT / "models" / "churn_model.joblib")

    # 300 клієнтів з test-частини -> "CRM" агента (комітиться в git, потрібна для CI)
    crm = df.loc[idx_te].head(300).drop(columns=["Churn", "RecordDate"])
    crm = add_synthetic_pii(crm)
    crm.to_csv(ROOT / "data" / "customers_crm.csv", index=False)
    print("saved models/churn_model.joblib and data/customers_crm.csv",
          f"({(ROOT / 'models' / 'churn_model.joblib').stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../telco-churn-mlops-synthetic-08/data/telco_customers.csv")
