"""Load, validate, and encode the IBM Telco Customer Churn data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path("data/telco_customer_churn.csv")
ID_COLUMN, TARGET = "customerID", "Churn"
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
ADD_ONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]


def load(path: Path = DATA_PATH) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty or ID_COLUMN not in frame or TARGET not in frame:
        raise ValueError("Expected the Telco churn columns, including customerID and Churn")
    if frame[ID_COLUMN].duplicated().any():
        raise ValueError("customerID must be unique")
    if not set(frame[TARGET].unique()) <= {"Yes", "No"}:
        raise ValueError("Churn must be Yes or No")
    # 11 brand-new customers (tenure 0) have a blank TotalCharges: they have not been billed yet.
    frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
    blank = frame["TotalCharges"].isna()
    if (frame.loc[blank, "tenure"] != 0).any():
        raise ValueError("Blank TotalCharges is only expected for customers with zero tenure")
    frame.loc[blank, "TotalCharges"] = 0.0
    return frame


def features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """One-hot encode categories and add two engineered features."""
    target = (frame[TARGET] == "Yes").astype(int)
    data = frame.drop(columns=[ID_COLUMN, TARGET]).copy()
    data["avg_monthly_spend"] = data["TotalCharges"] / np.maximum(data["tenure"], 1)
    data["add_on_services"] = (data[ADD_ONS] == "Yes").sum(axis=1)
    encoded = pd.get_dummies(data, drop_first=True).astype(float)
    return encoded, target
