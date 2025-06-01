"""Dataset loading and lightweight feature engineering."""

from pathlib import Path
from typing import Union

import pandas as pd

from config import TODAY


def prep_dates(df: pd.DataFrame, col: str) -> pd.Series:
    """Convert date column to age (years) relative to TODAY anchor."""
    return (TODAY - pd.to_datetime(df[col], errors="coerce")).dt.days / 365.25


def parse_subscription_list(cell):
    """Return the number of active digital subscriptions in a stringified list."""
    if pd.isna(cell):
        return 0
    try:
        return len(eval(cell))  # noqa: S307 – safe on synthetic data
    except Exception:
        return 0


def load_dataset(path: Union[str, Path]) -> pd.DataFrame:
    """Load CSV and derive helper columns used by models."""
    df = pd.read_csv(path)

    if "phone_purchase_date" in df.columns:
        df["phone_age_yrs"] = prep_dates(df, "phone_purchase_date")

    if "monthly_subscriptions" in df.columns:
        df["subscription_count"] = df["monthly_subscriptions"].apply(
            parse_subscription_list
        )

    return df
