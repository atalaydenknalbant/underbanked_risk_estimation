"""Utility helpers for preprocessing and pipeline assembly."""

from typing import List, Tuple

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def split_cols(df: pd.DataFrame, cols: List[str]) -> Tuple[List[str], List[str]]:
    """Return separate lists of categorical and numerical columns."""
    cat, num = [], []
    for c in cols:
        if (
            df[c].dtype == "object"
            or str(df[c].dtype).startswith("category")
            or df[c].dtype == "bool"
        ):
            cat.append(c)
        else:
            num.append(c)
    return cat, num


def _build_preprocessor(cat_cols: List[str], num_cols: List[str]) -> ColumnTransformer:
    """Create preprocessing transformer with sensible defaults."""
    num_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )

    cat_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=True)),
        ]
    )

    return ColumnTransformer(
        [("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
        remainder="drop",
        sparse_threshold=0.3,
    )


def make_pipeline(
    estimator, df: pd.DataFrame, feature_names: List[str]
) -> Pipeline:
    """Attach preprocessing and return a full sklearn Pipeline."""
    cat_cols, num_cols = split_cols(df[feature_names], feature_names)
    pre = _build_preprocessor(cat_cols, num_cols)
    return Pipeline([("pre", pre), ("clf", estimator)])
