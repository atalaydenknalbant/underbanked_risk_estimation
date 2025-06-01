#!/usr/bin/env python
"""Model training entry‑point.

Example
-------
python train.py --data data/istanbul_synthetic.csv --features demographic alternative --model catboost --out catboost.joblib
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm

from config import DEMOGRAPHIC, ALTERNATIVE, TARGET, SEED
from data import load_dataset
from models import get_model
from utils import make_pipeline


def average(scores):
    return {k: round(float(np.mean(v)), 4) for k, v in scores.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Train a credit‑scoring model and persist the fitted pipeline."
    )
    parser.add_argument("--data", required=True, help="Path to CSV dataset")
    parser.add_argument(
        "--features",
        nargs="+",
        default=["demographic", "alternative"],
        help="Space‑separated list of feature blocks or explicit columns",
    )
    parser.add_argument(
        "--model",
        choices=["catboost", "lightgbm", "xgboost", "logreg"],
        default="catboost",
    )
    parser.add_argument("--out", default="model.joblib", help="Output path")
    args = parser.parse_args()

    # Resolve feature names -------------------------------------------------
    feature_blocks = {
        "demographic": DEMOGRAPHIC,
        "alternative": ALTERNATIVE,
        "all": DEMOGRAPHIC + ALTERNATIVE,
    }

    selected = []
    for token in args.features:
        if token in feature_blocks:
            selected.extend(feature_blocks[token])
        else:
            selected.append(token)

    # Ingest data -----------------------------------------------------------
    df = load_dataset(args.data)
    X, y = df[selected], df[TARGET]

    # Build and cross‑validate ---------------------------------------------
    estimator = get_model(args.model)
    pipe = make_pipeline(estimator, df, selected)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = {"auc": [], "f1": [], "precision": [], "recall": []}

    for tr, te in tqdm(cv.split(X, y), total=5, desc="CV"):
        pipe.fit(X.iloc[tr], y.iloc[tr])
        proba = pipe.predict_proba(X.iloc[te])[:, 1]
        pred = (proba >= 0.5).astype(int)

        scores["auc"].append(roc_auc_score(y.iloc[te], proba))
        scores["f1"].append(f1_score(y.iloc[te], pred))
        scores["precision"].append(precision_score(y.iloc[te], pred))
        scores["recall"].append(recall_score(y.iloc[te], pred))

    print("Cross‑validated metrics:", json.dumps(average(scores), indent=2))

    # Fit on full data & persist -------------------------------------------
    pipe.fit(X, y)
    joblib.dump(pipe, args.out)
    print(f"Pipeline saved → {args.out}")


if __name__ == "__main__":
    main()
