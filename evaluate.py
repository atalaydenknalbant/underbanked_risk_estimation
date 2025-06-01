#!/usr/bin/env python
"""Evaluate a saved pipeline on a hold‑out file."""

import argparse

import joblib
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from config import TARGET
from data import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Hold‑out CSV file")
    parser.add_argument("--model", required=True, help="Path to .joblib pipeline")
    args = parser.parse_args()

    df = load_dataset(args.data)
    pipe = joblib.load(args.model)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    proba = pipe.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)

    print("AUC :", roc_auc_score(y, proba))
    print("F1  :", f1_score(y, pred))
    print("Precision:", precision_score(y, pred))
    print("Recall   :", recall_score(y, pred))


if __name__ == "__main__":
    main()
