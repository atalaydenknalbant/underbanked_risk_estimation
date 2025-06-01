#!/usr/bin/env python
"""Generate DICE counterfactuals for a random sample."""

import argparse

import dice_ml
import joblib
import pandas as pd

from config import TARGET
from data import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--n", type=int, default=5, help="Rows to explain")
    args = parser.parse_args()

    df = load_dataset(args.data)
    pipe = joblib.load(args.model)

    dice_data = dice_ml.Data(dataframe=df, continuous_features=None, outcome_name=TARGET)
    dice_model = dice_ml.Model(model=pipe, backend="sklearn")
    explainer = dice_ml.Dice(dice_data, dice_model, method="random")

    query = df.sample(args.n, random_state=1)
    cf = explainer.generate_counterfactuals(query, total_CFs=3, desired_class="opposite")

    print(cf.visualize_as_dataframe())


if __name__ == "__main__":
    main()
