
"""Model catalogue

This module defines a dictionary `model_specs` mapping
readable tags to (estimator, feature_list) tuples.
It is imported by train.py / evaluate.py so that command‑line
`--model TAG` resolves to the corresponding estimator + feature subset.
"""

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from config import demographic, full_features

# ----------------------------------------------------------------------
# Model registry
# ----------------------------------------------------------------------

model_specs: dict[str, tuple[object, list[str]]] = {}

# CatBoost
for tag, feats in (("CatBoost-Demo", demographic), ("CatBoost-Full", full_features)):
    model_specs[tag] = (
        CatBoostClassifier(
            iterations=1000,
            depth=6,
            learning_rate=0.3,
            task_type="GPU",
            devices="0",
            loss_function="Logloss",
            auto_class_weights="Balanced",
            od_type="Iter",
            od_wait=100,
            bootstrap_type="Bernoulli",
            subsample=0.8,
            l2_leaf_reg=3,
            verbose=False,
            random_state=42,
            allow_writing_files=False,
        ),
        feats,
    )

# LightGBM
for tag, feats in (("LGBM-Demo", demographic), ("LGBM-Full", full_features)):
    model_specs[tag] = (
        LGBMClassifier(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.3,
            device="gpu",
            class_weight="balanced",
            force_row_wise=True,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        ),
        feats,
    )

# XGBoost
for tag, feats in (("XGB-Demo", demographic), ("XGB-Full", full_features)):
    model_specs[tag] = (
        XGBClassifier(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.3,
            tree_method="hist",
            predictor="gpu_predictor",
            gpu_id=0,
            scale_pos_weight=int(75809 / 24191),
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
        feats,
    )

# Logistic Regression
for tag, feats in (("LogReg-Demo", demographic), ("LogReg-Full", full_features)):
    model_specs[tag] = (
        LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
            tol=1e-4,
        ),
        feats,
    )

# Random Forest
for tag, feats in (("RF-Demo", demographic), ("RF-Full", full_features)):
    model_specs[tag] = (
        RandomForestClassifier(
            n_estimators=1000,
            max_depth=6,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
        feats,
    )

# Decision Tree
for tag, feats in (("DT-Demo", demographic), ("DT-Full", full_features)):
    model_specs[tag] = (
        DecisionTreeClassifier(
            max_depth=6,
            class_weight="balanced",
            random_state=42,
        ),
        feats,
    )

# ----------------------------------------------------------------------
# Convenience helpers
# ----------------------------------------------------------------------

def list_models() -> list[str]:
    """Return available model tags."""
    return list(model_specs.keys())


def get_model(tag: str):
    """Return (estimator, feature list) for *tag*."""
    if tag not in model_specs:
        raise KeyError(
            f"Unknown model tag: {tag}. Available tags: {', '.join(list_models())}"
        )
    return model_specs[tag]
