# Credit Risk Estimation with Non-Financial Features: Evidence from a Synthetic Istanbul Dataset – Code Release

This repository contains a fully reproducible pipeline that accompanies the paper *Credit Risk Estimation with Non-Financial Features: Evidence from a Synthetic Istanbul Dataset* (ArXiv pre‑print, 2025).  The code trains and evaluates gradient‑boosting and logistic models on a 100 000‑row **synthetic** Istanbul dataset that purposefully *omits* all bureau‑style variables.  

## Project structure

| File | Purpose |
|------|---------|
| `config.py` | Central place for constants such as feature blocks, target name and random seed |
| `data.py` | Lightweight loader and feature‑engineering helpers (date parsing, list counting) |
| `utils.py` | Column‑type detection and a `make_pipeline` builder that attaches preprocessing to any estimator |
| `models.py` | Factory that returns correctly‑configured CatBoost, LightGBM, XGBoost or Elastic‑Net LogReg models |
| `train.py` | **CLI** script – five‑fold CV + full‑fit training; writes `joblib` pipeline |
| `evaluate.py` | **CLI** script – scores a saved pipeline on a hold‑out file |
| `explain.py` | **CLI** script – produces diverse counter‑factuals using DICE‑ML |
| `requirements.txt` | Third‑party dependencies (CatBoost, LightGBM, XGBoost, Dice‑ML, scikit‑learn, etc.) |

## Quick start

```bash
# 1) Install deps
pip install -r requirements.txt

# 2) Train a model (feature blocks can be mixed-and-matched)
python train.py --data istanbul_synthetic_data_v22.csv \
               --features demographic alternative \
               --model lightgbm --out lgbm_full.joblib

# 3) Evaluate on a separate file
python evaluate.py --data istanbul_synthetic_data_v22.csv --model lgbm_full.joblib

# 4) Inspect counter‑factual explanations
python explain.py --data istanbul_synthetic_data_v22.csv --model lgbm_full.joblib --n 10
```


## Reproducibility

* Every random process is anchored by `config.SEED` with a seed of 42.
* The synthetic dataset is deterministic; no personal data is included.
* Pipelines save the full preprocessing graph, ensuring environment portability.

---

Please cite the accompanying paper if this resource assists your research.
