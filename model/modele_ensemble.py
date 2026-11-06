"""Ensemble retenu : moyenne simple de LightGBM + XGBoost (Lucie) + Timestral.
Choisi sur la semaine de Lucie + le test final (notebooks/ensemble.ipynb) : 64.1 de RMSE débit en moyenne, contre 65.8 pour LightGBM seul.
Lancer depuis la racine : uv run python model/modele_ensemble.py  -> test final + model/output_ensemble.csv (le rendu)"""
import pandas as pd
from commun import lancer
from modele_lightgbm import predire as lightgbm
from modele_timestral import predire as timestral
from modele_xgboost import predire as xgboost

MODELES = [lightgbm, xgboost, timestral]


def predire(full, t0, cible):
    return pd.concat([m(full, t0, cible).clip(0) for m in MODELES], axis=1).mean(axis=1)


if __name__ == "__main__":
    lancer(predire, "ensemble")
