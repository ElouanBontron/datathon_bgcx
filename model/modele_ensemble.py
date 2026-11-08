"""Modèle final : moyenne de LightGBM, XGBoost et Timestral (1/3 chacun).
Lancer depuis la racine : uv run python model/modele_ensemble.py  -> data/output/output_flowerrangers.csv (le rendu)"""
import pandas as pd
from commun import lancer
from modele_lightgbm import predire as lightgbm
from modele_timestral import predire as timestral
from modele_xgboost import predire as xgboost

MODELES = {"lightgbm": lightgbm, "xgboost": xgboost, "timestral": timestral}


def predire(full, t0, cible):
    return pd.concat([m(full, t0, cible).clip(0) for m in MODELES.values()], axis=1).mean(axis=1)


if __name__ == "__main__":
    lancer(predire, "flowerrangers")
