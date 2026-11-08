"""LightGBM : un seul modèle pour les 3 arcs (axe est une feature), entraîné sur tout l'historique avant t0.
Lancer depuis la racine : uv run python model/modele_lightgbm.py"""
import lightgbm as lgb
import pandas as pd
from commun import lancer
from dataset_create import FEATURES

# débit : réglages trouvés à la main sur le backtest ; occupation : réglages par défaut
PARAMS = {"debit": dict(n_estimators=1000, learning_rate=0.05, num_leaves=127, objective="huber", alpha=100),
          "occupation": dict(n_estimators=500, learning_rate=0.05, num_leaves=31, min_child_samples=20)}


def predire(full, t0, cible):
    X = full[FEATURES].astype({c: float for c in FEATURES if full[c].dtype == bool})
    train, futur = (full.datetime < t0) & full[cible].notna(), full.datetime >= t0
    m = lgb.LGBMRegressor(verbose=-1, random_state=0, **PARAMS[cible]).fit(X[train], full.loc[train, cible])
    return pd.Series(m.predict(X[futur]), X.index[futur])


if __name__ == "__main__":
    lancer(predire, "lightgbm")
