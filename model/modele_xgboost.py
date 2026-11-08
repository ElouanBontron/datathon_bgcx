"""XGBoost (Lucie) : un modèle par arc, vacances en one-hot.
Lancer depuis la racine : uv run python model/modele_xgboost.py"""
import pandas as pd
from commun import lancer
from dataset_create import FEATURES
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

NUMERIQUES = [f for f in FEATURES if f not in ("axe", "vacances")]


def modele():
    colonnes = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), ["vacances"]), ("num", "passthrough", NUMERIQUES)])
    xgb = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
    return Pipeline([("colonnes", colonnes), ("xgb", xgb)])


def predire(full, t0, cible):
    X = full[FEATURES].astype({c: float for c in FEATURES if full[c].dtype == bool})
    out = []
    for axe in X.axe.unique():
        train = (full.datetime < t0) & full[cible].notna() & (X.axe == axe)
        futur = (full.datetime >= t0) & (X.axe == axe)
        m = modele().fit(X.loc[train, ["vacances", *NUMERIQUES]], full.loc[train, cible])
        out.append(pd.Series(m.predict(X.loc[futur, ["vacances", *NUMERIQUES]]), X.index[futur]))
    return pd.concat(out)


if __name__ == "__main__":
    lancer(predire, "xgboost")
