"""LightGBM : un seul modèle pour les 3 arcs.
1. Test   : entraîné sur tout avant le test final, on prédit le test final -> RMSE par arc.
2. Rendu  : entraîné sur tout (test final compris), on prédit le 07/11 -> 11/11 -> model/output_lightgbm.csv.
Lancer depuis la racine : uv run python model/modele_lightgbm.py"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))  # pour importer dataset_create et rendu

import lightgbm as lgb
import pandas as pd
from dataset_create import avec_recent, charger, test_final
from rendu import donnees_rendu, exporter

# Débit : meilleurs réglages trouvés à la main sur le backtest (90.8 contre 92.5 avec les réglages par défaut).
# Occupation : réglages par défaut (le huber alpha=100 est à l'échelle du débit, pas de l'occupation).
PARAMS = {"debit": dict(n_estimators=1000, learning_rate=0.05, num_leaves=127, objective="huber", alpha=100),
          "occupation": dict(n_estimators=500, learning_rate=0.05, num_leaves=31, min_child_samples=20)}
rmse = lambda y, p, axe: ((y - p) ** 2).groupby(axe, observed=True).mean() ** .5  # RMSE par arc

df = charger(imputer=True)
recent, preds = avec_recent(df), {}
for cible, params in PARAMS.items():
    modele = lambda X, y: lgb.LGBMRegressor(verbose=-1, random_state=0, **params).fit(X, y)

    # 1. Test : appris jusqu'au 01/11 00h, prédit du 01/11 01h à la dernière vraie valeur
    X_train, X_test, y_train, y_test = test_final(df, cible)
    p = modele(X_train, y_train).predict(X_test).clip(0)
    print(f"\nTest final {cible} :\n", pd.DataFrame({"modele": rmse(y_test, p, X_test.axe),
                                                  "naif_lag7j": rmse(y_test, X_test.lag_7j, X_test.axe)}).round(1))

    # 2. Rendu : appris jusqu'à la dernière vraie valeur, prédit les 360 heures du challenge
    X_train, y_train, X_rendu, cles = donnees_rendu(recent, cible)
    preds[cible] = modele(X_train, y_train).predict(X_rendu)

exporter(cles, preds["debit"], preds["occupation"], "lightgbm", Path(__file__).parent)
