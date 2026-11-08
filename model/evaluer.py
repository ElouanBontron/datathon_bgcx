"""Compare LightGBM, XGBoost, Timestral, leur moyenne (le modèle final) et le naïf (même heure 7 jours avant).
RMSE par arc, sur 3 tests :
1. semaine 2025 : dernière donnée le ven. 07/11/2025, on prédit sam. 08 -> mer. 12/11/2025 (11/11 férié inclus)
2. test final   : du 01/11/2026 (fin des données) au 06/11/2026 (vraies valeurs de l'API)
3. backtest     : avec --backtest, 1 vendredi sur 4 de nov. 2024 à oct. 2026 (moyenne des semaines, ~1 h)
Usage : uv run python model/evaluer.py [debit|occupation] [--backtest]"""
import sys

import pandas as pd
from commun import ORIGINES, charger, jeu_final, jeu_val, rmse
from modele_ensemble import MODELES

CIBLE = "occupation" if "occupation" in sys.argv else "debit"


def scores(full, t0, test):
    p = {nom: f(full, t0, CIBLE).clip(0) for nom, f in MODELES.items()}
    p["ensemble"] = pd.concat(p.values(), axis=1).mean(axis=1)
    p["naif"] = full.lag_7j[full.datetime >= t0]
    return pd.DataFrame({nom: rmse(full, test, v) for nom, v in p.items()})


def afficher(nom, t):
    print(f"\n===== {CIBLE} | {nom}\n", pd.concat([t, t.mean().to_frame("moyenne").T]).round(2 if CIBLE == "occupation" else 1), flush=True)


df = charger(imputer=True)
afficher("semaine 2025", scores(*jeu_val(df, CIBLE, "2025-11-07")))
afficher("test final", scores(*jeu_final(df, CIBLE)))
if "--backtest" in sys.argv:
    semaines = (jeu_val(df, CIBLE, t0) for t0 in ORIGINES[::4])
    afficher("backtest", pd.concat([scores(*s) for s in semaines if s[2].any()]).groupby(level=0).mean())
