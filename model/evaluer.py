"""Compare les modèles sur la validation (VAL : 1 vendredi sur 4, moyenne des RMSE par semaine) et sur le test final.
Lancer depuis la racine : uv run python model/evaluer.py [debit|occupation] [modèles...]
ex. uv run python model/evaluer.py debit lightgbm naif"""
import sys
from importlib import import_module

import pandas as pd
from commun import VAL, charger, jeu_final, jeu_val, rmse

naif = lambda full, t0, cible: full.lag_7j[full.datetime >= t0]  # même heure 7 jours avant
cibles = [a for a in sys.argv[1:] if a in ("debit", "occupation")] or ["debit", "occupation"]
noms = [a for a in sys.argv[1:] if a not in cibles] or ["naif", "lightgbm", "chronos", "prophet"]
MODELES = {n: naif if n == "naif" else import_module(f"modele_{n}").predire for n in noms}

df = charger(imputer=True)
for cible in cibles:
    val = []
    for t0 in VAL:
        full, t0, test = jeu_val(df, cible, t0)
        if test.any():
            val.append(pd.DataFrame({n: rmse(full, test, f(full, t0, cible)) for n, f in MODELES.items()}))
            print(f"  {cible} {t0:%Y-%m-%d} ok", flush=True)
    val = pd.concat(val).groupby(level=0).mean()
    full, t0, test = jeu_final(df, cible)
    final = pd.DataFrame({n: rmse(full, test, f(full, t0, cible)) for n, f in MODELES.items()})
    for nom, t in [("VALIDATION", val), ("TEST FINAL", final)]:
        print(f"\n===== {nom} {cible} ({len(VAL)} origines)" if nom == "VALIDATION" else f"\n===== {nom} {cible}")
        print(pd.concat([t, t.mean().to_frame("moyenne").T]).round(1), flush=True)
