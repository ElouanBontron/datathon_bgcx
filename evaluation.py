"""Les 3 tests communs à tous les notebooks, chacun comparé au naïf lag_7j (même heure 7 jours avant) :
1. semaine_lucie : dernière donnée le ven. 07/11/2025 -> sam. 08 -> mer. 12/11/2025 (même semaine un an avant, 11/11 inclus)
2. test_final    : vraies valeurs de la fin des données (01/11/2026) à la dernière heure téléchargée
3. backtest      : tous les vendredis de ORIGINES, RMSE par arc moyennée sur les semaines
Chaque modèle garde ses propres features : seul l'examen est commun.
Les prédictions sont gardées dans data/predictions/<nom>_<cible>.csv (pour le notebook ensemble).

Dans un notebook :  from evaluation import trois_tests ;  P = trois_tests(df, modele, "lightgbm", CIBLE)"""
from pathlib import Path

import numpy as np
import pandas as pd

from dataset_create import ORIGINES, split, test_final

DOSSIER = Path(__file__).parent / "data" / "predictions"
BACKTEST = False  # True : ajoute le backtest sur les 105 vendredis (long : 105 entraînements par modèle)


def resume(P: pd.DataFrame, cols=("pred", "naif_lag7j")) -> None:
    """Affiche la RMSE par arc (moyenne sur les semaines) et la moyenne des 3 arcs, pour chaque test."""
    e2 = P[list(cols)].sub(P.y, axis=0).pow(2).assign(test=P.test, t0=P.t0, axe=P.axe)
    r = e2.groupby(["test", "t0", "axe"]).mean().pow(.5).groupby(["test", "axe"]).mean()
    for test in ["semaine_lucie", "test_final", "backtest"]:
        if test in r.index:
            t = r.loc[test]
            print(f"\n===== {test}\n", pd.concat([t, t.mean().to_frame("moyenne").T]).round(1))


def trois_tests(df: pd.DataFrame, modele, nom: str, cible: str = "debit", features=None) -> pd.DataFrame:
    """modele(X_train, y_train, X_test) -> prédictions. Lance les 3 tests, affiche le résumé, sauvegarde les prédictions."""
    lignes = []

    def un_test(test, t0, X_train, X_test, y_train, y_test):
        if len(X_test):
            lignes.append(pd.DataFrame({"test": test, "t0": t0, "axe": X_test.axe.astype(str), "ligne": X_test.index, "y": y_test,
                                        "pred": np.clip(np.asarray(modele(X_train, y_train, X_test), dtype=float), 0, None),
                                        "naif_lag7j": X_test.lag_7j}))

    un_test("semaine_lucie", "2025-11-07", *split(df, "2025-11-07", cible=cible, features=features))
    un_test("test_final", "final", *test_final(df, cible, features))
    for t0 in ORIGINES if BACKTEST else []:
        un_test("backtest", t0, *split(df, t0, cible=cible, features=features))
    P = pd.concat(lignes, ignore_index=True)
    DOSSIER.mkdir(exist_ok=True)
    P.to_csv(DOSSIER / f"{nom}_{cible}.csv", index=False)
    resume(P)
    return P
