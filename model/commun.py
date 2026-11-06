"""Commun à tous les modèles. Un modèle = une fonction predire(full, t0, cible) -> pd.Series indexée comme les lignes
de full à partir de t0. full contient toutes les heures et toutes les features ; la cible y est vide à partir de t0.
- jeu_val   : rejoue le challenge à une origine t0 (comme split : 1 j de trou, 120 h notées, vraies valeurs seulement)
- jeu_final : de la fin des données à la dernière vraie valeur téléchargée (comme test_final)
- lancer    : test final puis rendu (entraîné avec les données récentes, 07/11 -> 11/11, export du csv)"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))  # pour importer dataset_create et rendu

import pandas as pd
from dataset_create import HORIZON, ORIGINES, ajouter_futur, avec_recent, charger, feature_engineering, vraies_valeurs
from rendu import FIN, exporter, rendu

VAL = ORIGINES  # tous les vendredis (105 semaines, nov. 2024 -> oct. 2026)


def jeu_val(df, cible, t0, min_reel=0.85):
    """-> full (cible vide à partir de t0, vraie valeur dans 'vrai'), t0, masque des heures notées."""
    t0 = pd.Timestamp(t0, tz="Europe/Paris")
    debut = t0 + pd.Timedelta("1D")
    full = feature_engineering(df, cible, t0)
    full = full[full.datetime < debut + pd.Timedelta(hours=HORIZON)]
    full = full.assign(vrai=full[cible], **{cible: full[cible].where(full.datetime < t0)})
    test = (full.datetime >= debut) & ~full[f"impute_{cible}"] & full.vrai.notna()
    test &= test.groupby(full.axe, observed=True).transform("sum") >= min_reel * HORIZON
    return full, t0, test


def jeu_final(df, cible):
    """-> full jusqu'à la dernière vraie valeur (dans 'vrai'), t0 = fin des données, masque des heures notées."""
    vrai = vraies_valeurs(df)
    full, t0 = ajouter_futur(df, cible, vrai.datetime.max())
    full["vrai"] = full[["axe", "datetime"]].astype({"axe": str}).merge(vrai, how="left")[cible].values
    return full, t0, (full.datetime >= t0) & full.vrai.notna()


def rmse(full, test, p):
    """RMSE par arc sur les heures notées."""
    e2 = (p.reindex(full.index[test]).clip(0) - full.vrai[test]) ** 2
    return e2.groupby(full.axe[test], observed=True).mean() ** .5


def lancer(predire, nom):
    """Test final (RMSE par arc), puis rendu : appris jusqu'à la dernière vraie valeur -> model/output_<nom>.csv."""
    df = charger(imputer=True)
    recent, preds = avec_recent(df), {}
    for cible in ["debit", "occupation"]:
        full, t0, test = jeu_final(df, cible)
        print(f"Test final {cible} :", rmse(full, test, predire(full, t0, cible)).round(1).to_dict(), flush=True)
        full, t0 = ajouter_futur(recent, cible, FIN)
        preds[cible] = predire(full, t0, cible)
    cles = full.loc[rendu(full), ["axe", "datetime"]]
    exporter(cles, preds["debit"][cles.index], preds["occupation"][cles.index], nom, Path(__file__).parent)
