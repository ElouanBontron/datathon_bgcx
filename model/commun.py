"""Commun aux modèles. Un modèle = une fonction predire(full, t0, cible) -> pd.Series des prédictions pour les lignes
de full à partir de t0. full contient toutes les heures et toutes les features ; la cible y est vide à partir de t0.
- jeu_val   : rejoue le challenge à un vendredi t0 (1 jour de trou, puis 120 heures notées, vraies valeurs seulement)
- jeu_final : test final, de la fin des données (01/11/2026) à la dernière vraie valeur téléchargée
- lancer    : test final, puis prédiction du 07/11 au 11/11 avec les données récentes -> data/output/output_<nom>.csv"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))  # pour importer dataset_create

import pandas as pd
from dataset_create import HORIZON, ORIGINES, ajouter_futur, avec_recent, charger, feature_engineering, vraies_valeurs
from rendu import DEBUT, FIN, exporter


def jeu_val(df, cible, t0, min_reel=0.85):
    """-> full (cible vide à partir de t0, vraie valeur dans 'vrai'), t0, masque des heures notées.
    Un arc n'est noté que s'il a au moins 85 % de vraies valeurs sur les 120 heures."""
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
    """Affiche le test final, puis prédit le rendu en apprenant jusqu'à la dernière vraie valeur."""
    df = charger(imputer=True)
    recent, preds = avec_recent(df), {}
    for cible in ["debit", "occupation"]:
        full, t0, test = jeu_final(df, cible)
        print(f"Test final {cible} :", rmse(full, test, predire(full, t0, cible)).round(1).to_dict(), flush=True)
        full, t0 = ajouter_futur(recent, cible, FIN)
        preds[cible] = predire(full, t0, cible)
    cles = full.loc[full.datetime >= pd.Timestamp(DEBUT, tz="Europe/Paris"), ["axe", "datetime"]]
    exporter(cles, preds["debit"][cles.index], preds["occupation"][cles.index], nom)
