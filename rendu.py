"""Rendu du challenge (ppt slides 11-13) : prédire du 07/11 00h au 11/11 23h, 3 arcs x 120 h = 360 lignes,
tester le format avec la fonction du jury et exporter output_<groupe>.csv.

Dans un notebook (depuis notebooks/, après sys.path.append("..")) :
    from dataset_create import avec_recent
    from rendu import donnees_rendu, exporter
    recent, preds = avec_recent(df), {}  # on entraîne aussi sur les données du test final
    for cible in ["debit", "occupation"]:
        X_train, y_train, X_rendu, cles = donnees_rendu(recent, cible)
        preds[cible] = modele(X_train, y_train, X_rendu)
    exporter(cles, preds["debit"], preds["occupation"], "nom_du_groupe")
Exemples complets : model/modele_lightgbm.py et model/modele_chronos.py."""
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from dataset_create import FEATURES, TOUTES, ajouter_futur

DEBUT, FIN = "2026-11-07 00:00", "2026-11-11 23:00"  # heures notées par le jury (heure de Paris)
NOMS = {"Champs-Elysees": "Champs-Elysées", "Convention": "Convention", "Saints-Peres": "Saint-Pères"}  # noms du jury


def rendu(full: pd.DataFrame) -> pd.Series:
    """Masque des 360 heures notées."""
    return full.datetime.between(pd.Timestamp(DEBUT, tz="Europe/Paris"), pd.Timestamp(FIN, tz="Europe/Paris"))


def donnees_rendu(df: pd.DataFrame, cible: str = "debit", features: list | str | None = None):
    """-> X_train, y_train (tout df), X_rendu (les 360 heures à prédire, features comme le jour J)
    et cles (arc + datetime de chaque ligne de X_rendu, à passer à exporter)."""
    full, t0 = ajouter_futur(df, cible, FIN)
    X = full[TOUTES if features == "toutes" else features or FEATURES]
    X = X.astype({c: float for c in X if X[c].dtype == bool})
    train = (full.datetime < t0) & full[cible].notna()
    return X[train], full.loc[train, cible], X[rendu(full)], full.loc[rendu(full), ["axe", "datetime"]]


def exporter(cles: pd.DataFrame, debit, occupation, groupe: str, dossier: str | Path = Path(__file__).parent) -> pd.DataFrame:
    """Met les prédictions au format du jury (valeurs négatives ramenées à 0), le vérifie et écrit output_<groupe>.csv."""
    out = pd.DataFrame({"arc": cles.axe.astype(str).map(NOMS).astype(object),
                        "datetime": cles.datetime.dt.strftime("%Y-%m-%d %H:%M").astype(object),
                        "debit_horaire": np.clip(np.asarray(debit, dtype=float), 0, None),
                        "taux_occupation": np.clip(np.asarray(occupation, dtype=float), 0, None)}).reset_index(drop=True)
    test_format_and_export_output(out, dossier, groupe)
    return out


def test_format_and_export_output(output_df, output_directory, name_of_the_group):
    """Fonction fournie par le jury (ppt slide 13), recopiée telle quelle."""
    output_columns = {"arc": object, "datetime": object, "debit_horaire": float, "taux_occupation": float}
    # 1. Check relevant columns are in output dataframe
    assert sorted(list(output_df.columns)) == list(
        output_columns.keys()
    ), "Some columns are missing or unnecessary columns are in output"
    # 2. Check types
    for col, col_type in output_columns.items():
        assert output_df[col].dtype == col_type, f"Column {col} does not have type {col_type}"
    # 3. Check datetime string has right format
    try:
        output_df.datetime.apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M"))
    except ValueError as e:
        raise e
    # 4. Check `arc` columns has right values
    assert sorted(list(output_df["arc"].unique())) == [
        "Champs-Elysées",
        "Convention",
        "Saint-Pères",
    ], "Output does not have expected unique values for column `arc`"
    # 5. Check dataframe has right number of rows
    assert output_df.shape[0] == 360, f"Expected number of rows is 360, output has {output_df.shape[0]}"
    # 6. Export output
    output_path = Path(output_directory) / f"output_{name_of_the_group}.csv"
    print(f"[SAVE OUTPUT] Saving output here: {output_path}")
    output_df[output_columns.keys()].to_csv(output_path, sep=";")
