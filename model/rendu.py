"""Format du rendu (brief, slides 11 à 13) : 3 arcs x 120 heures (07/11 00h -> 11/11 23h) = 360 lignes."""
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

DEBUT, FIN = "2026-11-07 00:00", "2026-11-11 23:00"  # heure de Paris
NOMS = {"Champs-Elysees": "Champs-Elysées", "Convention": "Convention", "Saints-Peres": "Saint-Pères"}  # noms attendus
SORTIE = Path(__file__).parents[1] / "data" / "output"


def exporter(cles, debit, occupation, nom):
    """Met les prédictions au format du jury, le vérifie et écrit data/output/output_<nom>.csv."""
    out = pd.DataFrame({"arc": cles.axe.astype(str).map(NOMS).astype(object),
                        "datetime": cles.datetime.dt.strftime("%Y-%m-%d %H:%M").astype(object),
                        "debit_horaire": np.clip(np.asarray(debit, dtype=float), 0, None),
                        "taux_occupation": np.clip(np.asarray(occupation, dtype=float), 0, None)}).reset_index(drop=True)
    SORTIE.mkdir(exist_ok=True)
    test_format_and_export_output(out, SORTIE, nom)


def test_format_and_export_output(output_df, output_directory, name_of_the_group):
    """Fonction fournie par le jury (slide 13), recopiée telle quelle."""
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
