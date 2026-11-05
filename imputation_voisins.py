"""Imputation des heures manquantes du débit de nos 3 arcs à partir des capteurs des rues voisines.

Usage : charger(imputer=True) dans dataset_create, ou directement imputer_voisins(df).
Explications : explication_voisins_v2.pdf
"""

import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parent
UNPROCESSED_DIR = ROOT / "data" / "unprocessed_data"
EXTERNAL_DIR = ROOT / "data" / "external"

VOISINS = {
    "Champs-Elysees": ["Av_des_Champs_Elysees", "Av_Champs_Elysees", "AV_Champs_Elysees", "Tilsitt", "Presbourg", "Av_Marceau"],
    "Convention": ["Convention", "Lecourbe", "Vaugirard"],
    "Saints-Peres": ["Sts_Peres", "Quai_Voltaire", "Quai_Malaquais", "Bd_St_Germain"],
}
RUES_VOISINES = sorted({r for rues in VOISINS.values() for r in rues})
API_COMPTAGES = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/comptages-routiers-permanents/exports/csv"


def _lire_voisins_txt(f: Path) -> pd.DataFrame:
    df = pd.read_csv(f, sep=";", usecols=["iu_ac", "libelle", "t_1h", "q"], encoding="latin-1")
    return df[df.libelle.isin(RUES_VOISINES)]


def voisins(cache: Path = EXTERNAL_DIR / "voisins_debit.csv") -> pd.DataFrame:
    """Débit horaire des capteurs des rues voisines : index UTC, une colonne '<rue>_<arc_id>' par capteur.
    Archives locales jusqu'au 30/09/2025 + API open data ensuite. 1er appel ~10 min, puis lu depuis le cache
    (le supprimer pour le mettre à jour)."""
    if not cache.exists():
        with ThreadPoolExecutor(8) as ex:
            old = pd.concat(ex.map(_lire_voisins_txt, sorted(UNPROCESSED_DIR.glob("*/*.txt"))))
        old["t_1h"] = (pd.to_datetime(old.t_1h).dt.tz_localize("Europe/Paris", ambiguous="NaT", nonexistent="NaT")
                       .dt.tz_convert("UTC"))
        where = f"libelle in ({','.join(map(repr, RUES_VOISINES))})"
        r = requests.get(API_COMPTAGES, params={"select": "iu_ac,libelle,t_1h,q", "where": where}, timeout=900)
        new = pd.read_csv(io.StringIO(r.content.decode("utf-8-sig")), sep=";")
        new["t_1h"] = pd.to_datetime(new.t_1h, utc=True)
        v = pd.concat([old[old.t_1h < new.t_1h.min()], new]).dropna(subset=["t_1h"])
        v["capteur"] = v.libelle.str.replace("AV_", "Av_") + "_" + v.iu_ac.astype(str)
        v.pivot_table(index="t_1h", columns="capteur", values="q", aggfunc="mean").to_csv(cache)
    v = pd.read_csv(cache, index_col=0)
    v.index = pd.to_datetime(v.index, utc=True)
    return v


def imputer_voisins(df: pd.DataFrame, seuil: float = 0.8) -> pd.DataFrame:
    """Complète les NaN de `debit` par un boosting sur les capteurs voisins, et met `impute_debit` = True sur ces lignes.
    Capteurs gardés : corrélation >= seuil ET mesurés pendant >= 24 h de trous de la cible (exclut les copies de l'arc)."""
    V, df = voisins(), df.assign(impute_debit=False)
    for axe, g in df.groupby("axe"):
        y = g.debit
        X = V.reindex(g.datetime_utc).set_index(g.index)
        cols = [c for c in X if c.rsplit("_", 1)[0] in VOISINS[axe]]
        corr, utile = X[cols].corrwith(y, min_periods=500), X.loc[y.isna(), cols].notna().sum() >= 24
        cols = corr[(corr >= seuil) & utile].index
        dispo = X[cols].notna().any(axis=1)
        X = X[cols].assign(heure=g.datetime.dt.hour, jour_sem=g.datetime.dt.day_of_week)
        model = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, random_state=0).fit(X[y.notna() & dispo], y[y.notna() & dispo])
        trou = (y.isna() & dispo)[lambda s: s].index
        df.loc[trou, "debit"] = model.predict(X.loc[trou]).clip(0)
        df.loc[trou, "impute_debit"] = True
    return df
