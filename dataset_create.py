"""Construction des datasets trafic des 3 arcs du challenge (2014 -> 2026).

Entrée  : data/unprocessed_data/<année>/
            - 2014 ... 2025 : archives open data Ville de Paris (*.txt, ancien format)
            - 2026          : exports du challenge (*.csv, nouveau format, depuis le 01/10/2025)
Sortie  : data/processed_data/{champs_elysees,convention,saints_peres}.csv

Une ligne = un axe x une heure, grille horaire complète (les heures sans mesure existent
avec debit / occupation vides et ligne_absente = True). Voir data/processed_data/README.md.

Usage :
    .venv/Scripts/python dataset_create.py                 # toutes les années
    .venv/Scripts/python dataset_create.py --annees 2024 2025 2026
    .venv/Scripts/python dataset_create.py --jobs 4        # lecture des années en parallèle

Depuis un notebook :
    from dataset_create import dataset_annee, dataset_complet
    df_2019 = dataset_annee(2019)
    df = dataset_complet()
"""

import argparse
import io
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from imputation_voisins import API_COMPTAGES, imputer_voisins

ROOT = Path(__file__).resolve().parent
UNPROCESSED_DIR = ROOT / "data" / "unprocessed_data"
PROCESSED_DIR = ROOT / "data" / "processed_data"
EXTERNAL_DIR = ROOT / "data" / "external"

# Arcs cibles : axe -> (noeud amont, noeud aval). Identique dans les deux formats.
ARCS_CIBLES = {
    "Champs-Elysees": ("Av_Champs_Elysees-Washington", "Av_Champs_Elysees-Berri"),
    "Convention": ("Lecourbe-Convention", "Convention-Blomet"),
    "Saints-Peres": ("Sts_Peres-Voltaire", "Sts_Peres-Universite"),
}

# Nom du fichier de sortie par axe
FICHIERS_SORTIE = {
    "Champs-Elysees": "champs_elysees.csv",
    "Convention": "convention.csv",
    "Saints-Peres": "saints_peres.csv",
}

COLONNES_SORTIE = ["axe", "arc_id", "datetime_utc", "debit", "occupation",
                   "etat_trafic", "etat_arc_code", "ligne_absente", "source", "etat_arc"]

# --- Ancien format (old_data, *.txt) ---
# On ignore 'dessin' (géométrie, très lourde)
COLONNES_ANCIEN = ["iu_ac", "libelle_nd_amont", "libelle_nd_aval", "t_1h", "q", "k",
                   "etat_trafic", "etat_barre"]
ETAT_TRAFIC_ANCIEN = {0: "Inconnu", 1: "Fluide", 2: "Pré-saturé", 3: "Saturé", 4: "Bloqué"}

# --- Nouveau format (exports du challenge, *.csv) ---
COLONNES_RECENT = {
    "Identifiant arc": "arc_id",
    "Libelle noeud amont": "libelle_nd_amont",
    "Libelle noeud aval": "libelle_nd_aval",
    "Date et heure de comptage": "datetime_utc",
    "Débit horaire": "debit",
    "Taux d'occupation": "occupation",
    "Etat trafic": "etat_trafic",
    "Etat arc": "etat_arc",
}


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def _filtrer_arcs(brut: pd.DataFrame) -> pd.DataFrame:
    """Garde uniquement les 3 arcs cibles et ajoute la colonne 'axe'."""
    morceaux = []
    for axe, (amont, aval) in ARCS_CIBLES.items():
        sel = brut[(brut["libelle_nd_amont"] == amont) & (brut["libelle_nd_aval"] == aval)]
        morceaux.append(sel.assign(axe=axe))
    return pd.concat(morceaux, ignore_index=True).drop(columns=["libelle_nd_amont", "libelle_nd_aval"])


def _dedoublonner(df: pd.DataFrame) -> pd.DataFrame:
    """Une seule ligne par (axe, heure) : en priorité celle qui contient une vraie mesure."""
    return (
        df.assign(_vide=df["debit"].isna() & df["occupation"].isna())
        .sort_values(["axe", "datetime_utc", "_vide"])
        .drop_duplicates(subset=["axe", "datetime_utc"], keep="first")
        .drop(columns="_vide")
    )


def _grille_complete(df: pd.DataFrame, debut=None, fin=None) -> pd.DataFrame:
    """Réindexe chaque axe sur une grille horaire UTC complète.

    Les heures ajoutées ont ligne_absente = True et des mesures vides.
    Par défaut la grille va du min au max observé sur l'ensemble des axes.
    """
    debut = df["datetime_utc"].min() if debut is None else debut
    fin = df["datetime_utc"].max() if fin is None else fin
    grille = pd.date_range(debut, fin, freq="h", tz="UTC", name="datetime_utc")

    complet = []
    for axe, g in df.groupby("axe"):
        arc_id = g["arc_id"].mode().iat[0]
        g = g.set_index("datetime_utc").reindex(grille)
        g["axe"], g["arc_id"] = axe, arc_id
        g["ligne_absente"] = g["ligne_absente"].astype("boolean").fillna(True).astype(bool)
        complet.append(g)
    return pd.concat(complet).reset_index()


def _format_sortie(df: pd.DataFrame) -> pd.DataFrame:
    for col in COLONNES_SORTIE:
        if col not in df:
            df[col] = pd.NA
    return df[COLONNES_SORTIE].sort_values(["axe", "datetime_utc"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Lecture d'un dossier
# ---------------------------------------------------------------------------

def _lire_ancien_format(fichiers: list[Path], verbose: bool) -> pd.DataFrame:
    """Archives open data (*.txt). t_1h est en heure de Paris sans fuseau et désigne la FIN
    de l'heure mesurée. Les heures du changement d'heure (inexistante / ambiguë) sont retirées.
    etat_barre : seul le code 2 (= Barré) est fiable, gardé brut dans etat_arc_code."""
    morceaux = []
    for i, f in enumerate(fichiers, 1):
        brut = pd.read_csv(f, sep=";", usecols=COLONNES_ANCIEN, encoding="latin-1")
        morceaux.append(_filtrer_arcs(brut))
        if verbose:
            print(f"  [{i}/{len(fichiers)}] {f.name}", end="\r")
    if verbose:
        print()

    df = pd.concat(morceaux, ignore_index=True).rename(columns={
        "iu_ac": "arc_id", "t_1h": "datetime_utc", "q": "debit", "k": "occupation",
        "etat_barre": "etat_arc_code",
    })
    df["etat_trafic"] = df["etat_trafic"].map(ETAT_TRAFIC_ANCIEN)
    df["datetime_utc"] = (
        pd.to_datetime(df["datetime_utc"])
        .dt.tz_localize("Europe/Paris", ambiguous="NaT", nonexistent="NaT")
        .dt.tz_convert("UTC")
    )
    return df.dropna(subset=["datetime_utc"]).assign(source="old_data")


def _lire_nouveau_format(fichiers: list[Path]) -> pd.DataFrame:
    """Exports du challenge (*.csv, séparateur ';', dates ISO avec fuseau)."""
    brut = pd.concat(
        [pd.read_csv(f, sep=";", usecols=list(COLONNES_RECENT), encoding="utf-8-sig") for f in fichiers],
        ignore_index=True,
    ).rename(columns=COLONNES_RECENT)
    df = _filtrer_arcs(brut)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], format="ISO8601", utc=True)
    return df.assign(source="recent")


def dataset_annee(annee: int | str, data_dir: Path = UNPROCESSED_DIR, verbose: bool = True) -> pd.DataFrame:
    """Dataset des 3 arcs pour un dossier data/unprocessed_data/<annee>/.

    Détecte le format tout seul (*.txt = archives, *.csv = exports du challenge).
    Retourne une grille horaire complète par axe, colonnes COLONNES_SORTIE.
    """
    dossier = Path(data_dir) / str(annee)
    txt, csv = sorted(dossier.glob("*.txt")), sorted(dossier.glob("*.csv"))
    if not txt and not csv:
        raise FileNotFoundError(f"Aucun fichier .txt ou .csv dans {dossier}")
    if verbose:
        print(f"{annee} : {len(txt) + len(csv)} fichiers")

    df = _lire_ancien_format(txt, verbose) if txt else _lire_nouveau_format(csv)
    df = _grille_complete(_dedoublonner(df.assign(ligne_absente=False)))
    df["source"] = df["source"].fillna(df["source"].dropna().iat[0])
    return _format_sortie(df)


# ---------------------------------------------------------------------------
# Dataset complet
# ---------------------------------------------------------------------------

def annees_disponibles(data_dir: Path = UNPROCESSED_DIR) -> list[str]:
    return sorted(p.name for p in Path(data_dir).iterdir() if p.is_dir() and p.name.isdigit())


def dataset_complet(annees=None, data_dir: Path = UNPROCESSED_DIR, jobs: int = 1,
                    verbose: bool = True) -> pd.DataFrame:
    """Concatène dataset_annee() pour chaque année et renvoie le dataset 2014 -> 2026.

    - Chevauchement entre années (quelques heures au 1er janvier) : on garde la vraie mesure.
    - Sur la période couverte par les exports récents (source 'recent'), on garde uniquement
      le récent (source officielle du challenge ; identique à old_data sur oct.-déc. 2025).
    - Grille horaire complète sur toute la période (comble aussi les années non générées).
    """
    annees = annees_disponibles(data_dir) if annees is None else [str(a) for a in annees]

    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            par_annee = list(ex.map(dataset_annee, annees, [data_dir] * len(annees), [False] * len(annees)))
    else:
        par_annee = [dataset_annee(a, data_dir, verbose) for a in annees]

    df = pd.concat(par_annee, ignore_index=True)

    recent = df["source"] == "recent"
    if recent.any():
        debut_recent = df.loc[recent, "datetime_utc"].min()
        df = df[recent | (df["datetime_utc"] < debut_recent)]

    df = df.sort_values(["axe", "datetime_utc", "ligne_absente"]).drop_duplicates(
        subset=["axe", "datetime_utc"], keep="first"
    )
    df = _grille_complete(df)
    return _format_sortie(df)


def sauvegarder_par_axe(df: pd.DataFrame, out_dir: Path = PROCESSED_DIR, verbose: bool = True) -> None:
    """Écrit un CSV par axe dans data/processed_data/."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for axe, g in df.groupby("axe"):
        out = out_dir / FICHIERS_SORTIE[axe]
        g.to_csv(out, index=False)
        if verbose:
            print(f"Sauvegardé : {out.relative_to(ROOT)}  ({len(g)} lignes)")


def bilan(df: pd.DataFrame) -> pd.DataFrame:
    """% de débit manquant par axe et par année (heure de Paris)."""
    annee = df["datetime_utc"].dt.tz_convert("Europe/Paris").dt.year
    return df.groupby(["axe", annee])["debit"].apply(lambda s: round(s.isna().mean() * 100, 1)).unstack("axe")


def imputer_profil(df: pd.DataFrame, c: str) -> pd.DataFrame:
    """Trous de `c` comblés par la moyenne des 52 dernières vraies valeurs à la même heure et le même jour de
    semaine (~1 an de données, uniquement du passé). Met `impute_<c>` = True sur ces lignes."""
    profil = lambda s: s.dropna().rolling(52, min_periods=1).mean().reindex(s.index).ffill()
    moy = df.groupby(["axe", df.datetime.dt.hour, df.datetime.dt.day_of_week])[c].transform(profil)
    trou = df[c].isna() & moy.notna()
    return df.assign(**{c: df[c].fillna(moy), f"impute_{c}": trou})


def charger(axe: str | None = None, imputer: bool = False, out_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Dataset nettoyé (un axe ou les 3) : grille horaire complète + météo/calendrier/événements.
    imputer=False : `debit` / `occupation` = NaN quand pas de mesure fiable (à gérer par le modèle, ex. LightGBM).
    imputer=True  : débit comblé via les rues voisines, occupation via son profil passé (imputer_profil)
                    (`impute_debit` / `impute_occupation` = True sur ces lignes)."""
    axes = list(FICHIERS_SORTIE) if axe is None else [axe]
    df = pd.concat([pd.read_csv(Path(out_dir) / FICHIERS_SORTIE[a], low_memory=False) for a in axes],
                   ignore_index=True)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    df["datetime"] = df["datetime_utc"].dt.tz_convert("Europe/Paris")

    for c in ["debit", "occupation"]:
        df[c] = df[c].mask(df.etat_arc == "Invalide")
        df[c] = df[c].mask(df[c] > 1.5 * df.groupby("axe")[c].transform("quantile", 0.999))  # aberrations
    df = df.assign(impute_debit=False, impute_occupation=False, barre=(df.etat_arc_code == 2) | (df.etat_arc == "Barré"))
    if imputer:
        df = imputer_profil(imputer_voisins(df), "occupation")
    return externes(df).drop(columns=["datetime_utc", "etat_arc_code", "ligne_absente", "source",
                                      "etat_arc"]).reset_index(drop=True)


def externes(df: pd.DataFrame) -> pd.DataFrame:
    """Joint météo (sur datetime_utc), calendrier et événements (sur le jour en heure de Paris)."""
    lire = lambda nom, date: pd.read_csv(EXTERNAL_DIR / nom, parse_dates=[date])
    df = df.merge(lire("meteo_paris.csv", "datetime_utc"), on="datetime_utc", how="left")
    df["date"] = df.datetime.dt.normalize().dt.tz_localize(None)
    df = df.merge(lire("calendrier_paris.csv", "date"), on="date", how="left")
    df = df.merge(lire("evenements_paris.csv", "date"), on=["date", "axe"], how="left")
    df["vacances"] = df.nom_vacances.fillna("Aucune").astype("category")
    return df.drop(columns=["date", "nom_vacances"])


BOOLS = ["ferie", "pont", "paris_respire", "evenement", "confinement"]
LAGS = (7, 14, 21, 28)  # jours ; >= horizon pour rester connus le jour J
FEATURES = ["axe", "heure", "jour_sem", "heure_sem", "type_jour", "mois", "lag_7j", "lag_14j", "lag_moy",
            "vacances", *BOOLS,
            "profil_normal",                # backtest : 92.5 -> 89.9, biais réduit sur les 3 arcs
            "fermeture_prevue", "lag_an"]   # backtest : erreur fériés/événements 136 -> 124
#                                     chantier retiré : base vide avant 2020, trou 2024-26, et 10 chantiers
#                                     sur les Champs la semaine notée -> occupation prédite x2.3
# Toutes les features utilisables (= connues le jour J), y compris celles testées puis écartées
TOUTES = [*FEATURES, "lag_21j", "lag_28j", "chantier", "pluie", "neige", "temperature_2m", "relative_humidity_2m",
          "precipitation", "snowfall", "cloud_cover", "wind_speed_10m", "weather_code",
          "veille_ferie", "lendemain_ferie", "profil_ferie"]


def feature_engineering(df: pd.DataFrame, cible: str = "debit", t0=None) -> pd.DataFrame:
    """Toutes les features candidates. Lags de `cible` (valeurs >= t0 traitées comme inconnues, comme le jour J) :
    même heure il y a 7/14/21/28 j + leur moyenne ; calendrier (type de jour semaine/samedi/dimanche-férié) ;
    météo en 2 indicateurs 0/1 (pluie >= 0.5 mm/h, neige > 0 ; NaN si météo inconnue)."""
    meteo = lambda s: s.astype(float).where(df.precipitation.notna())
    v = df[cible] if t0 is None else df[cible].where(df.datetime < t0)
    base = df[["axe", "datetime"]].assign(v=v)
    for k in LAGS:
        lag = base.assign(datetime=base.datetime + pd.Timedelta(days=k)).rename(columns={"v": f"lag_{k}j"})
        df = df.merge(lag, on=["axe", "datetime"], how="left")
    jour, heure, s = df.datetime.dt.dayofweek, df.datetime.dt.hour, pd.Series(v.values, df.index)
    df = df.assign(lag_moy=df[[f"lag_{k}j" for k in LAGS]].mean(axis=1), heure=heure, jour_sem=jour,
                   heure_sem=jour * 24 + heure, mois=df.datetime.dt.month, type_jour=np.select([df.ferie | (jour == 6), jour == 5], [2, 1], 0),
                   pluie=meteo(df.precipitation >= 0.5), neige=meteo(df.snowfall > 0), axe=df.axe.astype("category"))

    # --- Jours spéciaux (tout est calculé avec le passé uniquement) ---
    date = pd.to_datetime(df.datetime.dt.date)
    feries = date[df.ferie].unique()
    barre = df.get("barre", pd.Series(False, df.index)).astype("boolean").fillna(False).astype(bool)
    if t0 is not None:
        barre &= df.datetime < t0  # l'état "barré" n'est connu qu'après coup
    dernieres = lambda x, k: x.dropna().rolling(k, min_periods=1).mean().reindex(x.index).ffill().shift(1)  # moy. des k dernières valeurs connues
    atypique = df.ferie | df.pont | df.evenement | df.paris_respire | barre
    # même heure, même jour de semaine, 6 dernières semaines, sans les jours atypiques
    df["profil_normal"] = s.where(~atypique).groupby([df.axe, jour, heure]).transform(lambda x: x.shift(1).rolling(6, min_periods=1).mean())
    # même heure lors des 3 derniers fériés tombés en semaine
    df["profil_ferie"] = s.where(df.ferie & (jour < 5)).groupby([df.axe, heure]).transform(lambda x: dernieres(x, 3))
    # même jour calendaire, même heure, un an plus tôt (fériés à date fixe : 11/11, 14/07...)
    an = s.groupby([df.axe.astype(str), df.datetime.dt.tz_localize(None)]).mean()
    df["lag_an"] = an.reindex(pd.MultiIndex.from_arrays([df.axe.astype(str), df.datetime.dt.tz_localize(None) - pd.DateOffset(years=1)])).values
    # part des éditions passées du même événement (même date, ou Paris Respire) où cette heure était barrée
    cle = pd.Series(np.where(df.paris_respire, "PR", np.where(df.evenement, df.datetime.dt.strftime("%m-%d"), None)), df.index)
    passe = barre.astype(float).where(cle.notna() & ((df.datetime < t0) if t0 is not None else True))
    df["fermeture_prevue"] = passe.groupby([df.axe, heure, cle]).transform(lambda x: x.expanding().mean().shift(1)).fillna(0)
    return df.assign(veille_ferie=(date + pd.Timedelta("1D")).isin(feries), lendemain_ferie=(date - pd.Timedelta("1D")).isin(feries))


HORIZON = 120  # heures notées : 07/11 00h -> 11/11 23h (sam -> mer)
# coupures du backtest : vendredis 00h (dernière donnée = jeudi soir, test sam -> mer comme le jour J)
ORIGINES = [*pd.date_range("2024-11-01", "2026-10-30", freq="W-FRI").strftime("%Y-%m-%d")]


def split(df: pd.DataFrame, t0: str, gap: str = "1D", cible: str = "debit", min_reel: float = 0.85,
          features: list | str | None = None):
    """Rejoue le challenge à la date t0 (= dernière donnée connue) -> X_train, X_test, y_train, y_test.
    train = tout avant t0 ; test = HORIZON heures démarrant `gap` après t0, vraies valeurs seulement,
    pour les axes ayant >= min_reel de l'horizon observé (sinon l'axe est absent du test).
    features : None = sélection par défaut (FEATURES), "toutes" = TOUTES, ou une liste de colonnes."""
    t0 = pd.Timestamp(t0, tz="Europe/Paris")
    df = feature_engineering(df, cible, t0).dropna(subset=[cible])
    debut = t0 + pd.Timedelta(gap)
    test = df.datetime.between(debut, debut + pd.Timedelta(hours=HORIZON - 1)) & ~df[f"impute_{cible}"]
    test &= test.groupby(df.axe, observed=True).transform("sum") >= min_reel * HORIZON
    train = df.datetime < t0
    X, y = df[TOUTES if features == "toutes" else features or FEATURES], df[cible]
    X = X.astype({c: float for c in X if X[c].dtype == bool})
    return X[train], X[test], y[train], y[test]


def ajouter_futur(df: pd.DataFrame, cible: str, fin) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Ajoute à df les heures vides de la fin des données jusqu'à `fin` (avec calendrier / événements) et calcule
    les features comme le jour J (rien n'est connu après la dernière donnée t0) -> (dataset avec features, t0)."""
    t0, fin = df.datetime.max() + pd.Timedelta("1h"), pd.Timestamp(fin)
    fin = fin.tz_localize("Europe/Paris") if fin.tz is None else fin
    f = pd.DataFrame([(a, h) for a in df.axe.unique() for h in pd.date_range(t0, fin, freq="h")], columns=["axe", "datetime"])
    f = externes(f.assign(datetime_utc=f.datetime.dt.tz_convert("UTC"))).drop(columns="datetime_utc")
    full = pd.concat([df, f.assign(impute_debit=False, impute_occupation=False)], ignore_index=True)
    return feature_engineering(full.assign(vacances=full.vacances.astype("category")), cible, t0), t0


def vraies_valeurs(df: pd.DataFrame) -> pd.DataFrame:
    """Vraies valeurs après la fin de df (axe, datetime, debit, occupation), téléchargées sur l'API open data.
    Cache data/external/test_final.csv : le supprimer pour mettre à jour."""
    cache = EXTERNAL_DIR / "test_final.csv"
    if not cache.exists():
        t0 = df.datetime.max() + pd.Timedelta("1h")
        ids = {str(i): a for a, i in df.groupby("axe").arc_id.agg(lambda s: s.mode()[0]).items()}
        r = requests.get(API_COMPTAGES, timeout=300, params={"select": "iu_ac,t_1h,q,k,etat_barre",
                         "where": f"iu_ac in ({','.join(map(repr, ids))}) and t_1h >= date'{t0:%Y-%m-%d}'"})
        v = pd.read_csv(io.StringIO(r.content.decode("utf-8-sig")), sep=";", dtype={"iu_ac": str})
        v.loc[v.etat_barre == "Invalide", ["q", "k"]] = np.nan
        v.assign(axe=v.iu_ac.map(ids)).rename(columns={"t_1h": "datetime", "q": "debit", "k": "occupation"})[
            ["axe", "datetime", "debit", "occupation"]].to_csv(cache, index=False)
    return pd.read_csv(cache).assign(datetime=lambda v: pd.to_datetime(v.datetime, utc=True).dt.tz_convert("Europe/Paris"))


def avec_recent(df: pd.DataFrame) -> pd.DataFrame:
    """df + les vraies valeurs récentes (celles du test final), heure par heure pour chaque arc (NaN si manquante)."""
    v = vraies_valeurs(df)
    h = pd.date_range(df.datetime.max() + pd.Timedelta("1h"), v.datetime.max(), freq="h")
    v = pd.MultiIndex.from_product([df.axe.astype(str).unique(), h], names=["axe", "datetime"]).to_frame(index=False).merge(v, how="left")
    v = externes(v.assign(datetime_utc=v.datetime.dt.tz_convert("UTC"))).drop(columns="datetime_utc")
    full = pd.concat([df, v.assign(impute_debit=False, impute_occupation=False, barre=False)], ignore_index=True)
    return full.assign(axe=full.axe.astype("category"), vacances=full.vacances.astype("category"))


def test_final(df: pd.DataFrame, cible: str = "debit", features: list | str | None = None, fin=None):
    """Jeu de test FINAL : de la 1re heure après la fin de df jusqu'à `fin` -> X_train, X_test, y_train, y_test.
    Vraies valeurs téléchargées sur l'API open data (cache data/external/test_final.csv, le supprimer pour mettre à jour).
    Features calculées comme le jour J (lags masqués après la dernière donnée). Champs-Élysées absent (capteur invalide)."""
    full, t0 = ajouter_futur(df, cible, fin or vraies_valeurs(df).datetime.max())  # par défaut : jusqu'à la dernière vraie valeur
    y = full[["axe", "datetime"]].astype({"axe": str}).merge(vraies_valeurs(df), how="left")[cible].set_axis(full.index)
    train, test = (full.datetime < t0) & full[cible].notna(), (full.datetime >= t0) & y.notna()
    X = full[TOUTES if features == "toutes" else features or FEATURES]
    X = X.astype({c: float for c in X if X[c].dtype == bool})
    return X[train], X[test], full.loc[train, cible], y[test]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annees", nargs="+", help="années à traiter (défaut : tous les dossiers)")
    parser.add_argument("--jobs", type=int, default=1, help="nombre d'années lues en parallèle")
    args = parser.parse_args()

    df = dataset_complet(args.annees, jobs=args.jobs)
    print(f"\n{len(df)} lignes, du {df['datetime_utc'].min()} au {df['datetime_utc'].max()} (UTC)")
    print("\n% de débit manquant par année :")
    print(bilan(df).to_string())
    print()
    sauvegarder_par_axe(df)
