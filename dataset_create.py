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
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
UNPROCESSED_DIR = ROOT / "data" / "unprocessed_data"
PROCESSED_DIR = ROOT / "data" / "processed_data"

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


def charger(axe: str | None = None, out_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Recharge les CSV produits (un axe ou les 3), avec datetime_utc et datetime (Paris) parsés."""
    axes = list(FICHIERS_SORTIE) if axe is None else [axe]
    df = pd.concat([pd.read_csv(Path(out_dir) / FICHIERS_SORTIE[a], low_memory=False) for a in axes],
                   ignore_index=True)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    df["datetime"] = df["datetime_utc"].dt.tz_convert("Europe/Paris")
    return df


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
