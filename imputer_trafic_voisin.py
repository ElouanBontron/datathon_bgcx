"""Impute missing traffic flows from correlated nearby sensors.

Run from this directory with:
    python imputer_trafic_voisin.py

The source dataset is never modified. The script writes a new dataset with
flow_observed, flow_imputed and is_imputed columns, plus correlation and
validation reports next to it. Model validation randomly masks 10% of eligible
observations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline


ROOT = Path(__file__).resolve().parent
TARGET_AXES = ("Champs-Elysees", "Convention", "Saints-Peres")
TARGET_ARC_IDS = {"Champs-Elysees": 4264, "Convention": 5671, "Saints-Peres": 191}
MIN_CORRELATION_PAIRS = 30
CORRELATION_THRESHOLD = 0.80
VALIDATION_FRACTION = 0.10
RANDOM_SEED = 42


def _read_target(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    required = {"axe", "datetime_utc", "debit"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans {path.name}: {sorted(missing)}")

    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True, errors="raise")
    df["flow_observed"] = pd.to_numeric(df["debit"], errors="coerce")
    if df.duplicated(["axe", "datetime_utc"]).any():
        raise ValueError("Le fichier cible contient plusieurs lignes par axe et par heure.")
    absent_axes = set(TARGET_AXES).difference(df["axe"].dropna().unique())
    if absent_axes:
        raise ValueError(f"Axes cibles absents du fichier: {sorted(absent_axes)}")
    return df


def _read_candidates(directory: Path) -> tuple[pd.DataFrame, dict[str, int | None]]:
    candidate_columns: dict[str, pd.Series] = {}
    candidate_arc_ids: dict[str, int | None] = {}
    source_files = sorted(directory.glob("comptages-routiers-permanents_*.csv"))
    if not source_files:
        raise FileNotFoundError(f"Aucun fichier candidat trouvé dans {directory}")

    required = [
        "Identifiant arc",
        "Libelle",
        "Date et heure de comptage",
        "Débit horaire",
        "Libelle noeud amont",
        "Libelle noeud aval",
    ]
    for path in source_files:
        source = pd.read_csv(
            path, sep=";", encoding="utf-8-sig", usecols=required, low_memory=False
        )
        source["datetime_utc"] = pd.to_datetime(
            source["Date et heure de comptage"], utc=True, errors="coerce"
        )
        source["flow"] = pd.to_numeric(source["Débit horaire"], errors="coerce")
        source = source.dropna(subset=["datetime_utc"])

        identity_columns = [
            "Identifiant arc",
            "Libelle",
            "Libelle noeud amont",
            "Libelle noeud aval",
        ]
        for identity, rows in source.groupby(identity_columns, dropna=False, sort=True):
            arc_id, label, upstream, downstream = identity
            name = (
                f"{path.stem}|{arc_id}|{label}|"
                f"{upstream}->{downstream}"
            )
            if name in candidate_columns:
                raise ValueError(f"Identifiant de capteur candidat en double: {name}")
            series = rows.groupby("datetime_utc")["flow"].mean().rename(name)
            candidate_columns[name] = series
            candidate_arc_ids[name] = int(arc_id) if pd.notna(arc_id) else None

    if not candidate_columns:
        raise ValueError(f"Aucune mesure de candidat exploitable dans {directory}")
    return pd.concat(candidate_columns.values(), axis=1).sort_index(), candidate_arc_ids


def _make_model(kind: str):
    regressor = (
        LinearRegression()
        if kind == "linear"
        else RandomForestRegressor(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
    )
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        regressor,
    )


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    mae = float(mean_absolute_error(actual, predicted))
    return rmse, mae


def impute(
    input_path: Path,
    candidates_dir: Path,
    output_path: Path,
) -> tuple[Path, Path, Path]:
    result = _read_target(input_path)
    candidates, candidate_arc_ids = _read_candidates(candidates_dir)
    result["flow_imputed"] = result["flow_observed"]
    result["is_imputed"] = False
    result["imputation_method"] = np.where(
        result["flow_observed"].notna(), "observed", "missing"
    )

    correlation_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for axis_index, axis in enumerate(TARGET_AXES):
        axis_rows = result.index[result["axe"].eq(axis)]
        target = result.loc[axis_rows, "flow_observed"]
        all_features = candidates.reindex(
            pd.DatetimeIndex(result.loc[axis_rows, "datetime_utc"])
        )
        all_features.index = axis_rows
        direct_candidates = [
            name for name in all_features.columns
            if candidate_arc_ids[name] == TARGET_ARC_IDS[axis]
        ]
        features = all_features.drop(columns=direct_candidates)
        direct_features = all_features[direct_candidates]

        observed = target.notna()
        eligible = observed & features.notna().any(axis=1)
        eligible_positions = np.flatnonzero(eligible.to_numpy())
        rng = np.random.default_rng(RANDOM_SEED + axis_index)
        validation_count = int(np.ceil(len(eligible_positions) * VALIDATION_FRACTION))
        validation_positions = (
            rng.choice(eligible_positions, size=validation_count, replace=False)
            if validation_count
            else np.array([], dtype=int)
        )
        validation_mask = pd.Series(False, index=axis_rows)
        if len(validation_positions):
            validation_mask.iloc[validation_positions] = True
        training_observed = observed & ~validation_mask

        direct_values = pd.Series(np.nan, index=axis_rows, dtype=float)
        if not direct_features.empty:
            conflicting_direct_values = direct_features.nunique(axis=1, dropna=True).gt(1)
            if conflicting_direct_values.any():
                raise ValueError(
                    f"Sources divergentes pour l'arc cible {axis} "
                    f"à {int(conflicting_direct_values.sum())} dates."
                )
            direct_values = direct_features.bfill(axis=1).iloc[:, 0]
            direct_fill = target.isna() & direct_values.notna()
            result.loc[axis_rows[direct_fill.to_numpy()], "flow_imputed"] = (
                direct_values.loc[direct_fill].to_numpy()
            )
            result.loc[axis_rows[direct_fill.to_numpy()], "is_imputed"] = True
            result.loc[axis_rows[direct_fill.to_numpy()], "imputation_method"] = (
                "same_arc_source"
            )
        direct_fill_count = int((target.isna() & direct_values.notna()).sum())

        candidate_correlations: list[tuple[str, float]] = []
        for candidate_name in direct_candidates:
            paired = pd.concat(
                [target[training_observed], all_features[candidate_name][training_observed]],
                axis=1,
            ).dropna()
            correlation = (
                float(paired.iloc[:, 0].corr(paired.iloc[:, 1]))
                if len(paired) >= MIN_CORRELATION_PAIRS
                else np.nan
            )
            correlation_rows.append(
                {
                    "axe": axis,
                    "candidate": candidate_name,
                    "n_paires": len(paired),
                    "correlation": correlation,
                    "qualite_correlation": "meme_arc",
                    "status": "source_directe_exclue_du_modele",
                }
            )

        for candidate_name in features.columns:
            paired = pd.concat(
                [target[training_observed], features[candidate_name][training_observed]],
                axis=1,
            ).dropna()
            correlation = (
                float(paired.iloc[:, 0].corr(paired.iloc[:, 1]))
                if len(paired) >= MIN_CORRELATION_PAIRS
                else np.nan
            )
            retained = bool(
                np.isfinite(correlation) and correlation >= CORRELATION_THRESHOLD
            )
            quality = (
                "excellente"
                if np.isfinite(correlation) and correlation > 0.95
                else "tres_bonne"
                if np.isfinite(correlation) and correlation >= 0.90
                else "acceptable"
                if np.isfinite(correlation) and correlation >= 0.80
                else "a_eviter"
                if np.isfinite(correlation)
                else "non_evaluable"
            )
            correlation_rows.append(
                {
                    "axe": axis,
                    "candidate": candidate_name,
                    "n_paires": len(paired),
                    "correlation": correlation,
                    "qualite_correlation": quality,
                    "status": (
                        "retenu"
                        if retained
                        else "correlation_insuffisante"
                        if np.isfinite(correlation)
                        else "pas_assez_de_paires_ou_variance_nulle"
                    ),
                }
            )
            if retained:
                candidate_correlations.append((candidate_name, correlation))

        selected = [name for name, _ in candidate_correlations]
        correlations_by_name = dict(candidate_correlations)
        selected.sort(key=correlations_by_name.__getitem__, reverse=True)

        if not selected:
            summary_rows.append(
                {
                    "axe": axis,
                    "candidats_retenus": 0,
                    "modele_choisi": "aucun",
                    "heures_imputees": direct_fill_count,
                    "heures_reprises_meme_arc": direct_fill_count,
                    "heures_estimees_modele": 0,
                    "heures_manquantes_restantes": int(
                        result.loc[axis_rows, "flow_imputed"].isna().sum()
                    ),
                }
            )
            validation_rows.append(
                {
                    "axe": axis,
                    "modele": "aucun",
                    "n_observations_connues": int(observed.sum()),
                    "n_candidates_eligibles": len(eligible_positions),
                    "n_masquees": validation_count,
                    "n_train": int(training_observed.sum()),
                    "n_validation": 0,
                    "validation_type": "masquage_aleatoire_10_pct",
                    "debit_moyen_validation": np.nan,
                    "rmse": np.nan,
                    "mae": np.nan,
                    "rmse_pct_debit_moyen": np.nan,
                    "mae_pct_debit_moyen": np.nan,
                }
            )
            continue

        X = features[selected]
        train_mask = training_observed & X.notna().any(axis=1)
        validation_eval = (
            validation_mask
            & observed
            & X.notna().any(axis=1)
        )
        X_train = X.loc[train_mask]
        y_train = target.loc[train_mask]
        X_validation = X.loc[validation_eval]
        y_validation = target.loc[validation_eval]

        if len(X_train) < 2:
            summary_rows.append(
                {
                    "axe": axis,
                    "candidats_retenus": len(selected),
                    "modele_choisi": "aucun_donnees_entrainement_insuffisantes",
                    "heures_imputees": direct_fill_count,
                    "heures_reprises_meme_arc": direct_fill_count,
                    "heures_manquantes_restantes": int(
                        result.loc[axis_rows, "flow_imputed"].isna().sum()
                    ),
                }
            )
            continue

        model_scores: dict[str, float] = {}
        for kind in ("linear", "random_forest"):
            model = _make_model(kind)
            model.fit(X_train, y_train)
            if len(X_validation):
                predictions = np.maximum(model.predict(X_validation), 0.0)
                rmse, mae = _metrics(y_validation.to_numpy(), predictions)
                mean_validation_flow = float(y_validation.mean())
            else:
                rmse, mae = np.nan, np.nan
                mean_validation_flow = np.nan
            mae_percent = (
                mae / mean_validation_flow * 100
                if np.isfinite(mean_validation_flow) and mean_validation_flow != 0
                else np.nan
            )
            rmse_percent = (
                rmse / mean_validation_flow * 100
                if np.isfinite(mean_validation_flow) and mean_validation_flow != 0
                else np.nan
            )
            validation_rows.append(
                {
                    "axe": axis,
                    "modele": kind,
                    "n_observations_connues": int(observed.sum()),
                    "n_candidates_eligibles": len(eligible_positions),
                    "n_masquees": validation_count,
                    "n_train": len(X_train),
                    "n_validation": len(X_validation),
                    "validation_type": "masquage_aleatoire_10_pct",
                    "debit_moyen_validation": mean_validation_flow,
                    "rmse": rmse,
                    "mae": mae,
                    "rmse_pct_debit_moyen": rmse_percent,
                    "mae_pct_debit_moyen": mae_percent,
                }
            )
            if np.isfinite(rmse) and len(X_validation) >= 2:
                model_scores[kind] = rmse

        chosen_kind = min(model_scores, key=model_scores.get) if model_scores else "linear"
        final_train = observed & X.notna().any(axis=1)
        final_model = _make_model(chosen_kind)
        final_model.fit(X.loc[final_train], target.loc[final_train])

        still_missing = result.loc[axis_rows, "flow_imputed"].isna()
        predictable = still_missing & X.notna().any(axis=1)
        if predictable.any():
            predicted_flow = np.maximum(final_model.predict(X.loc[predictable]), 0.0)
            result.loc[axis_rows[predictable.to_numpy()], "flow_imputed"] = predicted_flow
            result.loc[axis_rows[predictable.to_numpy()], "is_imputed"] = True
            result.loc[axis_rows[predictable.to_numpy()], "imputation_method"] = (
                "correlated_sensor_model"
            )

        model_fill_count = int(predictable.sum())
        imputed_count = direct_fill_count + model_fill_count
        remaining_count = int(result.loc[axis_rows, "flow_imputed"].isna().sum())
        summary_rows.append(
            {
                "axe": axis,
                "candidats_retenus": len(selected),
                "modele_choisi": chosen_kind,
                "heures_imputees": imputed_count,
                "heures_reprises_meme_arc": direct_fill_count,
                "heures_estimees_modele": model_fill_count,
                "heures_manquantes_restantes": remaining_count,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    correlation_path = output_path.with_name(f"{output_path.stem}_correlations.csv")
    validation_path = output_path.with_name(f"{output_path.stem}_validation.csv")
    pd.DataFrame(correlation_rows).to_csv(correlation_path, index=False)
    pd.DataFrame(validation_rows).to_csv(validation_path, index=False)

    print(f"Dataset écrit: {output_path}")
    print(f"Corrélations écrites: {correlation_path}")
    print(f"Validation écrite: {validation_path}")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    return output_path, correlation_path, validation_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Impute les débits manquants à partir de capteurs corrélés."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "trafic_3_arcs_complet.csv",
        help="CSV cible d'origine (il ne sera pas modifié).",
    )
    parser.add_argument(
        "--candidates-dir",
        type=Path,
        default=ROOT,
        help="Dossier contenant les comptages-routiers-permanents_*.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "trafic_3_arcs_impute.csv",
        help="Nouveau CSV de sortie.",
    )
    args = parser.parse_args()
    impute(args.input, args.candidates_dir, args.output)


if __name__ == "__main__":
    main()
