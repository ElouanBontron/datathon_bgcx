"""Generate before/after missing-flow plots and model error summaries.

Run this script from any directory:
    python rapport_trafic_manquants.py

The imputed dataset and validation report are created by
imputer_trafic_voisin.py. The source traffic CSV is read but never modified.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator, PercentFormatter


ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "trafic_3_arcs_complet.csv"
IMPUTED_PATH = ROOT / "trafic_3_arcs_impute.csv"
VALIDATION_PATH = ROOT / "trafic_3_arcs_impute_validation.csv"
SMOOTHING_DAYS = 30

ZONES = (
    ("Champs-Elysees", "Champs-Élysées", "#167D9A"),
    ("Convention", "Convention", "#D17A22"),
    ("Saints-Peres", "Saint-Pères", "#7256A8"),
)


def _daily_missing_rates(
    path: Path, flow_column: str, output_column: str
) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        usecols=["axe", "datetime_utc", flow_column],
        low_memory=False,
    )
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True, errors="raise")
    df["jour"] = df["datetime_utc"].dt.floor("D")

    daily_frames = []
    for axis, label, _ in ZONES:
        zone = df.loc[df["axe"].eq(axis)]
        if zone.empty:
            raise ValueError(f"Aucune donnée pour {label} dans {path.name}.")
        daily = zone.groupby("jour")[flow_column].apply(
            lambda values: values.isna().mean() * 100
        )
        daily_frames.append(
            daily.rolling(
                window=SMOOTHING_DAYS,
                center=True,
                min_periods=1,
            )
            .mean()
            .rename(output_column)
            .to_frame()
            .assign(axe=axis)
        )
    return pd.concat(daily_frames).reset_index()


def _plot_missing_rates(
    daily: pd.DataFrame, output_path: Path, title: str, subtitle: str
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True, sharey=True)

    for ax, (axis, label, color) in zip(axes, ZONES):
        zone = daily.loc[daily["axe"].eq(axis)].sort_values("jour")
        if zone.empty:
            raise ValueError(f"Aucune série quotidienne pour {label}.")
        values = zone["taux_manquant"].to_numpy(dtype=float)
        dates = zone["jour"]
        ax.fill_between(dates, values, 0, color=color, alpha=0.13, linewidth=0)
        ax.plot(dates, values, color=color, linewidth=1.8)
        ax.set_title(label, loc="left", pad=8)
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_locator(MultipleLocator(20))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.grid(axis="y", color="#D9E1E5", linewidth=0.8)
        ax.grid(axis="x", visible=False)
        ax.set_xlim(dates.min(), dates.max())

    axes[1].set_ylabel("Débits manquants (moyenne mobile)")
    axes[-1].set_xlabel("Jour")
    fig.suptitle(title, fontsize=17, fontweight="bold", x=0.06, ha="left", y=0.985)
    fig.text(0.06, 0.947, subtitle, fontsize=10, color="#53636B")
    fig.patch.set_facecolor("#FFFFFF")
    fig.tight_layout(rect=(0.035, 0.035, 0.99, 0.925), h_pad=1.8)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _model_errors(validation_path: Path) -> pd.DataFrame:
    validation = pd.read_csv(validation_path)
    required = {
        "axe",
        "modele",
        "n_validation",
        "debit_moyen_validation",
        "rmse",
        "mae",
        "rmse_pct_debit_moyen",
        "mae_pct_debit_moyen",
    }
    missing_columns = required.difference(validation.columns)
    if missing_columns:
        raise ValueError(
            "Le rapport de validation ne contient pas encore les colonnes "
            f"{sorted(missing_columns)}. Relance d'abord imputer_trafic_voisin.py."
        )

    selected_rows = []
    for axis, label, _ in ZONES:
        options = validation.loc[
            validation["axe"].eq(axis) & validation["rmse"].notna()
        ]
        if options.empty:
            selected_rows.append(
                {
                    "zone": label,
                    "modele_retenu": "aucun",
                    "heures_test": 0,
                    "debit_moyen_test_veh_h": float("nan"),
                    "mae_veh_h": float("nan"),
                    "mae_pct": float("nan"),
                    "rmse_veh_h": float("nan"),
                    "rmse_pct": float("nan"),
                    "statut": "aucun candidat corrélé retenu",
                }
            )
            continue
        row = options.loc[options["rmse"].idxmin()].copy()
        selected_rows.append(
            {
                "zone": label,
                "modele_retenu": row["modele"],
                "heures_test": int(row["n_validation"]),
                "debit_moyen_test_veh_h": float(row["debit_moyen_validation"]),
                "mae_veh_h": float(row["mae"]),
                "mae_pct": float(row["mae_pct_debit_moyen"]),
                "rmse_veh_h": float(row["rmse"]),
                "rmse_pct": float(row["rmse_pct_debit_moyen"]),
                "statut": "masquage aléatoire (10 %)",
            }
        )

    return pd.DataFrame(selected_rows)    


def main() -> None:
    for required_path in (INPUT_PATH, IMPUTED_PATH, VALIDATION_PATH):
        if not required_path.is_file():
            raise FileNotFoundError(f"Fichier requis introuvable: {required_path}")

    before = _daily_missing_rates(INPUT_PATH, "debit", "taux_manquant")
    after = _daily_missing_rates(IMPUTED_PATH, "flow_imputed", "taux_manquant")
    start = min(before["jour"].min(), after["jour"].min()).strftime("%d/%m/%Y")
    end = max(before["jour"].max(), after["jour"].max()).strftime("%d/%m/%Y")
    period = f"{start}–{end}"
    smoothing = f"Moyenne mobile centrée sur {SMOOTHING_DAYS} jours"

    before_path = ROOT / "taux_debits_manquants_3_zones_avant_imputation.png"
    after_path = ROOT / "taux_debits_manquants_3_zones_apres_imputations.png"
    _plot_missing_rates(
        before,
        before_path,
        "Taux de débits manquants avant imputation",
        f"{smoothing} · données du CSV brut · {period}",
    )
    _plot_missing_rates(
        after,
        after_path,
        "Taux de débits manquants après imputation",
        f"{smoothing} · données après reprise et imputation · {period}",
    )

    errors = _model_errors(VALIDATION_PATH)
    errors_path = ROOT / "erreurs_modeles_trafic.csv"
    errors.to_csv(errors_path, index=False)
    display = errors.copy()
    for column in ("debit_moyen_test_veh_h", "mae_veh_h", "rmse_veh_h"):
        display[column] = display[column].map(
            lambda value: f"{value:.2f}" if pd.notna(value) else "—"
        )
    for column in ("mae_pct", "rmse_pct"):
        display[column] = display[column].map(
            lambda value: f"{value:.2f}%" if pd.notna(value) else "—"
        )

    print(f"Graphe avant imputation: {before_path}")
    print(f"Graphe après imputation: {after_path}")
    print(f"Erreurs détaillées: {errors_path}")
    print("\nErreurs des modèles retenus (masquage aléatoire de 10 %):")
    print(display.to_string(index=False))


if __name__ == "__main__":
    main()
