"""Real Paris sensor data and an explicit, reproducible historical baseline.

The repository contains modelling experiments, but no fitted production model.
This module therefore exposes a seasonal median, never a claimed AI prediction.
Source timestamps describe the END of an hour; the public API uses its START.
"""

from __future__ import annotations

from datetime import date, timedelta
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]  # racine du repo (data/)


def resolve_data_root(root: Path) -> Path:
    """Support a normal checkout and the original workspace with a nested clone.

    Prefer the checkout's own data directory even when its CSVs are incomplete:
    missing source files should produce a visible error, not a silent fallback.
    """
    root = Path(root)
    if (root / "data" / "processed_data").is_dir():
        return root
    return root / "upstream" / "datathon_bgcx"


UPSTREAM = resolve_data_root(ROOT)
PROCESSED = UPSTREAM / "data" / "processed_data"
PARIS = "Europe/Paris"
METHOD = "Profil historique saisonnier"

# Exact arc geometries from the challenge's geo_shape / geo_point_2d fields.
# These are instrumented street segments, not the geometry of the entire street.
AXES = {
    "Champs-Elysees": {
        "label": "Champs-Élysées", "arc_id": 4264,
        "lat": 48.87153587897718, "lon": 2.3017227924560624,
        "path": [[2.3009951475338775, 48.871777042598445],
                 [2.3024504373782473, 48.871294715355916]],
        "file": "champs_elysees.csv", "geometry_source": "geo_shape du challenge",
    },
    "Convention": {
        "label": "Convention", "arc_id": 5671,
        "lat": 48.838634371808965, "lon": 2.293205602717535,
        "path": [[2.291878306272707, 48.83923810621147],
                 [2.2945328991623635, 48.83803063740647]],
        "file": "convention.csv", "geometry_source": "geo_shape du challenge",
    },
    "Saints-Peres": {
        "label": "Saints-Pères", "arc_id": 191,
        "lat": 48.85728037029827, "lon": 2.332454190717672,
        "path": [[2.3332577811223216, 48.85826006602721],
                 [2.331650600313023, 48.856300674569326]],
        "file": "saints_peres.csv", "geometry_source": "geo_shape du challenge",
    },
}


def _distance_m(path: list[list[float]]) -> float:
    distance = 0.0
    for (lon1, lat1), (lon2, lat2) in zip(path, path[1:]):
        p1, p2 = radians(lat1), radians(lat2)
        delta_lat, delta_lon = p2 - p1, radians(lon2 - lon1)
        distance += 2 * 6_371_000 * asin(sqrt(
            sin(delta_lat / 2) ** 2 + cos(p1) * cos(p2) * sin(delta_lon / 2) ** 2
        ))
    return round(distance, 1)


for _info in AXES.values():
    _info["length_m"] = _distance_m(_info["path"])


def load_history() -> pd.DataFrame:
    """Load every source hour without inventing missing observations.

    Fixed physical validity bounds avoid thresholds that could depend on future
    data. NaN readings and invalid arc states remain NaN, not zero traffic.
    """
    frames = []
    for axe, info in AXES.items():
        path = PROCESSED / info["file"]
        if not path.exists():
            raise FileNotFoundError(
                f"Données absentes : {path}. Récupérez le dépôt avec actualiser_donnees.ps1."
            )
        frame = pd.read_csv(path, low_memory=False, usecols=[
            "axe", "datetime_utc", "debit", "occupation", "etat_arc",
        ])
        end = pd.to_datetime(frame.pop("datetime_utc"), utc=True)
        frame["datetime"] = (end - pd.Timedelta(hours=1)).dt.tz_convert(PARIS)
        invalid = frame.pop("etat_arc").eq("Invalide")
        frame["debit"] = pd.to_numeric(frame["debit"], errors="coerce")
        frame["occupation"] = pd.to_numeric(frame["occupation"], errors="coerce")
        frame.loc[invalid | ~frame["debit"].between(0, 12_000), "debit"] = np.nan
        frame.loc[invalid | ~frame["occupation"].between(0, 100), "occupation"] = np.nan
        frame["axe"] = axe
        frames.append(frame)
    return (pd.concat(frames, ignore_index=True)
            .sort_values(["axe", "datetime"])
            .drop_duplicates(["axe", "datetime"])
            .reset_index(drop=True))


def _calendar() -> dict[date, bool]:
    frame = pd.read_csv(UPSTREAM / "data" / "external" / "calendrier_paris.csv")
    return dict(zip(pd.to_datetime(frame["date"]).dt.date,
                    frame["ferie"].astype(str).str.lower().eq("true")))


def is_holiday(day: date) -> bool:
    """Challenge calendar, with French metropolitan public holidays as fallback."""
    day = pd.Timestamp(day).date()
    calendar = _calendar()
    if day in calendar:
        return bool(calendar[day])
    # Gregorian Easter (Meeus/Jones/Butcher); useful if the date picker is extended.
    year = day.year
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    easter = date(year, (h + l - 7 * m + 114) // 31,
                  (h + l - 7 * m + 114) % 31 + 1)
    fixed = {(1, 1), (5, 1), (5, 8), (7, 14), (8, 15), (11, 1), (11, 11), (12, 25)}
    return ((day.month, day.day) in fixed or
            day in {easter + timedelta(days=n) for n in (1, 39, 50)})


def _prepare_profile(history: pd.DataFrame, calendar: dict[date, bool]) -> pd.DataFrame:
    """Calculate local calendar features once per axis rather than once per hour."""
    # Each axis has its own latest reliable measurement: the Champs sensor is old.
    latest = history.loc[history[["debit", "occupation"]].notna().any(axis=1), "datetime"].max()
    if pd.isna(latest):
        raise ValueError("Aucune observation fiable avant la date sélectionnée.")
    recent = history[(history["datetime"] >= latest - pd.Timedelta(days=3 * 366)) &
                     history[["debit", "occupation"]].notna().any(axis=1)].copy()
    recent = recent[~recent["datetime"].dt.year.isin([2020, 2021])]
    if recent.empty:
        recent = history.copy()
    recent["_hour"] = recent["datetime"].dt.hour
    recent["_weekday"] = recent["datetime"].dt.day_of_week
    recent["_month"] = recent["datetime"].dt.month
    recent["_holiday"] = recent["datetime"].dt.date.map(calendar).fillna(False).astype(bool)
    return recent


def _profile_candidates(recent: pd.DataFrame, stamp: pd.Timestamp,
                        target_holiday: bool) -> pd.DataFrame:
    """Pick past comparable hours, broadening only when a bucket is too sparse."""
    same_hour = recent[recent["_hour"].eq(stamp.hour)]
    historic_holiday = same_hour["_holiday"]
    if target_holiday:
        comparable = same_hour[historic_holiday | same_hour["_weekday"].eq(6)]
    else:
        comparable = same_hour[same_hour["_weekday"].eq(stamp.day_of_week) & ~historic_holiday]
    month_distance = (comparable["_month"] - stamp.month).abs()
    seasonal = comparable[np.minimum(month_distance, 12 - month_distance) <= 2]
    if seasonal[["debit", "occupation"]].notna().all(axis=1).sum() >= 8:
        comparable = seasonal
    if comparable[["debit", "occupation"]].notna().all(axis=1).sum() < 3:
        comparable = same_hour
    if comparable.empty:
        comparable = recent
    return comparable.sort_values("datetime").tail(80)


def _forecast_from_history(day: date, history: pd.DataFrame) -> pd.DataFrame:
    """Separated for deterministic verification of absence of future leakage."""
    day = pd.Timestamp(day).date()
    start = pd.Timestamp(day, tz=PARIS)
    stop = pd.Timestamp(day + timedelta(days=1), tz=PARIS)
    # Retain the correct number of absolute hours through DST transitions.
    stamps = pd.date_range(start, stop, freq="h", inclusive="left")
    past = history[history["datetime"] < start]
    if past.empty:
        raise ValueError("Sélectionnez une date après le début de l’historique disponible.")
    calendar = _calendar()
    target_holiday = is_holiday(day)
    records = []
    for axe in AXES:
        axis_history = past[past["axe"].eq(axe)]
        if axis_history.empty:
            raise ValueError(f"Pas d’historique disponible pour {AXES[axe]['label']}.")
        latest = axis_history.loc[
            axis_history[["debit", "occupation"]].notna().any(axis=1), "datetime"
        ].max()
        profile_history = _prepare_profile(axis_history, calendar)
        for stamp in stamps:
            comparable = _profile_candidates(profile_history, stamp, target_holiday)
            values = comparable[["debit", "occupation"]]
            median = values.median()
            if median.isna().any():
                fallback = axis_history[["debit", "occupation"]].median()
                median = median.fillna(fallback)
            if median.isna().any():
                raise ValueError(f"Mesures insuffisantes pour {AXES[axe]['label']}.")
            occ = float(np.clip(median["occupation"], 0, 100))
            # Empirical spread of comparable past readings, NOT a confidence interval.
            quantiles = values["occupation"].quantile([0.2, 0.8]).fillna(occ)
            records.append({
                "axe": axe, "datetime": stamp,
                "debit": max(0.0, float(median["debit"])), "occupation": occ,
                "lower_occupation": float(np.clip(quantiles.loc[0.2], 0, occ)),
                "upper_occupation": float(np.clip(quantiles.loc[0.8], occ, 100)),
                "method": METHOD, "samples": int(values.notna().all(axis=1).sum()),
                "is_holiday": target_holiday, "last_observation": latest,
                "stale_days": int(max(0, (start - latest).total_seconds() // 86_400)),
            })
    return pd.DataFrame.from_records(records)


def get_forecast(day: date) -> pd.DataFrame:
    """Seasonal baseline per axis and hour; 72 rows on an ordinary Paris day.

    Only readings strictly before the selected local day are used. Historical
    dates therefore replay a real forecast rather than silently using the future.
    No claim of calibrated prediction accuracy is made.
    """
    return _forecast_from_history(day, load_history())


def load_quality() -> pd.DataFrame:
    """Quality for all source hours and the last 30 covered calendar days.

    ``missing_pct`` counts hours where at least one of the two targets is
    unavailable; ``n_observations`` counts complete pairs. ``last_observation``
    is the last hour with either valid target, as in forecast metadata. A
    separate ``last_complete_observation`` identifies the last complete pair.
    Recent completeness is relative to the end of the downloaded source, not
    today's date or the selected forecast day.
    """
    history = load_history()
    recent_start = history["datetime"].max() - pd.Timedelta(days=30)
    records = []
    for axe, group in history.groupby("axe", sort=False):
        valid = group[["debit", "occupation"]].notna().all(axis=1)
        any_valid = group[["debit", "occupation"]].notna().any(axis=1)
        recent = group[group["datetime"] > recent_start]
        recent_valid = recent[["debit", "occupation"]].notna().all(axis=1)
        records.append({
            "axe": axe, "label": AXES[axe]["label"],
            "last_observation": group.loc[any_valid, "datetime"].max(),
            "last_complete_observation": group.loc[valid, "datetime"].max(),
            "missing_pct": float(100 * (1 - valid.mean())),
            "recent_missing_pct": float(100 * (1 - recent_valid.mean())),
            "n_observations": int(valid.sum()), "n_hours": int(len(group)),
        })
    return pd.DataFrame.from_records(records)


def get_source_info() -> dict:
    history = load_history()
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(UPSTREAM), "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL, timeout=3,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        commit = "copie locale"
    return {
        "rows": len(history), "period_start": history["datetime"].min(),
        "period_end": history["datetime"].max(),
        "cutoff": history["datetime"].max() + pd.Timedelta(hours=1),
        "commit": commit, "repository": "https://github.com/ElouanBontron/datathon_bgcx",
        "method": METHOD, "geometry_source": "geo_shape des exports du challenge",
    }


def traffic_state(occupation: float) -> str:
    if pd.isna(occupation):
        return "Inconnu"
    if occupation < 15:
        return "Fluide"
    if occupation < 30:
        return "Pré-saturé"
    if occupation < 50:
        return "Saturé"
    return "Bloqué"


def prepare_map_rows(forecast: pd.DataFrame, hour: int) -> pd.DataFrame:
    """Map-ready exact sensor segments for a chosen local hour."""
    rows = forecast[forecast["datetime"].dt.hour.eq(hour)].copy()
    colors = {"Fluide": [18, 184, 134], "Pré-saturé": [236, 171, 68],
              "Saturé": [239, 111, 76], "Bloqué": [210, 57, 77], "Inconnu": [130, 143, 155]}
    rows["state"] = rows["occupation"].map(traffic_state)
    rows["color"] = rows["state"].map(colors)
    for column in ("label", "arc_id", "lat", "lon", "path", "length_m"):
        rows[column] = rows["axe"].map(lambda key: AXES[key][column])
    return rows.reset_index(drop=True)
