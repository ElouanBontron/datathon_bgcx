"""Delivery planning and transparent, adjustable business estimates.

Traffic occupancy is a sensor occupation percentage, not road speed. The travel
time relationship below is a scenario assumption, not a calibrated prediction.
"""

from __future__ import annotations

import hashlib
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd


SLOT_COLUMNS = ["start_hour", "end_hour", "occupation", "debit", "score", "risk", "label"]
SCHEDULE_COLUMNS = [
    "request_id", "axe", "start", "end", "vehicle", "occupation", "status", "reason",
    "travel_minutes", "service_minutes",
]
ASSUMPTION = (
    "Simulation : durée de trajet = durée en circulation fluide × (1 + occupation / 50). "
    "Relation illustrative, non calibrée sur des temps de parcours mesurés. "
    "Coût = temps total × coût horaire + temps de trajet × coût carburant horaire. "
    "Les économies sont des estimations à hypothèses constantes."
)


def traffic_risk(occupation: float) -> str:
    """The source traffic thresholds, applied to a sensor occupancy percentage."""
    if not np.isfinite(occupation):
        return "Inconnu"
    if occupation < 15:
        return "Fluide"
    if occupation < 30:
        return "Pré-saturé"
    if occupation < 50:
        return "Saturé"
    return "Bloqué"


def _forecast(forecast: pd.DataFrame) -> pd.DataFrame:
    needed = {"axe", "datetime", "occupation", "debit"}
    if not needed.issubset(forecast.columns):
        raise ValueError("Prévisions incomplètes : axe, datetime, occupation et debit sont requis.")
    result = forecast[list(needed)].copy()
    result["datetime"] = pd.to_datetime(result["datetime"], utc=True).dt.tz_convert("Europe/Paris")
    if result["datetime"].isna().any():
        raise ValueError("Les prévisions contiennent une date invalide.")
    if result["datetime"].dt.date.nunique() > 1:
        raise ValueError("Sélectionner une seule journée de prévision.")
    for column in ("occupation", "debit"):
        result[column] = pd.to_numeric(result[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    result["occupation"] = result["occupation"].clip(0, 100)
    result["debit"] = result["debit"].clip(lower=0)
    result["hour"] = result["datetime"].dt.hour
    return result


def rank_slots(
    forecast: pd.DataFrame,
    axes: list[str],
    start_hour: int = 7,
    end_hour: int = 20,
    window_hours: int = 2,
) -> pd.DataFrame:
    """Rank complete hourly windows; the smallest score is the best.

    score = 85% × occupancy + 15% × hourly flow as a percentage of each
    road's maximum daily flow. This operational preference is not a speed model.
    Every selected road must have both measurements at every hour in a window.
    """
    if not 0 <= start_hour < end_hour <= 24:
        raise ValueError("La plage horaire doit vérifier 0 ≤ début < fin ≤ 24.")
    if not isinstance(window_hours, (int, np.integer)) or window_hours < 1:
        raise ValueError("La durée du créneau doit être un nombre entier d'heures positif.")
    axes = list(dict.fromkeys(axes))
    if not axes:
        return pd.DataFrame(columns=SLOT_COLUMNS)
    frame = _forecast(forecast)
    frame = frame[frame["axe"].isin(axes)].dropna(subset=["occupation", "debit"])
    if frame.empty or set(axes) - set(frame["axe"]):
        return pd.DataFrame(columns=SLOT_COLUMNS)
    hourly = frame.groupby(["axe", "hour"], as_index=False)[["occupation", "debit"]].mean()
    maximum = hourly.groupby("axe")["debit"].transform("max")
    hourly["flow_ratio"] = hourly["debit"].div(maximum.replace(0, np.nan)).fillna(0)
    rows = []
    for hour in range(start_hour, end_hour - window_hours + 1):
        window = hourly[hourly["hour"].between(hour, hour + window_hours - 1)]
        if len(window) != len(axes) * window_hours:
            continue
        occupation = float(window["occupation"].mean())
        score = 0.85 * occupation + 15 * float(window["flow_ratio"].mean())
        rows.append({
            "start_hour": hour,
            "end_hour": hour + window_hours,
            "occupation": occupation,
            "debit": float(window["debit"].mean()),
            "score": float(np.clip(score, 0, 100)),
            "risk": traffic_risk(occupation),
            "label": f"{hour:02d}:00 – {hour + window_hours:02d}:00",
        })
    return pd.DataFrame(rows, columns=SLOT_COLUMNS).sort_values(
        ["score", "start_hour"], ignore_index=True,
    )


def _occupation_at(frame: pd.DataFrame, axes: list[str], hour: int) -> float:
    if not 0 <= hour <= 23:
        raise ValueError("L'heure de départ doit être comprise entre 0 et 23.")
    selected = frame[frame["axe"].isin(axes) & frame["hour"].eq(hour)]
    grouped = selected.groupby("axe")["occupation"].mean().reindex(axes)
    if grouped.isna().any():
        raise ValueError("Prévision d'occupation manquante pour ce départ ou cet axe.")
    return float(grouped.mean())


def estimate_impact(
    forecast: pd.DataFrame,
    axes: list[str],
    baseline_hour: int,
    optimized_hour: int,
    deliveries: int = 30,
    free_flow_minutes: float = 20,
    service_minutes: float = 8,
    hourly_cost: float = 32,
    fuel_cost_per_hour: float = 4,
    working_days: int = 220,
) -> dict:
    """Compare two departure hours using the explicit illustrative assumption.

    Each delivery has one travel leg and one service period. The annual figure
    repeats this exact daily scenario for ``working_days``; no fleet reduction
    or forecast model performance is implied. Negative savings are retained.
    """
    axes = list(dict.fromkeys(axes))
    if not axes:
        raise ValueError("Sélectionner au moins un axe.")
    numbers = [deliveries, free_flow_minutes, service_minutes, hourly_cost, fuel_cost_per_hour, working_days]
    if any(not np.isfinite(value) or value < 0 for value in numbers):
        raise ValueError("Les hypothèses doivent être des nombres positifs ou nuls.")
    frame = _forecast(forecast)
    baseline_occ = _occupation_at(frame, axes, baseline_hour)
    optimized_occ = _occupation_at(frame, axes, optimized_hour)
    baseline_travel = free_flow_minutes * (1 + baseline_occ / 50)
    optimized_travel = free_flow_minutes * (1 + optimized_occ / 50)
    baseline_minutes = baseline_travel + service_minutes
    optimized_minutes = optimized_travel + service_minutes
    baseline_cost = deliveries * (baseline_minutes * hourly_cost + baseline_travel * fuel_cost_per_hour) / 60
    optimized_cost = deliveries * (optimized_minutes * hourly_cost + optimized_travel * fuel_cost_per_hour) / 60
    time_saved = deliveries * (baseline_minutes - optimized_minutes)
    daily_savings = baseline_cost - optimized_cost
    return {
        "baseline_occupation": baseline_occ,
        "optimized_occupation": optimized_occ,
        "baseline_minutes_per_delivery": baseline_minutes,
        "optimized_minutes_per_delivery": optimized_minutes,
        "baseline_travel_minutes": baseline_travel,
        "optimized_travel_minutes": optimized_travel,
        "time_saved_minutes": time_saved,
        "time_saved_hours": time_saved / 60,
        "daily_savings": daily_savings,
        "annual_savings": daily_savings * working_days,
        "savings_pct": daily_savings / baseline_cost * 100 if baseline_cost else 0.0,
        "baseline_daily_cost": baseline_cost,
        "optimized_daily_cost": optimized_cost,
        "assumption": ASSUMPTION,
        "deliveries": deliveries,
        "working_days": working_days,
    }


def _empty_schedule() -> pd.DataFrame:
    return pd.DataFrame(columns=SCHEDULE_COLUMNS)


def _civil_boundary(day: pd.Timestamp, hour: float, *, latest: bool = False) -> pd.Timestamp:
    """Interpret an hour as Paris wall time, not elapsed hours since midnight.

    A repeated start uses its first occurrence; a repeated deadline uses its
    last occurrence. A nonexistent spring start advances to the next valid
    instant, while a nonexistent deadline retreats to the preceding instant.
    """
    civil = day.tz_localize(None).normalize() + pd.Timedelta(hours=hour)
    return civil.tz_localize(
        "Europe/Paris", ambiguous=not latest,
        nonexistent="shift_backward" if latest else "shift_forward",
    )


def build_schedule(
    forecast: pd.DataFrame,
    requests: pd.DataFrame,
    fleet_size: int = 3,
    start_hour: int = 7,
    end_hour: int = 20,
    base_travel_minutes: float = 20,
) -> pd.DataFrame:
    """Assign deliveries to vehicles and 15-minute departures without overlap.

    ``latest_hour`` is a completion deadline, including the estimated travel leg
    and service. The heuristic handles the tightest windows first and chooses
    the least occupied feasible departure. This is not route optimisation:
    each request assumes an independent travel leg of ``base_travel_minutes``.
    Windows use Paris civil time, including the 23/25-hour DST transition days.
    """
    if not isinstance(fleet_size, (int, np.integer)) or fleet_size < 1:
        raise ValueError("Le nombre de véhicules doit être un entier positif.")
    if not 0 <= start_hour < end_hour <= 24:
        raise ValueError("La plage horaire doit vérifier 0 ≤ début < fin ≤ 24.")
    if not np.isfinite(base_travel_minutes) or base_travel_minutes < 0:
        raise ValueError("La durée de trajet doit être positive ou nulle.")
    if requests.empty:
        return _empty_schedule()
    required = {"axe", "earliest_hour", "latest_hour", "service_minutes"}
    if not required.issubset(requests.columns):
        raise ValueError("Chaque demande doit préciser axe, earliest_hour, latest_hour et service_minutes.")
    frame = _forecast(forecast)
    if frame.empty:
        raise ValueError("Aucune prévision disponible pour planifier les livraisons.")
    day = frame["datetime"].iloc[0].normalize()
    # Keep distinct instants for the two occurrences of a repeated autumn hour.
    occupancy = frame.groupby(["axe", "datetime"])["occupation"].mean()
    requests = requests.copy().reset_index(drop=True)
    records = {}
    jobs = []
    for index, row in requests.iterrows():
        record = {
            "request_id": f"L{index + 1:03d}", "axe": str(row["axe"]),
            "start": pd.NaT, "end": pd.NaT, "vehicle": "", "occupation": np.nan,
            "status": "Non planifiée", "reason": "", "travel_minutes": np.nan,
            "service_minutes": row["service_minutes"],
        }
        records[index] = record
        try:
            earliest = float(row["earliest_hour"])
            latest = float(row["latest_hour"])
            service = float(row["service_minutes"])
        except (ValueError, TypeError):
            record["reason"] = "Heures ou durée de service invalides."
            continue
        if not all(math.isfinite(x) for x in (earliest, latest, service)) or not 0 <= earliest < latest <= 24 or service < 0:
            record["reason"] = "Vérifier la fenêtre horaire (0–24 h) et la durée de service."
            continue
        if str(row["axe"]) not in set(frame["axe"]):
            record["reason"] = "Aucune prévision pour cet axe."
            continue
        earliest, latest = max(start_hour, earliest), min(end_hour, latest)
        if earliest >= latest:
            record["reason"] = "La fenêtre demandée est hors des heures d'exploitation."
            continue
        jobs.append((latest - earliest, latest, index, earliest, service))

    occupied = {vehicle: [] for vehicle in range(1, fleet_size + 1)}
    for _, latest, index, earliest, service in sorted(jobs):
        record = records[index]
        candidates = []
        first_quarter = math.ceil(earliest * 4)
        first_start = _civil_boundary(day, first_quarter / 4)
        deadline = _civil_boundary(day, latest, latest=True)
        # Iterate actual instants: nonexistent spring hours are skipped and both
        # occurrences of repeated autumn hours retain their own traffic values.
        for start in pd.date_range(first_start, deadline, freq="15min"):
            forecast_hour = start.tz_convert("UTC").floor("h").tz_convert("Europe/Paris")
            occupation = occupancy.get((record["axe"], forecast_hour), np.nan)
            if not np.isfinite(occupation):
                continue
            travel = base_travel_minutes * (1 + occupation / 50)
            # Round up to whole minutes so a calendar export never shortens a job.
            duration = max(1, math.ceil(travel + service))
            end = start + pd.Timedelta(minutes=duration)
            if end > deadline:
                continue
            for vehicle, intervals in occupied.items():
                if all(end <= old_start or start >= old_end for old_start, old_end in intervals):
                    candidates.append((occupation, start, vehicle, end, travel))
        if not candidates:
            record["reason"] = "Aucun créneau disponible : fenêtre trop courte, capacité atteinte ou prévision absente."
            continue
        occupation, start, vehicle, end, travel = min(candidates)
        occupied[vehicle].append((start, end))
        record.update(
            start=start, end=end, vehicle=f"V{vehicle:02d}", occupation=float(occupation),
            status="Planifiée", reason="Créneau disponible à faible occupation prévue.",
            travel_minutes=float(travel), service_minutes=service,
        )
    return pd.DataFrame([records[index] for index in range(len(requests))], columns=SCHEDULE_COLUMNS)


def _escape_ics(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n").replace(";", "\\;").replace(",", "\\,")


def _fold_ics(line: str) -> list[str]:
    """RFC 5545 folding at 75 octets without splitting a UTF-8 character."""
    result, current = [], ""
    for character in line:
        if len((current + character).encode("utf-8")) > 75:
            result.append(current)
            current = " " + character
        else:
            current += character
    result.append(current)
    return result


def export_calendar(schedule: pd.DataFrame) -> bytes:
    """Export planned deliveries as UTC VEVENTs, importable by calendar tools."""
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//FlowPilot//Planning Livraisons//FR", "CALSCALE:GREGORIAN", "METHOD:PUBLISH"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not schedule.empty:
        for _, row in schedule[schedule["status"].eq("Planifiée")].iterrows():
            start, end = pd.Timestamp(row["start"]), pd.Timestamp(row["end"])
            if start.tzinfo is None or end.tzinfo is None:
                raise ValueError("Les dates du planning doivent comporter un fuseau horaire.")
            if end <= start:
                raise ValueError("La fin d'une livraison doit suivre son début.")
            identity = f"{row['request_id']}|{row['axe']}|{start.isoformat()}|{row['vehicle']}"
            uid = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
            description = (
                f"Véhicule {row['vehicle']}. Occupation prévue : {float(row['occupation']):.1f} %. "
                "Temps de trajet indicatif, selon le scénario FlowPilot ; vérifier les conditions réelles."
            )
            lines.extend([
                "BEGIN:VEVENT", f"UID:{uid}@flowpilot.local", f"DTSTAMP:{stamp}",
                "DTSTART:" + start.tz_convert("UTC").strftime("%Y%m%dT%H%M%SZ"),
                "DTEND:" + end.tz_convert("UTC").strftime("%Y%m%dT%H%M%SZ"),
                "SUMMARY:" + _escape_ics(f"{row['request_id']} · Livraison {row['axe']}"),
                "LOCATION:" + _escape_ics(row["axe"]),
                "DESCRIPTION:" + _escape_ics(description), "END:VEVENT",
            ])
    lines.append("END:VCALENDAR")
    return ("\r\n".join(part for line in lines for part in _fold_ics(line)) + "\r\n").encode("utf-8")
