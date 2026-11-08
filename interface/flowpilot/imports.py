"""Validate the team's BCG forecast CSV without manufacturing model metadata.

The four input columns are ``arc,datetime,debit_horaire,taux_occupation``.
Imported timestamps identify the START of the forecast hour in Paris: 09:00
means 09:00–10:00. They are not shifted like historical end-of-hour readings.

An imported file supplies point forecasts only. ``lower_occupation`` and
``upper_occupation`` both equal the point forecast; this is not an uncertainty
band or a calibrated confidence interval. Accuracy and sample counts are unknown.
"""

from __future__ import annotations

import csv
import io
import re
import unicodedata

import numpy as np
import pandas as pd

from .data import AXES, PARIS, is_holiday


REQUIRED_COLUMNS = ("arc", "datetime", "debit_horaire", "taux_occupation")
IMPORTED_METHOD = "Prévisions importées"


def _axis_key(value: str) -> str:
    folded = unicodedata.normalize("NFKD", str(value)).casefold()
    return re.sub(r"[^a-z0-9]", "", folded)


AXIS_ALIASES = {_axis_key(name): name for name in AXES}
AXIS_ALIASES.update({_axis_key(info["label"]): name for name, info in AXES.items()})
AXIS_ALIASES.update({"saintperes": "Saints-Peres", "saintpere": "Saints-Peres"})


def _read_csv(data: bytes) -> pd.DataFrame:
    if not isinstance(data, bytes):
        raise ValueError("Le fichier de prévisions doit être fourni sous forme d'octets CSV.")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("Le CSV doit être encodé en UTF-8 (BOM accepté).") from error
    if not text.strip():
        raise ValueError("Le fichier de prévisions est vide.")
    try:
        dialect = csv.Sniffer().sniff(text[:65_536], delimiters=",;")
        rows = [row for row in csv.reader(io.StringIO(text), dialect, strict=True)
                if any(cell.strip() for cell in row)]
    except csv.Error as error:
        raise ValueError("CSV illisible : utilisez une virgule ou un point-virgule comme séparateur.") from error
    columns = [name.strip().lower() for name in rows[0]]
    if len(columns) != len(set(columns)):
        raise ValueError("Le CSV contient des noms de colonnes en double.")
    missing = set(REQUIRED_COLUMNS) - set(columns)
    if missing:
        raise ValueError("Colonnes manquantes : " + ", ".join(sorted(missing)) + ".")
    if len(rows) == 1:
        raise ValueError("Le CSV contient un en-tête mais aucune prévision.")
    if any(len(row) != len(columns) for row in rows[1:]):
        raise ValueError("Le nombre de valeurs ne correspond pas à l'en-tête CSV.")
    return pd.DataFrame(rows[1:], columns=columns)


def _parse_datetimes(values: pd.Series) -> pd.DatetimeIndex:
    stamps = []
    awareness = set()
    for line, value in enumerate(values, start=2):
        value = str(value).strip()
        if not re.fullmatch(
            r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(?::\d{2}(?:\.\d{1,9})?)?(?:Z|[+-]\d{2}:?\d{2})?",
            value,
        ):
            raise ValueError(
                f"Date invalide à la ligne {line} : utilisez AAAA-MM-JJ HH:MM "
                "(décalage UTC ISO facultatif)."
            )
        try:
            stamp = pd.Timestamp(value)
            if pd.isna(stamp):
                raise ValueError("date vide")
        except (ValueError, TypeError, OverflowError) as error:
            raise ValueError(f"Date invalide à la ligne {line} : {value!r}.") from error
        if any((stamp.minute, stamp.second, stamp.microsecond, stamp.nanosecond)):
            raise ValueError(f"La date de la ligne {line} doit tomber sur une heure entière.")
        aware = stamp.tzinfo is not None
        awareness.add(aware)
        if len(awareness) > 1:
            raise ValueError("Ne mélangez pas des dates sans fuseau et des dates avec décalage horaire.")
        try:
            if aware:
                local = stamp.tz_convert(PARIS)
                if (stamp.utcoffset() != local.utcoffset() or
                        stamp.tz_localize(None) != local.tz_localize(None)):
                    raise ValueError("décalage incompatible avec Paris")
            else:
                local = stamp.tz_localize(PARIS, ambiguous="raise", nonexistent="raise")
        except Exception as error:
            # pandas uses different exception classes for invalid DST hours
            # depending on its timezone implementation.
            raise ValueError(
                f"Fuseau ou heure de Paris invalide à la ligne {line}. "
                "Précisez le bon décalage UTC pour une heure de changement d'heure."
            ) from error
        stamps.append(local)
    return pd.DatetimeIndex(stamps)


def parse_predictions(data: bytes) -> pd.DataFrame:
    """Return validated point forecasts in the ``get_forecast`` dataframe format.

    Both comma and semicolon delimiters, a UTF-8 BOM, French decimal commas
    inside CSV cells, accents and the challenge's street-name variants are
    accepted. Naive dates are Paris local time. Explicit UTC offsets must match
    Paris at that local date; mixing naive and timezone-aware dates is refused.

    This function validates records, not coverage of a requested day. The caller
    must check the selected day has every expected axis/hour before using it.
    """
    raw = _read_csv(data)
    axes = raw["arc"].map(lambda value: AXIS_ALIASES.get(_axis_key(value)))
    if axes.isna().any():
        unknown = raw.loc[axes.isna(), "arc"].drop_duplicates().tolist()
        raise ValueError("Axes inconnus : " + ", ".join(repr(name) for name in unknown) + ".")
    result = pd.DataFrame({"axe": axes, "datetime": _parse_datetimes(raw["datetime"])})
    for input_name, output_name, upper in (
        ("debit_horaire", "debit", 50_000),
        ("taux_occupation", "occupation", 100),
    ):
        values = pd.to_numeric(raw[input_name].str.strip().str.replace(",", ".", regex=False), errors="coerce")
        valid = np.isfinite(values.to_numpy(dtype=float)) & values.between(0, upper).to_numpy()
        if not valid.all():
            lines = ", ".join(str(index + 2) for index in np.flatnonzero(~valid)[:5])
            raise ValueError(
                f"{input_name} : valeurs finies comprises entre 0 et {upper} requises "
                f"(lignes {lines})."
            )
        result[output_name] = values.astype(float)
    if result.duplicated(["axe", "datetime"]).any():
        raise ValueError("Prévisions en double pour un même axe et une même heure.")
    result["lower_occupation"] = result["occupation"]
    result["upper_occupation"] = result["occupation"]
    result["method"] = IMPORTED_METHOD
    result["samples"] = 0
    holiday = {day: is_holiday(day) for day in result["datetime"].dt.date.unique()}
    result["is_holiday"] = result["datetime"].dt.date.map(holiday).astype(bool)
    result["last_observation"] = pd.Series(pd.NaT, index=result.index, dtype=f"datetime64[ns, {PARIS}]")
    result["stale_days"] = np.nan
    return result.sort_values(["axe", "datetime"]).reset_index(drop=True)
