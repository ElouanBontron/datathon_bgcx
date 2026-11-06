"""Prophet : un modèle par arc, entraîné sur les 2 dernières années avant t0.
Tendance + saisonnalités annuelle et hebdo + profil journalier différent semaine / week-end + calendrier en régresseurs.
Lancer depuis la racine : uv run python model/modele_prophet.py  -> test final + model/output_prophet.csv"""
import logging

import pandas as pd
from commun import lancer
from prophet import Prophet

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
REG = ["ferie", "pont", "paris_respire", "evenement", "confinement", "vacances"]  # connus à l'avance
ANS = 2  # années d'historique


def predire(full, t0, cible):
    out = []
    for _, g in full.groupby("axe", observed=True):
        d = pd.DataFrame({"ds": g.datetime.dt.tz_localize(None), "y": g[cible], "weekend": g.datetime.dt.dayofweek >= 5,
                          **{c: g[c].astype(float) for c in REG[:-1]}, "vacances": (g.vacances.astype(str) != "Aucune").astype(float)})
        d["semaine"] = ~d.weekend
        passe = (g.datetime < t0) & (g.datetime >= t0 - pd.DateOffset(years=ANS)) & d.y.notna()
        m = Prophet(daily_seasonality=False)
        m.add_seasonality("jour_semaine", period=1, fourier_order=10, condition_name="semaine")
        m.add_seasonality("jour_weekend", period=1, fourier_order=10, condition_name="weekend")
        for c in REG:
            if d.loc[passe, c].nunique() > 1:  # régresseur constant sur la période (ex. confinement) -> inutile
                m.add_regressor(c)
        m.fit(d[passe])
        futur = g.datetime >= t0
        out.append(pd.Series(m.predict(d[futur]).yhat.values, g.index[futur]))
    return pd.concat(out)


if __name__ == "__main__":
    lancer(predire, "prophet")
