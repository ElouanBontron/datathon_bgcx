"""Chronos-2 (zero-shot) : lit les 2048 dernières heures de chaque arc avant t0 et prolonge la série,
avec le calendrier connu à l'avance en covariables. Pas d'entraînement.
Lancer depuis la racine : uv run python model/modele_chronos.py  -> test final + model/output_chronos.csv"""
import pandas as pd
from chronos import Chronos2Pipeline
from commun import lancer

COV = ["ferie", "pont", "vacances", "paris_respire", "evenement", "confinement", "type_jour"]  # connus à l'avance
CTX = 2048  # heures d'historique lues
pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")


def predire(full, t0, cible):
    g = full.assign(item_id=full.axe.astype(str), vacances=full.vacances.cat.codes, timestamp=full.datetime.dt.tz_convert(None))
    g = g[["item_id", "timestamp", cible, *COV]].astype({c: float for c in COV})
    passe, futur = full.datetime.between(t0 - pd.Timedelta(hours=CTX), t0, inclusive="left"), full.datetime >= t0
    c = pipe.predict_df(g[passe], future_df=g[futur].drop(columns=cible), target=cible,
                        prediction_length=futur.sum() // full.axe.nunique(), quantile_levels=[0.5])
    p = g[futur][["item_id", "timestamp"]].merge(c[["item_id", "timestamp", "predictions"]], how="left").predictions
    return pd.Series(p.values, full.index[futur])


if __name__ == "__main__":
    lancer(predire, "chronos")
