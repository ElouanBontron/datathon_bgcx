"""Timestral-209M de Mistral (zero-shot), repris du notebook de Lucie (notebooks/timestral.ipynb) :
pour chaque arc, lit les 2048 dernières heures avant t0 et le calendrier connu jusqu'à la fin de l'horizon.
Lancer depuis la racine : uv run python model/modele_timestral.py  -> test final + model/output_timestral.csv"""
import numpy as np
import pandas as pd
import torch
from commun import lancer
from timestral import TimestralForecaster

pipe = TimestralForecaster.from_pretrained("mistralai/Timestral-209M", device="cuda" if torch.cuda.is_available() else "cpu")
COV_NUM = ["ferie", "pont", "paris_respire", "evenement", "confinement"]
COV_CAT = ["vacances_code", "type_jour"]
CTX = 2048


def predire(full, t0, cible):
    codes = full.vacances.cat.codes
    g = full.assign(vacances_code=codes.astype(float).where(codes >= 0), type_jour=full.type_jour.astype(float),
                    timestamp=full.datetime.dt.tz_convert(None)).astype({c: np.float32 for c in COV_NUM + COV_CAT})
    debut = (t0 - pd.Timedelta(hours=CTX)).tz_convert(None)
    out = []
    for _, s in g.groupby("axe", observed=True):
        futur = s.datetime >= t0
        heures = pd.date_range(debut, periods=CTX + futur.sum(), freq="h")  # grille UTC régulière : contexte + horizon
        serie = s.set_index("timestamp").reindex(heures)
        forecast = pipe.forecast(serie[cible].to_numpy(np.float32)[:CTX][None, None, :], prediction_length=int(futur.sum()),
                                 known_covariates=serie[COV_NUM].to_numpy(np.float32).T[None],
                                 categorical_known_covariates=serie[COV_CAT].to_numpy(np.float32).T[None])
        mediane = pd.Series(forecast.median[0, 0].cpu().numpy(), heures[CTX:])
        out.append(pd.Series(mediane.reindex(s.timestamp[futur]).values, s.index[futur]))
    return pd.concat(out)


if __name__ == "__main__":
    lancer(predire, "timestral")
