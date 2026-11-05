"""Chronos-2 (zero-shot) : lit les 2048 dernières heures de chaque arc et prolonge la série, avec le calendrier en covariables.
1. Test   : historique jusqu'au 01/11 00h, on prédit le test final -> RMSE par arc.
2. Rendu  : historique test final compris, on prédit le 07/11 -> 11/11 -> model/output_chronos.csv.
Lancer depuis la racine : uv run python model/modele_chronos.py"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))  # pour importer dataset_create et rendu

import pandas as pd
from chronos import Chronos2Pipeline
from dataset_create import ajouter_futur, avec_recent, charger, vraies_valeurs
from rendu import FIN, exporter, rendu

COV = ["ferie", "pont", "vacances", "paris_respire", "evenement", "confinement", "type_jour"]  # connus à l'avance
CTX = 2048  # heures d'historique lues
pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")


def chronos(df, cible, fin):
    """Prédit toutes les heures de la fin de df jusqu'à fin -> (arc, datetime, pred)."""
    full, t0 = ajouter_futur(df, cible, fin)
    g = full.assign(item_id=full.axe.astype(str), vacances=full.vacances.cat.codes, timestamp=full.datetime.dt.tz_convert(None))
    g = g[["item_id", "timestamp", cible, *COV]].astype({c: float for c in COV})
    passe, futur = full.datetime.between(t0 - pd.Timedelta(hours=CTX), t0, inclusive="left"), full.datetime >= t0
    c = pipe.predict_df(g[passe], future_df=g[futur].drop(columns=cible), target=cible,
                        prediction_length=futur.sum() // full.axe.nunique(), quantile_levels=[0.5])
    p = g[futur][["item_id", "timestamp"]].merge(c[["item_id", "timestamp", "predictions"]], how="left").predictions
    return full.loc[futur, ["axe", "datetime"]].assign(pred=p.clip(0).values)


df = charger(imputer=True)
vrai, recent, preds = vraies_valeurs(df), avec_recent(df), {}
for cible in ["debit", "occupation"]:
    # 1. Test : historique jusqu'au 01/11 00h, prédit jusqu'à la dernière vraie valeur
    e = chronos(df, cible, vrai.datetime.max()).astype({"axe": str}).merge(vrai).dropna(subset=[cible])
    print(f"\nTest final {cible} :\n", (((e.pred - e[cible]) ** 2).groupby(e.axe).mean() ** .5).round(1))

    # 2. Rendu : historique jusqu'à la dernière vraie valeur, prédit les 360 heures du challenge
    p = chronos(recent, cible, FIN)
    preds[cible], cles = p.pred[rendu(p)], p[rendu(p)]

exporter(cles, preds["debit"], preds["occupation"], "chronos", Path(__file__).parent)
