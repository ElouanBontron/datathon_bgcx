"""Temporal holdout: select hyperparams on 2024-2025, evaluate on 2026 (unseen).

Caches Timestral predictions once per date, then evaluates all configs
on both the selection set (pre-2026) and the holdout (2026+).
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
from dataset_create import charger, split, ORIGINES, HORIZON, feature_engineering

print("Loading...", flush=True)
df = charger(imputer=True)
from timestral import TimestralForecaster
pipe = TimestralForecaster.from_pretrained("mistralai/Timestral-209M", device="cpu")

FOCUS = ["Convention", "Saints-Peres"]
COV_NUM = ["ferie", "pont", "paris_respire", "evenement", "confinement",
           "heure_sin", "heure_cos", "jour_sin", "jour_cos"]
COV_BASE = ["ferie", "pont", "paris_respire", "evenement", "confinement"]
COV_CAT = ["vacances_code", "type_jour"]
CTX, H = 2048, 24 + HORIZON
LGB_P = dict(n_estimators=1000, learning_rate=0.05, num_leaves=127, objective="huber", alpha=100, verbose=-1, random_state=0)

fe = feature_engineering(df)
vc = fe.vacances.cat.codes.astype(float).where(fe.vacances.cat.codes >= 0)
base_assign = dict(item_id=fe.axe.astype(str), timestamp=fe.datetime.dt.tz_convert(None),
    vacances_code=vc, type_jour=fe.type_jour.astype(float),
    heure_sin=np.sin(2*np.pi*fe.datetime.dt.hour/24), heure_cos=np.cos(2*np.pi*fe.datetime.dt.hour/24),
    jour_sin=np.sin(2*np.pi*fe.datetime.dt.dayofweek/7), jour_cos=np.cos(2*np.pi*fe.datetime.dt.dayofweek/7))
g = fe.assign(**base_assign).astype({c: np.float32 for c in COV_NUM + COV_CAT})

def ts_pred(X_train, X_test, cov_num, cible="debit", ctx=CTX):
    t0 = fe.loc[X_train.index, "datetime"].max().tz_convert(None) + pd.Timedelta("1h")
    heures = pd.date_range(t0 - pd.Timedelta(hours=ctx), periods=ctx + H, freq="h")
    preds = []
    for axe in X_test.axe.astype(str).unique():
        s = g.loc[g.item_id == axe].set_index("timestamp").reindex(heures)
        fc = pipe.forecast(s[cible].to_numpy(dtype=np.float32)[:ctx][None, None, :], prediction_length=H,
                           known_covariates=s[cov_num].to_numpy(dtype=np.float32).T[None, :, :],
                           categorical_known_covariates=s[COV_CAT].to_numpy(dtype=np.float32).T[None, :, :])
        preds.append(pd.DataFrame({"item_id": axe, "timestamp": heures[ctx:], "prediction": fc.median[0,0].cpu().numpy()}))
    pred = pd.concat(preds, ignore_index=True)
    test = pd.DataFrame({"item_id": X_test.axe.astype(str), "timestamp": fe.loc[X_test.index, "datetime"].dt.tz_convert(None)})
    return test.merge(pred, on=["item_id","timestamp"], how="left", validate="many_to_one").prediction.to_numpy()

dates = ORIGINES[::3]  # ~35 dates spanning 2024-2026
print(f"{len(dates)} dates", flush=True)
rows = []
for t0_str in dates:
    t0t = time.time()
    X_train, X_test, y_train, y_test = split(df, t0_str, cible="debit")
    if X_test.empty: continue
    m = X_test.axe.astype(str).isin(FOCUS)
    if not m.any(): continue
    Xf, yf = X_test[m], y_test[m]
    year = int(t0_str[:4])
    p_cyc = ts_pred(X_train, Xf, COV_NUM)    # cyclical
    p_base = ts_pred(X_train, Xf, COV_BASE)  # baseline
    p_lgb = lgb.LGBMRegressor(**LGB_P).fit(X_train, y_train).predict(Xf).clip(0)
    for i in range(len(Xf)):
        rows.append({"t0": t0_str, "year": year, "axe": Xf.axe.iloc[i], "y": yf.iloc[i],
                     "ts_cyc": p_cyc[i], "ts_base": p_base[i], "lgb": p_lgb[i], "naif": Xf.lag_7j.iloc[i]})
    print(f"  {t0_str} (year={year}): {len(Xf)} rows ({time.time()-t0t:.0f}s)", flush=True)

D = pd.DataFrame(rows)
D.to_csv("holdout_detail.csv", index=False)
print(f"\nSaved {len(D)} rows, {D.t0.nunique()} dates")

def evaluate(Dsub, label):
    print(f"\n=== {label} ({Dsub.t0.nunique()} dates, {len(Dsub)} rows) ===")
    print(f"{'config':16s} {'Convention':>10s} {'Saints-Peres':>12s} {'mean/date':>10s} {'std':>6s}")
    configs = [("ts_baseline", Dsub.ts_base), ("ts_cyclical", Dsub.ts_cyc), ("lgb", Dsub.lgb), ("naif", Dsub.naif)]
    for w in [0.25, 0.35, 0.5]:
        configs.append((f"base+{w:.0%}lgb", w*Dsub.lgb + (1-w)*Dsub.ts_base))
        configs.append((f"cyc+{w:.0%}lgb", w*Dsub.lgb + (1-w)*Dsub.ts_cyc))
    for name, p in configs:
        r = ((Dsub.y - p)**2).groupby(Dsub.axe).mean() ** .5
        pd_m = Dsub.assign(e2=(Dsub.y-p)**2).groupby("t0").e2.mean() ** .5
        print(f"{name:16s} {r.get('Convention',float('nan')):10.1f} {r.get('Saints-Peres',float('nan')):12.1f} {pd_m.mean():10.1f} {pd_m.std():6.1f}")

# Selection set: 2024-2025 ; Holdout: 2026
evaluate(D[D.year < 2026], "SELECTION (2024-2025)")
evaluate(D[D.year >= 2026], "HOLDOUT (2026, unseen)")
evaluate(D, "ALL dates")
