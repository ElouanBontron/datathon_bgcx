"""Compare Timestral+LGB vs Timestral+XGB blends, same protocol as focus test.
Caches Timestral per date, sweeps weights for both backends.
Focus Convention+Saints-Peres, 25 dates.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
import xgboost as xgb
from dataset_create import charger, split, ORIGINES, HORIZON, feature_engineering

print("Loading...", flush=True)
df = charger(imputer=True)
from timestral import TimestralForecaster
pipe = TimestralForecaster.from_pretrained("mistralai/Timestral-209M", device="cpu")

FOCUS = ["Convention", "Saints-Peres"]
COV_NUM = ["ferie", "pont", "paris_respire", "evenement", "confinement",
           "heure_sin", "heure_cos", "jour_sin", "jour_cos"]
COV_CAT = ["vacances_code", "type_jour"]
CTX, H = 2048, 24 + HORIZON

LGB_P = dict(n_estimators=1000, learning_rate=0.05, num_leaves=127, objective="huber", alpha=100, verbose=-1, random_state=0)
XGB_P = dict(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8,
             colsample_bytree=0.8, objective="reg:squarederror", random_state=42, n_jobs=-1)
CAT = ["axe", "vacances"]

fe = feature_engineering(df)
vc = fe.vacances.cat.codes.astype(float).where(fe.vacances.cat.codes >= 0)
g = fe.assign(item_id=fe.axe.astype(str), timestamp=fe.datetime.dt.tz_convert(None),
    vacances_code=vc, type_jour=fe.type_jour.astype(float),
    heure_sin=np.sin(2*np.pi*fe.datetime.dt.hour/24), heure_cos=np.cos(2*np.pi*fe.datetime.dt.hour/24),
    jour_sin=np.sin(2*np.pi*fe.datetime.dt.dayofweek/7), jour_cos=np.cos(2*np.pi*fe.datetime.dt.dayofweek/7),
).astype({c: np.float32 for c in COV_NUM + COV_CAT})

def ts_pred(X_train, X_test, cible="debit"):
    t0 = fe.loc[X_train.index, "datetime"].max().tz_convert(None) + pd.Timedelta("1h")
    heures = pd.date_range(t0 - pd.Timedelta(hours=CTX), periods=CTX + H, freq="h")
    preds = []
    for axe in X_test.axe.astype(str).unique():
        s = g.loc[g.item_id == axe].set_index("timestamp").reindex(heures)
        fc = pipe.forecast(s[cible].to_numpy(dtype=np.float32)[:CTX][None, None, :], prediction_length=H,
                           known_covariates=s[COV_NUM].to_numpy(dtype=np.float32).T[None, :, :],
                           categorical_known_covariates=s[COV_CAT].to_numpy(dtype=np.float32).T[None, :, :])
        preds.append(pd.DataFrame({"item_id": axe, "timestamp": heures[CTX:], "prediction": fc.median[0,0].cpu().numpy()}))
    pred = pd.concat(preds, ignore_index=True)
    test = pd.DataFrame({"item_id": X_test.axe.astype(str), "timestamp": fe.loc[X_test.index, "datetime"].dt.tz_convert(None)})
    return test.merge(pred, on=["item_id","timestamp"], how="left", validate="many_to_one").prediction.to_numpy()

def gbm(X_train, y_train, X_test, backend):
    if backend == "lgb":
        return lgb.LGBMRegressor(**LGB_P).fit(X_train, y_train).predict(X_test).clip(0)
    Xtr = pd.get_dummies(X_train.astype({c: str for c in CAT}), columns=CAT)
    Xte = pd.get_dummies(X_test.astype({c: str for c in CAT}), columns=CAT).reindex(columns=Xtr.columns, fill_value=0)
    return xgb.XGBRegressor(**XGB_P).fit(Xtr, y_train).predict(Xte).clip(0)

dates = ORIGINES[::4][:25]
print(f"{len(dates)} dates, focus={FOCUS}", flush=True)
rows = []
for t0_str in dates:
    t0t = time.time()
    X_train, X_test, y_train, y_test = split(df, t0_str, cible="debit")
    if X_test.empty: continue
    m = X_test.axe.astype(str).isin(FOCUS)
    if not m.any(): continue
    Xf, yf = X_test[m], y_test[m]
    p_ts = ts_pred(X_train, Xf)
    p_lgb = gbm(X_train, y_train, Xf, "lgb")
    p_xgb = gbm(X_train, y_train, Xf, "xgb")
    for i in range(len(Xf)):
        rows.append({"t0": t0_str, "axe": Xf.axe.iloc[i], "y": yf.iloc[i],
                     "ts": p_ts[i], "lgb": p_lgb[i], "xgb": p_xgb[i], "naif": Xf.lag_7j.iloc[i]})
    print(f"  {t0_str}: {len(Xf)} rows ({time.time()-t0t:.0f}s)", flush=True)

D = pd.DataFrame(rows)
print(f"\n=== {D.t0.nunique()} dates, {len(D)} rows, focus Convention+Saints-Peres ===\n")
print(f"{'config':16s} {'Convention':>10s} {'Saints-Peres':>12s} {'mean/date':>10s} {'std':>6s}")
configs = [("timestral", D.ts), ("lgb", D.lgb), ("xgb", D.xgb), ("naif", D.naif)]
for w in [0.25, 0.35, 0.5]:
    configs.append((f"ts+{w:.0%}lgb", w*D.lgb + (1-w)*D.ts))
    configs.append((f"ts+{w:.0%}xgb", w*D.xgb + (1-w)*D.ts))
for name, p in configs:
    r = ((D.y - p)**2).groupby(D.axe).mean() ** .5
    pd_mean = D.assign(e2=(D.y-p)**2).groupby("t0").e2.mean() ** .5
    print(f"{name:16s} {r.get('Convention',float('nan')):10.1f} {r.get('Saints-Peres',float('nan')):12.1f} {pd_mean.mean():10.1f} {pd_mean.std():6.1f}")
D.to_csv("blend_lgb_xgb_detail.csv", index=False)
