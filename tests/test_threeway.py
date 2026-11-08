"""3-way ensemble: Timestral + LightGBM + XGBoost.
Grid search weights, temporal holdout (select 2024-25, test 2026).
Focus Convention+Saints-Peres, also report all-3-arcs.
"""
import sys, time, itertools
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

dates = ORIGINES[::3]  # ~35 dates
print(f"{len(dates)} dates", flush=True)
rows = []
for t0_str in dates:
    t0t = time.time()
    X_train, X_test, y_train, y_test = split(df, t0_str, cible="debit")
    if X_test.empty: continue
    year = int(t0_str[:4])
    # Timestral once; LGB+XGB on focus subset AND full test (for all-arcs eval)
    p_ts_full = ts_pred(X_train, X_test)
    p_lgb_full = gbm(X_train, y_train, X_test, "lgb")
    p_xgb_full = gbm(X_train, y_train, X_test, "xgb")
    for i in range(len(X_test)):
        rows.append({"t0": t0_str, "year": year, "axe": X_test.axe.iloc[i], "y": y_test.iloc[i],
                     "ts": p_ts_full[i], "lgb": p_lgb_full[i], "xgb": p_xgb_full[i],
                     "naif": X_test.lag_7j.iloc[i]})
    print(f"  {t0_str} y={year}: {len(X_test)} rows ({time.time()-t0t:.0f}s)", flush=True)

D = pd.DataFrame(rows)
D.to_csv("threeway_detail.csv", index=False)
print(f"\nSaved {len(D)} rows, {D.t0.nunique()} dates")

# Weight grid: w_ts from 0.5 to 0.9, split rest between lgb/xgb
WEIGHTS = []
for w_ts in [0.5, 0.6, 0.65, 0.7, 0.75, 0.8]:
    for w_lgb in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        w_xgb = 1 - w_ts - w_lgb
        if w_xgb < -0.001: continue
        WEIGHTS.append((w_ts, max(w_lgb, 0), max(w_xgb, 0)))

def evaluate(Dsub, label):
    focus = Dsub[Dsub.axe.isin(FOCUS)]
    print(f"\n=== {label} | {Dsub.t0.nunique()} dates | focus 2 arcs: {len(focus)} rows ===")
    results = []
    for w_ts, w_lgb, w_xgb in WEIGHTS:
        p = w_ts*focus.ts + w_lgb*focus.lgb + w_xgb*focus.xgb
        per_date = focus.assign(e2=(focus.y-p)**2).groupby("t0").e2.mean() ** .5
        r_arc = ((focus.y-p)**2).groupby(focus.axe).mean() ** .5
        results.append({"w_ts": w_ts, "w_lgb": w_lgb, "w_xgb": w_xgb,
                        "mean_date": per_date.mean(), "std": per_date.std(),
                        "conv": r_arc.get("Convention", np.nan), "sp": r_arc.get("Saints-Peres", np.nan)})
    R = pd.DataFrame(results).sort_values("mean_date")
    print(R.head(12).to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    return R

R_sel = evaluate(D[D.year < 2026], "SELECTION 2024-2025")
R_test = evaluate(D[D.year >= 2026], "HOLDOUT 2026")
R_all = evaluate(D, "ALL")

# Best on selection, evaluated on holdout
best_sel = R_sel.iloc[0]
print(f"\n>>> Best on selection: ts={best_sel.w_ts:.2f} lgb={best_sel.w_lgb:.2f} xgb={best_sel.w_xgb:.2f} (sel={best_sel.mean_date:.1f})")
mask = (R_test.w_ts==best_sel.w_ts)&(R_test.w_lgb==best_sel.w_lgb)&(R_test.w_xgb==best_sel.w_xgb)
print(f">>> Same weights on holdout 2026: {R_test[mask].mean_date.values[0]:.1f}" if mask.any() else "n/a")
# Also: 2-way best for comparison on holdout
print(f"\n>>> Reference on holdout: ts alone, and best 2-way ts+lgb")
for name, p in [("ts", D[D.year>=2026].ts), ("ts+0.35lgb", 0.65*D[D.year>=2026].ts + 0.35*D[D.year>=2026].lgb)]:
    f = D[(D.year>=2026) & D.axe.isin(FOCUS)]
    pp = p.loc[f.index]
    pd_m = f.assign(e2=(f.y-pp)**2).groupby("t0").e2.mean() ** .5
    print(f"    {name}: {pd_m.mean():.1f}")
