"""Fair comparison: XGBoost vs LightGBM vs Timestral+blend, same protocol.
38 dates, focus Convention+Saints-Peres, per-date mean RMSE.
Also reports the single-date global RMSE (lucie/xgboost style) for reference.
"""
import sys, argparse, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from dataset_create import charger, split, ORIGINES, FEATURES

print("Loading data...", flush=True)
df = charger(imputer=True)

FOCUS = ["Convention", "Saints-Peres"]

XGB_PARAMS = dict(n_estimators=500, max_depth=6, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror",
                  random_state=42, n_jobs=-1)
LGB_PARAMS = dict(n_estimators=1000, learning_rate=0.05, num_leaves=127,
                  objective="huber", alpha=100, verbose=-1, random_state=0)

# Categorical handling for XGBoost (one-hot like lucie's notebook)
CAT = ["axe", "vacances"]

def prep(X):
    X = X.copy()
    for c in CAT:
        if c in X:
            X[c] = X[c].astype(str)
    return pd.get_dummies(X, columns=CAT, drop_first=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dates", type=int, default=40)
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--cible", type=str, default="debit")
    args = parser.parse_args()

    dates = ORIGINES[::args.step][:args.dates]
    print(f"{len(dates)} dates, focus={FOCUS}, cible={args.cible}", flush=True)

    rows = []
    single_date_global = {}  # lucie-style: global RMSE on one date

    for t0_str in dates:
        t0t = time.time()
        X_train, X_test, y_train, y_test = split(df, t0_str, cible=args.cible)
        if X_test.empty:
            continue
        mask = X_test.axe.astype(str).isin(FOCUS)
        if not mask.any():
            continue
        Xf, yf = X_test[mask], y_test[mask]

        # LightGBM (native categoricals)
        p_lgb = lgb.LGBMRegressor(**LGB_PARAMS).fit(X_train, y_train).predict(Xf).clip(0)

        # XGBoost (one-hot like lucie)
        Xtr_d = prep(X_train)
        Xte_d = prep(Xf).reindex(columns=Xtr_d.columns, fill_value=0)
        p_xgb = xgb.XGBRegressor(**XGB_PARAMS).fit(Xtr_d, y_train).predict(Xte_d).clip(0)

        # naif
        p_naif = Xf.lag_7j.values

        # lucie-style global RMSE on this date (all arcs in X_test, not just focus)
        if t0_str == ORIGINES[-3]:  # 2026-10-16, same as lucie's BACKTEST_ORIGIN
            Xa, ya = X_test, y_test
            pa_xgb = xgb.XGBRegressor(**XGB_PARAMS).fit(prep(X_train), y_train).predict(prep(Xa).reindex(columns=prep(X_train).columns, fill_value=0)).clip(0)
            pa_lgb = lgb.LGBMRegressor(**LGB_PARAMS).fit(X_train, y_train).predict(Xa).clip(0)
            single_date_global['xgb_global'] = float(np.sqrt(((ya - pa_xgb) ** 2).mean()))
            single_date_global['lgb_global'] = float(np.sqrt(((ya - pa_lgb) ** 2).mean()))
            single_date_global['naif_global'] = float(np.sqrt(((ya - Xa.lag_7j) ** 2).mean()))
            print(f"  [lucie-style {t0_str}] xgb_global={single_date_global['xgb_global']:.1f} "
                  f"lgb_global={single_date_global['lgb_global']:.1f} naif_global={single_date_global['naif_global']:.1f}", flush=True)

        for i in range(len(Xf)):
            rows.append({"t0": t0_str, "axe": Xf.axe.iloc[i], "y": yf.iloc[i],
                         "lgb": p_lgb[i], "xgb": p_xgb[i], "naif": p_naif[i]})
        print(f"  {t0_str}: {len(Xf)} rows ({time.time()-t0t:.0f}s)", flush=True)

    D = pd.DataFrame(rows)
    print(f"\n=== Fair comparison: {D.t0.nunique()} dates, focus={FOCUS}, cible={args.cible} ===")
    print(f"\n{'model':12s} {'Convention':>10s} {'Saints-Peres':>12s} {'per-date mean':>14s} {'std':>6s}")
    for col in ["lgb", "xgb", "naif"]:
        r_arc = ((D.y - D[col]) ** 2).groupby(D.axe).mean() ** .5
        per_date = D.assign(e2=(D.y - D[col]) ** 2).groupby("t0").e2.mean() ** .5
        print(f"{col:12s} {r_arc.get('Convention', float('nan')):10.1f} {r_arc.get('Saints-Peres', float('nan')):12.1f} "
              f"{per_date.mean():14.1f} {per_date.std():6.1f}")

    if single_date_global:
        print(f"\n=== Lucie-style single-date global RMSE ({ORIGINES[-3]}, all arcs) ===")
        for k, v in single_date_global.items():
            print(f"  {k}: {v:.1f}")

    D.to_csv("xgb_vs_lgb_detail.csv", index=False)
