"""Focused test: optimize for Convention + Saints-Peres only (exclude Champs-Elysees).

Caches Timestral predictions per date, then sweeps LGB blend weights.
Usage: python tests/test_focus_2arcs.py --dates 30 --step 3 --cible debit
"""
import sys, argparse, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
from dataset_create import charger, split, ORIGINES, HORIZON, feature_engineering

print("Loading data...", flush=True)
df = charger(imputer=True)

print("Loading Timestral...", flush=True)
from timestral import TimestralForecaster
pipe = TimestralForecaster.from_pretrained(
    "mistralai/Timestral-209M",
    device="cuda" if torch.cuda.is_available() else "cpu",
)

FOCUS_ARCS = ["Convention", "Saints-Peres"]  # ignore Champs-Elysees

COV_BASE_NUM = ["ferie", "pont", "paris_respire", "evenement", "confinement"]
COV_CAT = ["vacances_code", "type_jour"]
COV_CYCL_NUM = ["heure_sin", "heure_cos", "jour_sin", "jour_cos"]
CTX, H = 2048, 24 + HORIZON

LGB_PARAMS = {
    "debit": dict(n_estimators=1000, learning_rate=0.05, num_leaves=127, objective="huber", alpha=100, verbose=-1, random_state=0),
    "occupation": dict(n_estimators=500, learning_rate=0.05, num_leaves=31, min_child_samples=20, verbose=-1, random_state=0),
}


def make_g(fe, with_cyclical):
    vacances_code = fe.vacances.cat.codes.astype(float).where(fe.vacances.cat.codes >= 0)
    cov_num = COV_BASE_NUM + (COV_CYCL_NUM if with_cyclical else [])
    g = fe.assign(
        item_id=fe.axe.astype(str),
        timestamp=fe.datetime.dt.tz_convert(None),
        vacances_code=vacances_code,
        type_jour=fe.type_jour.astype(float),
        heure_sin=np.sin(2 * np.pi * fe.datetime.dt.hour / 24),
        heure_cos=np.cos(2 * np.pi * fe.datetime.dt.hour / 24),
        jour_sin=np.sin(2 * np.pi * fe.datetime.dt.dayofweek / 7),
        jour_cos=np.cos(2 * np.pi * fe.datetime.dt.dayofweek / 7),
    )
    return g.astype({c: np.float32 for c in cov_num + COV_CAT}), cov_num


def timestral_pred(X_train, X_test, cible, fe_full, cov_num, ctx=CTX):
    t0 = fe_full.loc[X_train.index, "datetime"].max().tz_convert(None) + pd.Timedelta("1h")
    heures = pd.date_range(t0 - pd.Timedelta(hours=ctx), periods=ctx + H, freq="h")
    g, _ = make_g(fe_full, with_cyclical=True)
    # use requested cov_num
    g2 = fe_full.assign(
        item_id=fe_full.axe.astype(str), timestamp=fe_full.datetime.dt.tz_convert(None),
        vacances_code=fe_full.vacances.cat.codes.astype(float).where(fe_full.vacances.cat.codes >= 0),
        type_jour=fe_full.type_jour.astype(float),
        heure_sin=np.sin(2 * np.pi * fe_full.datetime.dt.hour / 24),
        heure_cos=np.cos(2 * np.pi * fe_full.datetime.dt.hour / 24),
        jour_sin=np.sin(2 * np.pi * fe_full.datetime.dt.dayofweek / 7),
        jour_cos=np.cos(2 * np.pi * fe_full.datetime.dt.dayofweek / 7),
    ).astype({c: np.float32 for c in cov_num + COV_CAT})

    predictions = []
    for axe in X_test.axe.astype(str).unique():
        serie = g2.loc[g2.item_id == axe].set_index("timestamp").reindex(heures)
        contexte = serie[cible].to_numpy(dtype=np.float32)[:ctx][None, None, :]
        conn = serie[cov_num].to_numpy(dtype=np.float32).T[None, :, :]
        cats = serie[COV_CAT].to_numpy(dtype=np.float32).T[None, :, :]
        forecast = pipe.forecast(contexte, prediction_length=H,
                                 known_covariates=conn, categorical_known_covariates=cats)
        predictions.append(pd.DataFrame({
            "item_id": axe, "timestamp": heures[ctx:],
            "prediction": forecast.median[0, 0].cpu().numpy(),
        }))
    pred = pd.concat(predictions, ignore_index=True)
    test = pd.DataFrame({"item_id": X_test.axe.astype(str),
                         "timestamp": fe_full.loc[X_test.index, "datetime"].dt.tz_convert(None)})
    return test.merge(pred, on=["item_id", "timestamp"], how="left",
                      validate="many_to_one").prediction.to_numpy()


def rmse_focus(y, p, axe):
    """RMSE only on FOCUS_ARCS."""
    mask = axe.astype(str).isin(FOCUS_ARCS)
    if mask.sum() == 0:
        return pd.Series(dtype=float)
    return ((y[mask] - p[mask]) ** 2).groupby(axe[mask], observed=True).mean() ** .5


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dates", type=int, default=30)
    parser.add_argument("--step", type=int, default=3)
    parser.add_argument("--cible", type=str, default="debit")
    args = parser.parse_args()

    dates = ORIGINES[::args.step][:args.dates]
    print(f"Testing {len(dates)} dates, focus on {FOCUS_ARCS}, target={args.cible}", flush=True)

    # Store per-date predictions
    records = []
    for t0_str in dates:
        t_start = time.time()
        X_train, X_test, y_train, y_test = split(df, t0_str, cible=args.cible)
        if X_test.empty:
            continue
        # Only keep focus arcs in test
        mask_focus = X_test.axe.astype(str).isin(FOCUS_ARCS)
        if mask_focus.sum() == 0:
            continue
        X_test_f = X_test[mask_focus]
        y_test_f = y_test[mask_focus]

        t0_ts = pd.Timestamp(t0_str, tz="Europe/Paris")
        fe_t0 = feature_engineering(df, args.cible, t0_ts)

        # Timestral cyclical
        p_cyc = timestral_pred(X_train, X_test_f, args.cible, fe_t0, COV_BASE_NUM + COV_CYCL_NUM)
        # Timestral baseline
        p_base = timestral_pred(X_train, X_test_f, args.cible, fe_t0, COV_BASE_NUM)
        # LightGBM
        p_lgb = lgb.LGBMRegressor(**LGB_PARAMS[args.cible]).fit(X_train, y_train).predict(X_test_f).clip(0)

        rec = {
            "t0": t0_str,
            "axe": X_test_f.axe.astype(str).values,
            "y": y_test_f.values,
            "p_cyc": p_cyc,
            "p_base": p_base,
            "p_lgb": p_lgb,
            "naif": X_test_f.lag_7j.values,
        }
        records.append(rec)
        print(f"  {t0_str}: {len(X_test_f)} rows, arcs={X_test_f.axe.astype(str).unique().tolist()} ({time.time()-t_start:.0f}s)", flush=True)

    # Aggregate
    print("\n=== RMSE on Convention + Saints-Peres only ===", flush=True)
    all_rows = []
    for rec in records:
        for i in range(len(rec["axe"])):
            all_rows.append({
                "t0": rec["t0"], "axe": rec["axe"][i], "y": rec["y"][i],
                "p_cyc": rec["p_cyc"][i], "p_base": rec["p_base"][i],
                "p_lgb": rec["p_lgb"][i], "naif": rec["naif"][i],
            })
    D = pd.DataFrame(all_rows)

    def eval_col(col):
        return ((D.y - D[col]) ** 2).groupby(D.axe).mean() ** .5

    configs = {
        "timestral_baseline": "p_base",
        "timestral_cyclical": "p_cyc",
        "lightgbm": "p_lgb",
        "naif_lag7j": "naif",
    }
    # blends of cyclical + lgb
    for w in [0.1, 0.2, 0.25, 0.3, 0.4, 0.5]:
        D[f"blend_{w}"] = w * D.p_lgb + (1 - w) * D.p_cyc
        configs[f"cyc+{w}*lgb"] = f"blend_{w}"

    print(f"\n{'config':20s} {'Convention':>10s} {'Saints-Peres':>12s} {'mean':>8s}")
    results = {}
    for name, col in configs.items():
        r = eval_col(col)
        mean_2 = r.mean()
        results[name] = {"Convention": round(r.get("Convention", np.nan), 1),
                         "Saints-Peres": round(r.get("Saints-Peres", np.nan), 1),
                         "mean": round(mean_2, 1)}
        print(f"{name:20s} {r.get('Convention', float('nan')):10.1f} {r.get('Saints-Peres', float('nan')):12.1f} {mean_2:8.1f}")

    # Per-date mean for best configs
    print("\n=== Per-date mean RMSE (2 arcs) for key configs ===")
    for name in ["timestral_baseline", "timestral_cyclical", "cyc+0.25*lgb", "cyc+0.1*lgb", "lightgbm"]:
        col = configs[name]
        per_date = D.assign(e2=(D.y - D[col])**2).groupby("t0").e2.mean() ** .5
        print(f"  {name:20s}: mean={per_date.mean():.1f}  std={per_date.std():.1f}  n={len(per_date)}")

    with open("focus_2arcs_results.json", "w") as f:
        json.dump(results, f, indent=2)
    D.to_csv("focus_2arcs_detail.csv", index=False)
    print("\nSaved focus_2arcs_results.json and focus_2arcs_detail.csv")
