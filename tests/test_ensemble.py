"""Test ensemble: Timestral (best config) + LightGBM blend.

Run after test_timestral_variants.py to find the best timestral config.
Usage: python tests/test_ensemble.py --timestral-config all_cov --dates 10 --step 10
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
rmse = lambda y, p, axe: ((y - p) ** 2).groupby(axe, observed=True).mean() ** .5

print("Loading Timestral...", flush=True)
from timestral import TimestralForecaster
pipe = TimestralForecaster.from_pretrained(
    "mistralai/Timestral-209M",
    device="cuda" if torch.cuda.is_available() else "cpu",
)

COV_BASE_NUM = ["ferie", "pont", "paris_respire", "evenement", "confinement"]
COV_BASE_CAT = ["vacances_code", "type_jour"]
COV_EXTRA_NUM = ["veille_ferie", "lendemain_ferie", "pluie", "neige",
                 "profil_normal", "profil_ferie", "fermeture_prevue", "lag_an", "lag_moy"]
COV_CYCL_NUM = ["heure_sin", "heure_cos", "jour_sin", "jour_cos"]

CONFIGS = {
    "baseline": dict(cov_num=COV_BASE_NUM, cov_cat=COV_BASE_CAT, ctx=2048),
    "more_cov": dict(cov_num=COV_BASE_NUM + COV_EXTRA_NUM, cov_cat=COV_BASE_CAT, ctx=2048),
    "cyclical": dict(cov_num=COV_BASE_NUM + COV_CYCL_NUM, cov_cat=COV_BASE_CAT, ctx=2048),
    "all_cov":  dict(cov_num=COV_BASE_NUM + COV_EXTRA_NUM + COV_CYCL_NUM, cov_cat=COV_BASE_CAT, ctx=2048),
}

LGB_PARAMS = dict(n_estimators=1000, learning_rate=0.05, num_leaves=127,
                  objective="huber", alpha=100, verbose=-1, random_state=0)


def make_g(fe, cov_num, cov_cat):
    vacances_code = fe.vacances.cat.codes.astype(float).where(fe.vacances.cat.codes >= 0)
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
    all_cov = cov_num + cov_cat
    return g.astype({c: np.float32 for c in all_cov if c in g.columns})


def timestral_pred(X_train, X_test, cible, fe_t0, cfg):
    t0 = fe_t0.loc[X_train.index, "datetime"].max().tz_convert(None) + pd.Timedelta("1h")
    ctx, H = cfg["ctx"], 24 + HORIZON
    heures = pd.date_range(t0 - pd.Timedelta(hours=ctx), periods=ctx + H, freq="h")
    g = make_g(fe_t0, cfg["cov_num"], cfg["cov_cat"])
    predictions = []
    for axe in X_test.axe.astype(str).unique():
        serie = g.loc[g.item_id == axe].set_index("timestamp").reindex(heures)
        contexte = serie[cible].to_numpy(dtype=np.float32)[:ctx][None, None, :]
        conn = serie[cfg["cov_num"]].to_numpy(dtype=np.float32).T[None, :, :]
        cats = serie[cfg["cov_cat"]].to_numpy(dtype=np.float32).T[None, :, :]
        forecast = pipe.forecast(contexte, prediction_length=H,
                                 known_covariates=conn, categorical_known_covariates=cats)
        predictions.append(pd.DataFrame({
            "item_id": axe, "timestamp": heures[ctx:],
            "prediction": forecast.median[0, 0].cpu().numpy(),
        }))
    pred = pd.concat(predictions, ignore_index=True)
    test = pd.DataFrame({"item_id": X_test.axe.astype(str),
                         "timestamp": fe_t0.loc[X_test.index, "datetime"].dt.tz_convert(None)})
    return test.merge(pred, on=["item_id", "timestamp"], how="left",
                      validate="many_to_one").prediction.to_numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestral-config", type=str, default="all_cov")
    parser.add_argument("--dates", type=int, default=10)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--cible", type=str, default="debit")
    args = parser.parse_args()

    dates = ORIGINES[::args.step][:args.dates]
    cfg = CONFIGS[args.timestral_config]
    print(f"Ensemble test: Timestral[{args.timestral_config}] + LightGBM, {len(dates)} dates", flush=True)

    results = []
    for t0_str in dates:
        t_start = time.time()
        X_train, X_test, y_train, y_test = split(df, t0_str, cible=args.cible)
        if X_test.empty:
            continue
        t0_ts = pd.Timestamp(t0_str, tz="Europe/Paris")
        fe_t0 = feature_engineering(df, args.cible, t0_ts)

        p_timestral = timestral_pred(X_train, X_test, args.cible, fe_t0, cfg)
        p_lgb = lgb.LGBMRegressor(**LGB_PARAMS).fit(X_train, y_train).predict(X_test).clip(0)

        row = {"t0": t0_str}
        for w in [0.0, 0.25, 0.5, 0.75, 1.0]:
            p = w * p_lgb + (1 - w) * p_timestral
            r = rmse(y_test, p, X_test.axe)
            row[f"w_lgb={w}"] = r
        row["naif"] = rmse(y_test, X_test.lag_7j, X_test.axe)
        results.append(pd.DataFrame(row))
        print(f"  {t0_str}: " + "  ".join(f"w={k.split('=')[1]}:{v.mean():.1f}" for k, v in row.items() if k.startswith('w_')) + f"  ({time.time()-t_start:.0f}s)", flush=True)

    df_res = pd.concat(results)
    print("\n=== Per-arc mean RMSE by blend weight ===")
    for c in df_res.columns:
        if c.startswith("w_") or c == "naif":
            print(f"\n{c}:")
            print(df_res[c].groupby(level=0).mean().round(1))
            print(f"  MEAN: {df_res[c].mean().mean():.1f}")

    # Save
    df_res.to_csv("ensemble_results.csv")
    print("\nSaved to ensemble_results.csv")
