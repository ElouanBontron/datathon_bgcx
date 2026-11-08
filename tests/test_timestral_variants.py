"""Compare Timestral variants on backtest to find the best config.

Run: python tests/test_timestral_variants.py [--dates N] [--configs baseline,more_cov,cyclical,all,ctx1024,ctx4096]
"""
import sys, os, argparse, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
import torch
from dataset_create import charger, split, ORIGINES, HORIZON, feature_engineering

# ---------------------------------------------------------------------------
# Load data once
# ---------------------------------------------------------------------------
print("Loading data...", flush=True)
df = charger(imputer=True)
rmse = lambda y, p, axe: ((y - p) ** 2).groupby(axe, observed=True).mean() ** .5

# ---------------------------------------------------------------------------
# Timestral pipeline
# ---------------------------------------------------------------------------
print("Loading Timestral...", flush=True)
from timestral import TimestralForecaster
pipe = TimestralForecaster.from_pretrained(
    "mistralai/Timestral-209M",
    device="cuda" if torch.cuda.is_available() else "cpu",
)
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}", flush=True)

# Covariate sets to test
COV_BASE_NUM = ["ferie", "pont", "paris_respire", "evenement", "confinement"]
COV_BASE_CAT = ["vacances_code", "type_jour"]

COV_EXTRA_NUM = ["veille_ferie", "lendemain_ferie", "pluie", "neige",
                 "profil_normal", "profil_ferie", "fermeture_prevue", "lag_an", "lag_moy"]
COV_CYCL_NUM = ["heure_sin", "heure_cos", "jour_sin", "jour_cos"]


def make_g(fe, cov_num, cov_cat):
    """Build the dataframe Timestral expects, with given covariate sets."""
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


def modele(X_train, y_train, X_test, cible, fe_full, cov_num, cov_cat, ctx):
    """Run Timestral for one split. fe_full must be feature_engineering WITH t0 masking."""
    t0 = fe_full.loc[X_train.index, "datetime"].max().tz_convert(None) + pd.Timedelta("1h")
    H = 24 + HORIZON
    debut = t0 - pd.Timedelta(hours=ctx)
    heures = pd.date_range(debut, periods=ctx + H, freq="h")
    g = make_g(fe_full, cov_num, cov_cat)
    predictions = []

    for axe in X_test.axe.astype(str).unique():
        serie = g.loc[g.item_id == axe].set_index("timestamp").reindex(heures)
        contexte = serie[cible].to_numpy(dtype=np.float32)[:ctx][None, None, :]
        conn = serie[cov_num].to_numpy(dtype=np.float32).T[None, :, :]
        cats = serie[cov_cat].to_numpy(dtype=np.float32).T[None, :, :] if cov_cat else None

        forecast = pipe.forecast(
            contexte,
            prediction_length=H,
            known_covariates=conn,
            categorical_known_covariates=cats,
        )
        mediane = forecast.median[0, 0].cpu().numpy()
        predictions.append(pd.DataFrame({
            "item_id": axe, "timestamp": heures[ctx:], "prediction": mediane,
        }))

    pred = pd.concat(predictions, ignore_index=True)
    test = pd.DataFrame({
        "item_id": X_test.axe.astype(str),
        "timestamp": fe_full.loc[X_test.index, "datetime"].dt.tz_convert(None),
    })
    return test.merge(pred, on=["item_id", "timestamp"], how="left", validate="many_to_one").prediction.to_numpy()


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------
CONFIGS = {
    "baseline":   dict(cov_num=COV_BASE_NUM, cov_cat=COV_BASE_CAT, ctx=2048),
    "more_cov":   dict(cov_num=COV_BASE_NUM + COV_EXTRA_NUM, cov_cat=COV_BASE_CAT, ctx=2048),
    "cyclical":   dict(cov_num=COV_BASE_NUM + COV_CYCL_NUM, cov_cat=COV_BASE_CAT, ctx=2048),
    "all_cov":    dict(cov_num=COV_BASE_NUM + COV_EXTRA_NUM + COV_CYCL_NUM,
                       cov_cat=COV_BASE_CAT, ctx=2048),
    "ctx1024":    dict(cov_num=COV_BASE_NUM + COV_EXTRA_NUM + COV_CYCL_NUM,
                       cov_cat=COV_BASE_CAT, ctx=1024),
    "ctx4096":    dict(cov_num=COV_BASE_NUM + COV_EXTRA_NUM + COV_CYCL_NUM,
                       cov_cat=COV_BASE_CAT, ctx=4096),
}

# ---------------------------------------------------------------------------
# Backtest runner
# ---------------------------------------------------------------------------
def run_backtest(config_name, cfg, dates, cible="debit"):
    res = []
    for t0_str in dates:
        t_start = time.time()
        X_train, X_test, y_train, y_test = split(df, t0_str, cible=cible)
        if X_test.empty:
            continue
        # Recompute features WITH t0 masking to avoid leakage on lag-derived covariates
        t0_ts = pd.Timestamp(t0_str, tz="Europe/Paris")
        fe_t0 = feature_engineering(df, cible, t0_ts)
        p = modele(X_train, y_train, X_test, cible, fe_t0,
                   cfg["cov_num"], cfg["cov_cat"], cfg["ctx"])
        r = pd.DataFrame({"modele": rmse(y_test, p, X_test.axe),
                          "naif_lag7j": rmse(y_test, X_test.lag_7j, X_test.axe)})
        res.append(r)
        print(f"  [{config_name}] {t0_str}: RMSE mean={r['modele'].mean():.1f} ({time.time()-t_start:.0f}s)", flush=True)
    if not res:
        return None
    df_res = pd.concat(res)
    par_arc = df_res.groupby(level=0).mean(numeric_only=True)
    return par_arc


# ---------------------------------------------------------------------------
# LightGBM reference
# ---------------------------------------------------------------------------
def run_lightgbm(dates, cible="debit"):
    import lightgbm as lgb
    params = dict(n_estimators=1000, learning_rate=0.05, num_leaves=127,
                  objective="huber", alpha=100, verbose=-1, random_state=0)
    res = []
    for t0_str in dates:
        X_train, X_test, y_train, y_test = split(df, t0_str, cible=cible)
        if X_test.empty:
            continue
        p = lgb.LGBMRegressor(**params).fit(X_train, y_train).predict(X_test).clip(0)
        res.append(pd.DataFrame({"lgb": rmse(y_test, p, X_test.axe),
                                 "naif_lag7j": rmse(y_test, X_test.lag_7j, X_test.axe)}))
    if not res:
        return None
    return pd.concat(res).groupby(level=0).mean(numeric_only=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dates", type=int, default=14, help="number of backtest dates (every Nth)")
    parser.add_argument("--step", type=int, default=8, help="step in ORIGINES")
    parser.add_argument("--configs", type=str, default="all", help="comma-separated config names")
    parser.add_argument("--cible", type=str, default="debit")
    args = parser.parse_args()

    dates = ORIGINES[::args.step][:args.dates]
    print(f"Backtest on {len(dates)} dates: {dates}", flush=True)

    if args.configs == "all":
        configs_to_run = list(CONFIGS.keys())
    else:
        configs_to_run = args.configs.split(",")

    all_results = {}

    # LightGBM reference first (fast)
    print("\n=== LightGBM reference ===", flush=True)
    lgb_res = run_lightgbm(dates, args.cible)
    if lgb_res is not None:
        print(lgb_res.round(1))
        print(f"Mean: {lgb_res['lgb'].mean():.1f}")
        all_results["lightgbm"] = lgb_res

    # Timestral variants
    for name in configs_to_run:
        print(f"\n=== Timestral: {name} ===", flush=True)
        t0 = time.time()
        res = run_backtest(name, CONFIGS[name], dates, args.cible)
        if res is not None:
            print(res.round(1))
            print(f"Mean modele: {res['modele'].mean():.1f}  Mean naif: {res['naif_lag7j'].mean():.1f}  ({time.time()-t0:.0f}s)")
            all_results[name] = res

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY (mean RMSE across 3 arcs)")
    print("=" * 60)
    summary = {}
    for name, res in all_results.items():
        col = "lgb" if name == "lightgbm" else "modele"
        summary[name] = round(res[col].mean(), 1)
        print(f"  {name:15s}: {res[col].mean():.1f}")
    if "naif_lag7j" in all_results.get("baseline", pd.DataFrame()):
        print(f"  {'naif_lag7j':15s}: {all_results['baseline']['naif_lag7j'].mean():.1f}")

    # Save results
    out = {k: v.round(3).to_dict() for k, v in all_results.items()}
    with open("test_results.json", "w") as f:
        json.dump({"summary": summary, "detail": out}, f, indent=2, default=str)
    print("\nSaved to test_results.json")
