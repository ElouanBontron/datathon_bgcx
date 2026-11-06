"""Fine-tune Timestral (head+representation, transformer frozen) on recent windows,
then compare zero-shot vs fine-tuned on 2026 holdout, focus Convention+Saints-Peres.

Usage: python test_finetune_full.py --epochs 3 --n-windows 48 --lr 1e-4
"""
import sys, argparse, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
from dataset_create import charger, split, ORIGINES, HORIZON, feature_engineering

print("Loading data...", flush=True)
df = charger(imputer=True)
from timestral import TimestralForecaster
print("Loading Timestral...", flush=True)
pipe = TimestralForecaster.from_pretrained("mistralai/Timestral-209M", device="cpu")

FOCUS = ["Convention", "Saints-Peres"]
COV_NUM = ["ferie", "pont", "paris_respire", "evenement", "confinement",
           "heure_sin", "heure_cos", "jour_sin", "jour_cos"]
COV_CAT = ["vacances_code", "type_jour"]
CTX, H = 2048, 24 + HORIZON
LGB_P = dict(n_estimators=1000, learning_rate=0.05, num_leaves=127, objective="huber",
             alpha=100, verbose=-1, random_state=0)

fe = feature_engineering(df)
vc = fe.vacances.cat.codes.astype(float).where(fe.vacances.cat.codes >= 0)
g = fe.assign(item_id=fe.axe.astype(str), timestamp=fe.datetime.dt.tz_convert(None),
    vacances_code=vc, type_jour=fe.type_jour.astype(float),
    heure_sin=np.sin(2*np.pi*fe.datetime.dt.hour/24), heure_cos=np.cos(2*np.pi*fe.datetime.dt.hour/24),
    jour_sin=np.sin(2*np.pi*fe.datetime.dt.dayofweek/7), jour_cos=np.cos(2*np.pi*fe.datetime.dt.dayofweek/7),
).astype({c: np.float32 for c in COV_NUM + COV_CAT})


def make_window(t0, axe, cible="debit"):
    heures = pd.date_range(t0 - pd.Timedelta(hours=CTX), periods=CTX + H, freq="h")
    serie = g.loc[g.item_id == axe].set_index("timestamp").reindex(heures)
    ctxv = serie[cible].to_numpy(dtype=np.float32)[:CTX]
    futv = serie[cible].to_numpy(dtype=np.float32)[CTX:]
    if np.isnan(ctxv).mean() > 0.2 or np.isnan(futv).mean() > 0.2:
        return None
    obs = ~np.isnan(ctxv)
    return {
        "context": np.where(obs, ctxv, 0.0),
        "obs": obs,
        "future": np.where(np.isnan(futv), 0.0, futv),
        "future_mask": ~np.isnan(futv),
        "known": serie[COV_NUM].to_numpy(dtype=np.float32).T,  # (V, CTX+H)
    }


def ts_predict(pipe, X_train, X_test, cible="debit"):
    """Same as notebook's timestral_pred but parameterized pipe (for FT vs zero-shot)."""
    t0 = fe.loc[X_train.index, "datetime"].max().tz_convert(None) + pd.Timedelta("1h")
    heures = pd.date_range(t0 - pd.Timedelta(hours=CTX), periods=CTX + H, freq="h")
    preds = []
    for axe in X_test.axe.astype(str).unique():
        s = g.loc[g.item_id == axe].set_index("timestamp").reindex(heures)
        fc = pipe.forecast(
            s[cible].to_numpy(dtype=np.float32)[:CTX][None, None, :], prediction_length=H,
            known_covariates=s[COV_NUM].to_numpy(dtype=np.float32).T[None, :, :],
            categorical_known_covariates=s[COV_CAT].to_numpy(dtype=np.float32).T[None, :, :])
        preds.append(pd.DataFrame({"item_id": axe, "timestamp": heures[CTX:],
                                   "prediction": fc.median[0, 0].cpu().numpy()}))
    pred = pd.concat(preds, ignore_index=True)
    test = pd.DataFrame({"item_id": X_test.axe.astype(str),
                         "timestamp": fe.loc[X_test.index, "datetime"].dt.tz_convert(None)})
    return test.merge(pred, on=["item_id", "timestamp"], how="left",
                      validate="many_to_one").prediction.to_numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--n-windows", type=int, default=48)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--months", type=int, default=14, help="training windows from last N months")
    args = parser.parse_args()

    model = pipe.model
    for p in model.parameters():
        p.requires_grad = False
    # Train ONLY the head (4.9M): freezing representation too avoids storing
    # activations through the 201M transformer (OOM on CPU otherwise).
    for p in model.head.parameters():
        p.requires_grad = True
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable: {n_train/1e6:.1f}M / 209M (head only)", flush=True)

    # Build training windows from recent data
    t_end = g.timestamp.max() - pd.Timedelta(hours=H)
    t_start = t_end - pd.Timedelta(days=args.months * 30)
    t0s = pd.date_range(t_start + pd.Timedelta(hours=CTX), t_end, freq="24h")
    if len(t0s) > args.n_windows:
        t0s = t0s[:: max(1, len(t0s) // args.n_windows)][: args.n_windows]
    windows = []
    for t0 in t0s:
        for axe in g.item_id.unique():
            w = make_window(t0, axe)
            if w is not None:
                windows.append(w)
    print(f"{len(windows)} training windows from {t_start.date()} to {t_end.date()}", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.01)
    med = pipe.median_index

    for epoch in range(args.epochs):
        t0t = time.time()
        model.train()
        np.random.shuffle(windows)
        losses = []
        for i in range(0, len(windows), args.batch):
            batch = windows[i:i + args.batch]
            B = len(batch)
            ctx = torch.tensor(np.stack([w["context"] for w in batch]))[:, None, :]
            obs = torch.tensor(np.stack([w["obs"] for w in batch]))[:, None, :]
            known = torch.tensor(np.stack([w["known"] for w in batch]))
            fut = torch.tensor(np.stack([w["future"] for w in batch]))[:, None, :]
            futm = torch.tensor(np.stack([w["future_mask"] for w in batch]))[:, None, :]
            opt.zero_grad()
            rows, mask, role_ids, kf, kf_mask, n_tgt = pipe._assemble_rows(
                ctx, obs, H, past_covariates=None, known_covariates=known)
            q = pipe._block_ar_decode(rows, mask, H, role_ids, kf, kf_mask)[:, :n_tgt]
            pred = q[:, 0, :, med]                       # (B, H)
            err = (pred - fut[:, 0, :]) ** 2 * futm[:, 0, :]
            loss = err.sum() / futm[:, 0, :].sum().clamp(min=1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            losses.append(loss.item())
            if (i // args.batch) % 3 == 0:
                print(f"  ep{epoch} step{i//args.batch}: loss={loss.item():.0f}", flush=True)
        print(f"Epoch {epoch}: mean loss={np.mean(losses):.0f} ({time.time()-t0t:.0f}s)", flush=True)

    torch.save(model.state_dict(), "timestral_ft.pt")
    print("Saved timestral_ft.pt", flush=True)

    # Reload zero-shot for comparison
    pipe_zs = TimestralForecaster.from_pretrained("mistralai/Timestral-209M", device="cpu")

    # Evaluate on 2026 holdout
    dates = [d for d in ORIGINES if d >= "2026-01-01"][::2]  # ~22 dates
    print(f"\nEvaluating on {len(dates)} holdout dates (2026)...", flush=True)
    rows = []
    for t0_str in dates:
        t0t = time.time()
        X_train, X_test, y_train, y_test = split(df, t0_str, cible="debit")
        if X_test.empty:
            continue
        m = X_test.axe.astype(str).isin(FOCUS)
        if not m.any():
            continue
        Xf, yf = X_test[m], y_test[m]
        p_zs = ts_predict(pipe_zs, X_train, Xf)
        p_ft = ts_predict(pipe, X_train, Xf)
        p_lgb = lgb.LGBMRegressor(**LGB_P).fit(X_train, y_train).predict(Xf).clip(0)
        for i in range(len(Xf)):
            rows.append({"t0": t0_str, "axe": Xf.axe.iloc[i], "y": yf.iloc[i],
                         "zs": p_zs[i], "ft": p_ft[i], "lgb": p_lgb[i],
                         "naif": Xf.lag_7j.iloc[i]})
        print(f"  {t0_str}: ok ({time.time()-t0t:.0f}s)", flush=True)

    D = pd.DataFrame(rows)
    print(f"\n=== HOLDOUT 2026: {D.t0.nunique()} dates, {len(D)} rows, focus={FOCUS} ===")
    print(f"{'config':18s} {'Convention':>10s} {'Saints-Peres':>12s} {'mean/date':>10s} {'std':>6s}")
    for name, p in [("ts zero-shot", D.zs), ("ts fine-tuned", D.ft), ("lgb", D.lgb), ("naif", D.naif),
                    ("zs+0.35lgb", 0.65*D.zs+0.35*D.lgb), ("ft+0.35lgb", 0.65*D.ft+0.35*D.lgb),
                    ("ft+0.5lgb", 0.5*D.ft+0.5*D.lgb)]:
        r = ((D.y - p)**2).groupby(D.axe).mean() ** .5
        pd_m = D.assign(e2=(D.y-p)**2).groupby("t0").e2.mean() ** .5
        print(f"{name:18s} {r.get('Convention',float('nan')):10.1f} {r.get('Saints-Peres',float('nan')):12.1f} {pd_m.mean():10.1f} {pd_m.std():6.1f}")
    D.to_csv("finetune_eval.csv", index=False)

if __name__ == "__main__":
    main()
