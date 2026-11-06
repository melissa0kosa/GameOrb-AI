"""
Stage 2: hype correction model.

Stage 1 (train_model.py) predicts sales from what is on the store page, trained on
thousands of scraped games. Stage 2 learns how much pre-launch hype moves a game
above or below that expectation, trained on the (fewer) games tracked before launch
by track_upcoming.py and turned into rows by build_prelaunch_dataset.py.

Inputs (only 4, so it stays stable with ~200 rows):
  stage1_pred          Stage 1 prediction of log10 owners at the same horizon (e.g. 90 days)
  log_followers        log10(1 + followers at launch)
  log_growth           signed log10(1 + followers gained per day in the final month)
  log_days_tracked     log10(1 + days between first observation and launch)

Model: ridge regression with all coefficients constrained to be >= 0, so more hype or
a higher Stage 1 expectation can never lower the prediction (monotonic, no odd
patterns from a small sample). Evaluated on the newest launches (time-based split)
against Stage 1 alone and Stage 1 recalibrated without hype.

Usage:
  python build_prelaunch_dataset.py           # creates data/prelaunch_dataset.csv
  python train_hype_model.py                  # trains data/hype_model.joblib
"""

import argparse
import os
import sqlite3
import sys
from datetime import date

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scraper.enrichment import CatalogIndex, enrich
from train_model import add_age_feature, FeatureBuilder  # noqa: F401  (unpickling)

HYPE_FEATURES = ["stage1_pred", "log_followers", "log_growth", "log_days_tracked"]
MIN_ROWS = 30
ALPHAS = [0.01, 0.1, 0.3, 1, 3, 10, 30, 100]


# --------------------------------------------------------------------------- shared helpers
def signed_log(x):
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log10(1 + np.abs(x))


def hype_inputs(followers, growth, days_tracked) -> dict:
    return {
        "log_followers": float(np.log10(1 + max(followers or 0, 0))),
        "log_growth": 0.0 if growth is None or pd.isna(growth) else float(signed_log(growth)),
        "log_days_tracked": float(np.log10(1 + max(days_tracked or 0, 0))),
    }


def stage1_rows(appids, dataset_csv: str) -> pd.DataFrame:
    """Raw feature rows for these games: from the scraped CSV when available,
    otherwise fetched (and cached) through the scraper pipeline."""
    rows = []
    known = pd.read_csv(dataset_csv) if os.path.exists(dataset_csv) else pd.DataFrame(columns=["appid"])
    known = known.drop_duplicates("appid", keep="last").set_index("appid")
    missing = [a for a in appids if a not in known.index]
    pipe = None
    if missing:
        from scraper.pipeline import ScraperPipeline
        pipe = ScraperPipeline(request_delay=1.0)
        print(f"Fetching store data for {len(missing)} tracked games not in the scraped dataset...")
    for a in appids:
        if a in known.index:
            rows.append(known.loc[a].to_dict() | {"appid": a})
        else:
            r = pipe.fetch_game(int(a))
            if r:
                rows.append(r)
    return pd.DataFrame(rows)


def stage1_predict(bundle: dict, rows: pd.DataFrame, horizon_days: float, catalog=None) -> pd.Series:
    df = enrich(rows.reset_index(drop=True), catalog=catalog)
    df = add_age_feature(df, horizon_years=horizon_days / 365.25)
    X = bundle["feature_builder"].transform(df)
    return pd.Series(bundle["model"].predict(X), index=rows["appid"].to_numpy())


def make_model(alpha: float):
    # positive=True: every coefficient >= 0 -> monotonic in hype and in the Stage 1 prediction
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha, positive=True))


def live_hype_from_tracking(appid: int, db: str = "data/velocity_history.db"):
    """Current pre-launch hype inputs for a tracked game (None if not tracked)."""
    if not os.path.exists(db):
        return None
    conn = sqlite3.connect(db)
    try:
        s = pd.read_sql("SELECT snap_date, followers FROM daily_snapshots WHERE appid = ? AND is_released = 0 "
                        "AND followers IS NOT NULL ORDER BY snap_date", conn, params=(int(appid),),
                        parse_dates=["snap_date"])
    except Exception:
        return None
    finally:
        conn.close()
    if s.empty:
        return None
    last = s.iloc[-1]
    month_ago = s[s.snap_date <= last.snap_date - pd.Timedelta(days=25)]
    growth = None
    if not month_ago.empty:
        m = month_ago.iloc[-1]
        growth = (last.followers - m.followers) / max((last.snap_date - m.snap_date).days, 1)
    return {"followers": int(last.followers), "growth": growth,
            "days_tracked": int((last.snap_date - s.snap_date.min()).days)}


# --------------------------------------------------------------------------- training
def cv_mae(X, y, alpha, n_splits=5):
    k = min(n_splits, len(y))
    pred = np.zeros(len(y))
    for tr, te in KFold(k, shuffle=True, random_state=0).split(X):
        pred[te] = make_model(alpha).fit(X[tr], y[tr]).predict(X[te])
    return float(np.mean(np.abs(pred - y))), pred


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prelaunch", default="data/prelaunch_dataset.csv")
    ap.add_argument("--dataset", default="data/game_commercial_dataset.csv")
    ap.add_argument("--stage1", default="data/success_model.joblib")
    ap.add_argument("--save", default="data/hype_model.joblib")
    ap.add_argument("--test-frac", type=float, default=0.25, help="Newest launches held out for testing")
    args = ap.parse_args()

    if not os.path.exists(args.prelaunch):
        sys.exit(f"{args.prelaunch} not found. Run build_prelaunch_dataset.py first.")
    pre = pd.read_csv(args.prelaunch)
    if len(pre) < MIN_ROWS:
        sys.exit(f"Only {len(pre)} pre-launch rows (need >= {MIN_ROWS}). Keep the daily tracker running.")
    horizon = float(pre["horizon_days"].iloc[0])

    bundle = joblib.load(args.stage1)
    catalog = CatalogIndex(bundle["catalog_db"]) if bundle.get("catalog_db") else None
    rows = stage1_rows(pre["appid"].tolist(), args.dataset)
    s1 = stage1_predict(bundle, rows, horizon, catalog)

    pre = pre[pre["appid"].isin(s1.index)].copy()
    pre["stage1_pred"] = pre["appid"].map(s1)
    feats = pd.DataFrame([hype_inputs(f, g, d) for f, g, d in
                          zip(pre.prelaunch_followers, pre.prelaunch_follower_growth, pre.prelaunch_days_tracked)],
                         index=pre.index)
    pre = pd.concat([pre, feats], axis=1).sort_values("launch_date").reset_index(drop=True)
    X = pre[HYPE_FEATURES].to_numpy()
    y = pre["log_owners_at_horizon"].to_numpy()
    print(f"Stage 2 training rows: {len(pre)} games tracked before launch, outcome at {horizon:g} days.")

    # --- time-based test: newest launches held out
    n_test = max(int(len(pre) * args.test_frac), 5)
    tr, te = slice(0, len(pre) - n_test), slice(len(pre) - n_test, len(pre))
    alpha = min(ALPHAS, key=lambda a: cv_mae(X[tr], y[tr], a)[0])
    stage2 = make_model(alpha).fit(X[tr], y[tr]).predict(X[te])
    recal = make_model(alpha).fit(X[tr][:, :1], y[tr]).predict(X[te][:, :1])
    mae = lambda p: float(np.mean(np.abs(p - y[te])))
    print(f"\nTime-based test on the newest {n_test} launches "
          f"({pre.launch_date.iloc[len(pre) - n_test]} .. {pre.launch_date.iloc[-1]}):")
    print(f"  Stage 1 alone                 MAE log10 {mae(X[te][:, 0]):.3f}")
    print(f"  Stage 1 recalibrated, no hype MAE log10 {mae(recal):.3f}")
    print(f"  Stage 1 + hype (Stage 2)      MAE log10 {mae(stage2):.3f}")

    # --- final fit on all rows
    alpha = min(ALPHAS, key=lambda a: cv_mae(X, y, a)[0])
    _, oof = cv_mae(X, y, alpha)
    final = make_model(alpha).fit(X, y)
    ridge = final[-1]
    scale = final[0].scale_
    print(f"\nFinal model (alpha={alpha}): effect per 1 log10 step (10x) of each input:")
    for name, c, sd in zip(HYPE_FEATURES, ridge.coef_, scale):
        print(f"  {name:<18} {c / sd:+.3f} log10 owners  (x{10 ** (c / sd):.2f})")

    out = {
        "model": final, "features": HYPE_FEATURES, "alpha": alpha, "horizon_days": horizon,
        "residual_q": np.quantile(y - oof, [0.1, 0.9]).tolist(), "n_train": len(pre),
        "test": {"n": n_test, "stage1": mae(X[te][:, 0]), "recalibrated": mae(recal), "stage2": mae(stage2)},
        "trained_on": date.today().isoformat(), "stage1_path": args.stage1,
    }
    joblib.dump(out, args.save)
    print(f"\nSaved Stage 2 model to {args.save}")


if __name__ == "__main__":
    main()
