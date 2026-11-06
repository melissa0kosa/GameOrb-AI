"""
Turn the daily tracking data (track_upcoming.py) into a leak-free pre-launch dataset.

For every tracked game that has launched and been on sale for at least --horizon days:
  features (all measured BEFORE launch):
    prelaunch_followers         followers at the last snapshot before release (<= 7 days before)
    prelaunch_follower_growth   followers gained per day over the final ~30 days before launch
    prelaunch_days_tracked      how long before launch we started observing it
  target (measured AFTER launch):
    reviews_at_horizon          total reviews at ~horizon days after launch (+-14 days)
    log_owners_at_horizon       log10(reviews * 35), same scale as train_model.py

Rows are written to data/prelaunch_dataset.csv. Join them with the normal scraped
features on `appid` to train a model that uses real pre-launch hype.

Usage:
  python build_prelaunch_dataset.py              # 90-day horizon
  python build_prelaunch_dataset.py --horizon 365
"""

import argparse
import sqlite3
from datetime import date, timedelta

import numpy as np
import pandas as pd

REVIEW_MULTIPLIER = 35.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/velocity_history.db")
    ap.add_argument("--horizon", type=int, default=90, help="Days after launch to measure the outcome")
    ap.add_argument("--out", default="data/prelaunch_dataset.csv")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    games = pd.read_sql("SELECT appid, name, release_seen_date FROM tracked_games WHERE released = 1", conn)
    snaps = pd.read_sql("SELECT * FROM daily_snapshots", conn, parse_dates=["snap_date"])
    n_tracked = conn.execute("SELECT COUNT(*) FROM tracked_games").fetchone()[0]
    conn.close()

    rows, waiting, no_launch_obs = [], 0, 0
    for g in games.itertuples():
        s = snaps[snaps.appid == g.appid].sort_values("snap_date")
        launch = pd.Timestamp(g.release_seen_date)
        pre = s[(s.is_released == 0) & s.followers.notna()]
        last_pre = pre[pre.snap_date >= launch - timedelta(days=7)]
        if last_pre.empty:
            no_launch_obs += 1          # we never saw it in the final week before launch
            continue
        if (pd.Timestamp(date.today()) - launch).days < args.horizon - 14:
            waiting += 1                # not on sale long enough yet
            continue
        post = s[(s.is_released == 1) & s.total_reviews.notna()].copy()
        post["dist"] = ((post.snap_date - launch).dt.days - args.horizon).abs()
        post = post[post.dist <= 14]
        if post.empty:
            continue
        outcome = post.sort_values("dist").iloc[0]

        f_launch = int(last_pre.iloc[-1].followers)
        month_ago = pre[pre.snap_date <= launch - timedelta(days=25)]
        if not month_ago.empty:
            m = month_ago.iloc[-1]
            days = max((last_pre.iloc[-1].snap_date - m.snap_date).days, 1)
            growth = (f_launch - m.followers) / days
        else:
            growth = np.nan
        reviews = int(outcome.total_reviews)
        rows.append({
            "appid": g.appid, "name": g.name, "launch_date": launch.date().isoformat(),
            "prelaunch_followers": f_launch,
            "prelaunch_follower_growth": growth,
            "prelaunch_days_tracked": (launch - pre.snap_date.min()).days,
            "reviews_at_horizon": reviews,
            "log_owners_at_horizon": float(np.log10(max(reviews, 1) * REVIEW_MULTIPLIER)),
            "horizon_days": args.horizon,
        })

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"Tracked games: {n_tracked}; launched since tracking: {len(games)}")
    print(f"  ready (launch observed + {args.horizon}d outcome): {len(df)}  -> {args.out}")
    print(f"  launched, waiting for the {args.horizon}-day outcome: {waiting}")
    print(f"  launched but not observed in the last week before launch: {no_launch_obs}")
    if len(df) < 200:
        print("Not enough rows for training yet; keep the daily tracker running "
              "(roughly 200+ ready games are needed for a meaningful model).")


if __name__ == "__main__":
    main()
