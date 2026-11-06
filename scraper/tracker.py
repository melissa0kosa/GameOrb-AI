"""
Follower and Wishlist Velocity Tracker Module.

Tracks community follower counts over time to calculate:
- Pre-launch follower velocity (daily follower growth)
- Pre-launch wishlist estimation (using empirical follower-to-wishlist multiplier ~9.6x)
- Wishlist velocity prior to launch
- Conversion metrics (followers to reviews ratio, launch momentum score)
"""

import sqlite3
import os
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone

# Standard empirical multiplier from game industry research (GameDiscoverCo & HowToMarketAGame)
WISHLIST_TO_FOLLOWER_MULTIPLIER = 9.6


class FollowerTracker:
    """
    Persists and computes time-series velocity for game followers and wishlists.
    Uses SQLite database for local persistence.
    """

    def __init__(self, db_path: str = "data/velocity_history.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        conn = self._get_connection()
        try:
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS follower_snapshots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        appid INTEGER NOT NULL,
                        timestamp TEXT NOT NULL,
                        followers INTEGER NOT NULL,
                        days_until_release INTEGER,
                        is_released INTEGER NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_snapshots_appid_time
                    ON follower_snapshots (appid, timestamp)
                """)
        finally:
            conn.close()

    def record_snapshot(
        self,
        appid: int,
        followers: int,
        days_until_release: Optional[int] = None,
        is_released: int = 0,
        timestamp: Optional[datetime] = None
    ):
        """Record a follower count observation."""
        if followers is None or followers < 0:
            return

        ts = (timestamp or datetime.now(timezone.utc)).isoformat()
        conn = self._get_connection()
        try:
            with conn:
                conn.execute("""
                    INSERT INTO follower_snapshots
                    (appid, timestamp, followers, days_until_release, is_released)
                    VALUES (?, ?, ?, ?, ?)
                """, (appid, ts, followers, days_until_release, is_released))
        finally:
            conn.close()

    def compute_velocity(self, appid: int) -> Dict[str, Any]:
        """
        Compute follower and wishlist velocity metrics for a given game.
        Returns:
            - latest_followers: int
            - estimated_wishlists: int
            - snapshot_count: int
            - follower_velocity_7d: float (followers gained per day over last 7 days)
            - wishlist_velocity_7d: float
            - follower_velocity_total: float (followers gained per day over all recorded snapshots)
            - wishlist_velocity_total: float
            - launch_momentum_score: float
        """
        res = {
            "latest_followers": None,
            "estimated_wishlists": None,
            "snapshot_count": 0,
            "follower_velocity_7d": 0.0,
            "wishlist_velocity_7d": 0.0,
            "follower_velocity_total": 0.0,
            "wishlist_velocity_total": 0.0,
            "launch_momentum_score": 0.0,
        }

        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT timestamp, followers, days_until_release, is_released
                FROM follower_snapshots
                WHERE appid = ?
                ORDER BY timestamp ASC
            """, (appid,))
            rows = cur.fetchall()
        finally:
            conn.close()

        if not rows:
            return res

        res["snapshot_count"] = len(rows)
        latest_row = rows[-1]
        res["latest_followers"] = latest_row["followers"]
        res["estimated_wishlists"] = int(latest_row["followers"] * WISHLIST_TO_FOLLOWER_MULTIPLIER)

        if len(rows) < 2:
            return res

        # Total window velocity
        first_row = rows[0]
        dt_first = datetime.fromisoformat(first_row["timestamp"])
        dt_latest = datetime.fromisoformat(latest_row["timestamp"])
        delta_days_total = max((dt_latest - dt_first).total_seconds() / 86400.0, 0.001)

        delta_followers_total = latest_row["followers"] - first_row["followers"]
        vel_total = delta_followers_total / delta_days_total
        res["follower_velocity_total"] = round(vel_total, 2)
        res["wishlist_velocity_total"] = round(vel_total * WISHLIST_TO_FOLLOWER_MULTIPLIER, 2)

        # 7-day window velocity
        recent_rows = []
        for r in reversed(rows):
            dt_r = datetime.fromisoformat(r["timestamp"])
            if (dt_latest - dt_r).total_seconds() <= 7 * 86400.0:
                recent_rows.append(r)
            else:
                break

        if len(recent_rows) >= 2:
            dt_7d = datetime.fromisoformat(recent_rows[-1]["timestamp"])
            delta_days_7d = max((dt_latest - dt_7d).total_seconds() / 86400.0, 0.001)
            delta_followers_7d = latest_row["followers"] - recent_rows[-1]["followers"]
            vel_7d = delta_followers_7d / delta_days_7d
            res["follower_velocity_7d"] = round(vel_7d, 2)
            res["wishlist_velocity_7d"] = round(vel_7d * WISHLIST_TO_FOLLOWER_MULTIPLIER, 2)
        else:
            res["follower_velocity_7d"] = res["follower_velocity_total"]
            res["wishlist_velocity_7d"] = res["wishlist_velocity_total"]

        # Launch Momentum Score: logarithmic scale of velocity and follower base
        # Score = log10(max(followers, 1)) * max(1.0, 1.0 + vel_7d / 100.0)
        import math
        f_base = max(latest_row["followers"], 1)
        res["launch_momentum_score"] = round(
            math.log10(f_base) * max(0.5, 1.0 + (res["follower_velocity_7d"] / 500.0)),
            3
        )

        return res
