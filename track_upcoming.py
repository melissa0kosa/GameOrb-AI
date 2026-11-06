"""
Daily pre-launch tracker: builds the hype time-series the model is missing.

Run once a day (see run_daily_tracking.bat / Windows Task Scheduler). Each run:
  1. Discovers upcoming games from Steam's full "coming soon" list (incl. small,
     unhyped games) and the "popular coming soon" list, and adds new ones.
  2. Snapshots every tracked game that is due:
       - before launch: daily follower count (+ release date / days until release)
       - after launch : weekly follower + review count, until ~400 days after launch
  3. Stores snapshots in data/velocity_history.db (tables tracked_games, daily_snapshots).

After a few months, build_prelaunch_dataset.py turns this into training rows with
followers *at launch* (not today's inflated count) and the sales outcome N days later.

Usage:
  python track_upcoming.py                     # normal daily run
  python track_upcoming.py --max-games 300     # cap work per run
  python track_upcoming.py --status            # show tracking stats only
"""

import argparse
import logging
import sqlite3
import time
from datetime import date, datetime, timezone

from scraper.discovery import GameDiscoverer
from scraper.pipeline import ScraperPipeline
from scraper.steam_api import parse_release_date

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("tracker")

DB = "data/velocity_history.db"
POST_LAUNCH_DAYS = 400       # keep following outcomes for this long after release
POST_LAUNCH_EVERY = 7        # days between post-launch snapshots
PRE_LAUNCH_DAILY_WINDOW = 60 # daily snapshots when launch is at most this many days away
STOREFRONT_MAX_AGE_H = 72    # refresh release info at least every 3 days

SCHEMA = """
CREATE TABLE IF NOT EXISTS tracked_games (
    appid INTEGER PRIMARY KEY,
    name TEXT,
    source TEXT,
    first_seen TEXT,
    release_raw TEXT,
    released INTEGER DEFAULT 0,
    release_seen_date TEXT,          -- first day we observed it as released
    last_snapshot TEXT
);
CREATE TABLE IF NOT EXISTS daily_snapshots (
    appid INTEGER NOT NULL,
    snap_date TEXT NOT NULL,
    followers INTEGER,
    is_released INTEGER,
    days_until_release INTEGER,
    days_since_release INTEGER,
    total_reviews INTEGER,
    positive_reviews INTEGER,
    PRIMARY KEY (appid, snap_date)
);
"""


def connect():
    conn = sqlite3.connect(DB)
    conn.executescript(SCHEMA)
    return conn


def discover(conn, discoverer: GameDiscoverer, n_all: int, n_popular: int) -> int:
    today = date.today().isoformat()
    found = [(a, "comingsoon") for a in discoverer.discover_coming_soon(n_all)]
    found += [(a, "popularcomingsoon") for a in discoverer.discover_coming_soon(n_popular, popular_only=True)]
    before = conn.execute("SELECT COUNT(*) FROM tracked_games").fetchone()[0]
    with conn:
        conn.executemany(
            "INSERT OR IGNORE INTO tracked_games (appid, source, first_seen) VALUES (?, ?, ?)",
            [(a, s, today) for a, s in found])
    added = conn.execute("SELECT COUNT(*) FROM tracked_games").fetchone()[0] - before
    log.info("Discovery: %d listed, %d new games added to tracking.", len(found), added)
    return added


def due_games(conn, max_games: int):
    """Pre-launch: daily within 60 days of launch, every 3 days further out, weekly when the
    date is vague ("Coming soon", "Q1 2027"). Post-launch: weekly until POST_LAUNCH_DAYS.
    Steam's community site throttles hard, so this keeps a daily run manageable."""
    today = date.today()
    rows = conn.execute(
        "SELECT appid, released, release_seen_date, last_snapshot FROM tracked_games").fetchall()
    dur = dict(conn.execute("""
        SELECT appid, days_until_release FROM daily_snapshots s
        WHERE snap_date = (SELECT MAX(snap_date) FROM daily_snapshots WHERE appid = s.appid)
    """).fetchall())
    pre, post = [], []
    for appid, released, seen, last in rows:
        last_d = date.fromisoformat(last) if last else None
        age = (today - last_d).days if last_d else None
        if not released:
            d = dur.get(appid)
            interval = 1 if (d is not None and d <= PRE_LAUNCH_DAILY_WINDOW) else (3 if d is not None else 7)
            if age is None or age >= interval:
                pre.append(appid)
        else:
            since = (today - date.fromisoformat(seen)).days if seen else 0
            if since <= POST_LAUNCH_DAYS and (age is None or age >= POST_LAUNCH_EVERY):
                post.append(appid)
    # Closest launches first
    pre.sort(key=lambda a: (dur.get(a) is None, dur.get(a) if dur.get(a) is not None else 0))
    # Launch window and outcomes matter most; never starve post-launch snapshots
    return (post + pre)[:max_games], len(pre), len(post)


def snapshot(conn, pipe: ScraperPipeline, appid: int) -> bool:
    today = date.today().isoformat()
    details = pipe._get_cached_data(appid, "storefront", max_age_hours=STOREFRONT_MAX_AGE_H)
    if details is None:
        details = pipe.steam_api.get_app_details(appid)
        if details:
            pipe._set_cached_data(appid, "storefront", details)
    if not details:
        return False

    rel = parse_release_date(details.get("release_date"))
    released = int(rel["is_released"] == 1)
    followers = pipe.steam_api.get_community_followers(appid)

    total = positive = None
    if released:
        spy = pipe.steamspy.get_app_details(appid)
        if spy:
            positive = int(spy.get("positive") or 0)
            total = positive + int(spy.get("negative") or 0)

    with conn:
        conn.execute("""INSERT OR REPLACE INTO daily_snapshots VALUES (?,?,?,?,?,?,?,?)""",
                     (appid, today, followers, released, rel["days_until_release"],
                      rel["days_since_release"], total, positive))
        conn.execute("""
            UPDATE tracked_games SET name = ?, release_raw = ?, last_snapshot = ?,
                   released = MAX(released, ?),
                   release_seen_date = COALESCE(release_seen_date, CASE WHEN ? = 1 THEN ? END)
            WHERE appid = ?""",
            (details.get("name"), rel["release_raw"], today, released, released, today, appid))
        if followers is not None:
            # Keep the legacy velocity table in sync so pipeline velocity features keep working
            conn.execute("""INSERT INTO follower_snapshots (appid, timestamp, followers, days_until_release, is_released)
                            VALUES (?,?,?,?,?)""",
                         (appid, datetime.now(timezone.utc).isoformat(), followers,
                          rel["days_until_release"], released))
    return True


def status(conn):
    t = conn.execute("SELECT COUNT(*), SUM(released) FROM tracked_games").fetchone()
    s = conn.execute("SELECT COUNT(*), COUNT(DISTINCT snap_date), MIN(snap_date), MAX(snap_date), "
                     "SUM(followers IS NOT NULL) FROM daily_snapshots").fetchone()
    log.info("Tracked games: %d (%d released since tracking began)", t[0] or 0, t[1] or 0)
    log.info("Snapshots: %d over %d days (%s .. %s), followers found in %d",
             s[0] or 0, s[1] or 0, s[2], s[3], s[4] or 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-games", type=int, default=1500, help="Max snapshots per run")
    ap.add_argument("--discover-all", type=int, default=300, help="Games to read from the full coming-soon list")
    ap.add_argument("--discover-popular", type=int, default=200, help="Games to read from popular coming soon")
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    conn = connect()
    if args.status:
        status(conn)
        return

    start = time.time()
    pipe = ScraperPipeline(request_delay=args.delay)   # also creates follower_snapshots table
    discover(conn, GameDiscoverer(), args.discover_all, args.discover_popular)
    todo, n_pre, n_post = due_games(conn, args.max_games)
    log.info("Due: %d pre-launch, %d post-launch -> snapshotting %d", n_pre, n_post, len(todo))

    ok = fail = 0
    for i, appid in enumerate(todo, 1):
        try:
            if snapshot(conn, pipe, appid):
                ok += 1
            else:
                fail += 1
        except Exception as e:
            fail += 1
            log.warning("Snapshot failed for %d: %s", appid, e)
        if i % 100 == 0:
            log.info("  %d/%d done (%d ok, %d failed)", i, len(todo), ok, fail)
    log.info("Run finished in %.1f min: %d snapshots, %d failed.", (time.time() - start) / 60, ok, fail)
    status(conn)
    conn.close()


if __name__ == "__main__":
    main()
