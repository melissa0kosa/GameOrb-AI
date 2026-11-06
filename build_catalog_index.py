"""
Download the full SteamSpy catalog into a local SQLite table.

Used to compute developer / publisher track-record features: for any game we can
look up every *earlier* game (lower AppID) by the same studio and how it sold.

SteamSpy allows ~1 'all' request per minute, so the full catalog (~90 pages of
1,000 games) takes about 1.5 hours. The script is resumable: pages already
stored are skipped.

Usage:
  python build_catalog_index.py                 # fetch until the catalog ends
  python build_catalog_index.py --max-pages 20
"""

import argparse
import logging
import sqlite3
import time

from scraper.steamspy_api import SteamSpyClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("catalog")

SCHEMA = """
CREATE TABLE IF NOT EXISTS catalog (
    appid INTEGER PRIMARY KEY,
    name TEXT, developer TEXT, publisher TEXT,
    positive INTEGER, negative INTEGER, owners TEXT, initialprice INTEGER,
    page INTEGER
);
CREATE TABLE IF NOT EXISTS pages_done (page INTEGER PRIMARY KEY, n_games INTEGER, fetched_at REAL);
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/steamspy_catalog.db")
    ap.add_argument("--max-pages", type=int, default=200)
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    conn.executescript(SCHEMA)
    done = {r[0] for r in conn.execute("SELECT page FROM pages_done")}
    last = conn.execute("SELECT page FROM pages_done WHERE n_games < 1000 ORDER BY page LIMIT 1").fetchone()
    if last:
        log.info("Catalog already complete (last page %d). Delete %s to rebuild.", last[0], args.db)
        conn.close()
        return
    client = SteamSpyClient(request_delay=1.0)

    fetched_any = False
    for page in range(args.max_pages):
        if page in done:
            continue
        if fetched_any:
            time.sleep(61)  # SteamSpy limit for 'all' requests
        data = None
        for attempt in range(3):
            try:
                data = client.get_all_page(page=page)
                break
            except Exception as e:
                log.warning("Page %d attempt %d failed: %s", page, attempt + 1, e)
                time.sleep(61)
        fetched_any = True
        if data is None:
            log.error("Giving up on page %d; rerun later to resume.", page)
            continue
        rows = [
            (int(k), g.get("name"), g.get("developer"), g.get("publisher"),
             g.get("positive") or 0, g.get("negative") or 0, g.get("owners"),
             int(g.get("initialprice") or 0), page)
            for k, g in data.items() if str(k).isdigit() and isinstance(g, dict)
        ]
        if not rows:
            log.info("Page %d is empty - catalog complete.", page)
            break
        with conn:
            conn.executemany("INSERT OR REPLACE INTO catalog VALUES (?,?,?,?,?,?,?,?,?)", rows)
            conn.execute("INSERT OR REPLACE INTO pages_done VALUES (?,?,?)", (page, len(rows), time.time()))
        total = conn.execute("SELECT COUNT(*) FROM catalog").fetchone()[0]
        log.info("Page %d stored (%d games, %d total).", page, len(rows), total)
        if len(rows) < 1000:
            # Last page. SteamSpy answers pages past the end with HTTP 500 / invalid JSON
            # rather than an empty result, so stop here instead of retrying them.
            log.info("Page %d is partial - catalog complete.", page)
            break
    conn.close()
    log.info("CATALOG DONE")


if __name__ == "__main__":
    main()
