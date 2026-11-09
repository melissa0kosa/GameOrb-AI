"""
Download SteamSpy tag membership lists (which games carry each community tag) into
data/steamspy_catalog.db, table `tag_members(tag, appid)`.

Used by the genre-context features in scraper/enrichment.py: how crowded a game's
genre was when it came out, and how well earlier games of that genre sold.

    python build_tag_index.py
"""

import argparse
import logging
import sqlite3
import sys
import time

import requests

from scraper.steamspy_api import TOP_COMMUNITY_TAGS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("tags")

# SteamSpy wants the store's display name; a few of ours are spelled differently there.
NAME_VARIANTS = {
    "rpg": ["RPG"], "fps": ["FPS"], "vr": ["VR"], "2d": ["2D"], "3d": ["3D"],
    "roguelike": ["Rogue-like", "Roguelike"], "roguelite": ["Rogue-lite", "Roguelite"],
    "souls-like": ["Souls-like"], "deckbuilder": ["Deckbuilding", "Deckbuilder", "Card Battler"],
    "base building": ["Base-Building", "Base Building"], "co-op": ["Co-op"], "online co-op": ["Online Co-Op"], "sci-fi": ["Sci-fi"],
    "turn-based": ["Turn-Based"], "turn-based combat": ["Turn-Based Combat"], "real-time": ["Real-Time"],
    "third-person shooter": ["Third-Person Shooter"], "post-apocalyptic": ["Post-apocalyptic"],
}
MIN_PLAUSIBLE = 300   # fewer hits than this means SteamSpy did not recognise the name


def candidates(tag: str) -> list:
    return NAME_VARIANTS.get(tag, []) + [tag.title()]


def fetch(session: requests.Session, name: str) -> dict:
    for attempt in range(4):
        try:
            r = session.get("https://steamspy.com/api.php", params={"request": "tag", "tag": name}, timeout=60)
            r.raise_for_status()
            return r.json() or {}
        except (requests.RequestException, ValueError) as e:
            log.warning("%s: %s (retry %d)", name, e, attempt + 1)
            time.sleep(5 * (attempt + 1))
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog-db", default="data/steamspy_catalog.db")
    ap.add_argument("--delay", type=float, default=2.0)
    args = ap.parse_args()

    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (GameOrbAI)"
    con = sqlite3.connect(args.catalog_db)
    con.execute("CREATE TABLE IF NOT EXISTS tag_members (tag TEXT, appid INTEGER, PRIMARY KEY (tag, appid))")
    missing = []
    for tag in TOP_COMMUNITY_TAGS:
        best = {}
        for name in candidates(tag):
            data = fetch(session, name)
            time.sleep(args.delay)
            if len(data) > len(best):
                best = data
            if len(best) >= MIN_PLAUSIBLE:
                break
        if len(best) < MIN_PLAUSIBLE:
            missing.append(tag)
            log.warning("%-22s only %d games - skipped", tag, len(best))
            continue
        con.execute("DELETE FROM tag_members WHERE tag = ?", (tag,))
        con.executemany("INSERT OR IGNORE INTO tag_members VALUES (?, ?)", [(tag, int(a)) for a in best])
        con.commit()
        log.info("%-22s %6d games", tag, len(best))
    con.close()
    if missing:
        log.warning("No usable list for: %s", ", ".join(missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
