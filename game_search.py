"""
Game-name search with suggestions (like the Steam store search box).

Primary source: Steam's public store search (finds released AND upcoming games).
Fallback: the local SteamSpy catalog (82k released games), used when Steam is slow,
rate-limits us, or returns nothing.
"""

import re
import sqlite3
import threading
import time
from collections import deque
from typing import List, Optional, Tuple

import requests

STEAM_SEARCH_URL = "https://store.steampowered.com/api/storesearch/"
HEADERS = {"User-Agent": "Mozilla/5.0 (GameOrbAI)"}
# Not games: soundtracks, artbooks, wallpapers, supporter packs, tools...
NON_GAME = re.compile(r"\b(soundtrack|ost|artbook|art book|wallpapers?|supporter pack|"
                      r"season pass|dedicated server|sdk|demo)\b", re.I)
MAX_RESULTS = 8
STEAM_CALLS_PER_MINUTE = 60

Suggestion = Tuple[str, int]  # (label shown in the dropdown, appid)


class GameSearch:
    def __init__(self, catalog_db: str = "data/steamspy_catalog.db"):
        self._lock = threading.Lock()
        self._calls = deque()
        self._catalog: List[Tuple[str, str, int, int]] = []  # (lower name, name, appid, reviews)
        try:
            with sqlite3.connect(catalog_db) as con:
                rows = con.execute("SELECT appid, name, COALESCE(positive,0)+COALESCE(negative,0) "
                                   "FROM catalog WHERE name IS NOT NULL AND name != ''").fetchall()
            # most-reviewed first, so popular games win ties
            self._catalog = sorted(((n.lower(), n, a, r) for a, n, r in rows), key=lambda t: -t[3])
        except sqlite3.Error:
            pass

    # ------------------------------------------------------------------ sources
    def _steam_allowed(self) -> bool:
        now = time.time()
        with self._lock:
            while self._calls and now - self._calls[0] > 60:
                self._calls.popleft()
            if len(self._calls) >= STEAM_CALLS_PER_MINUTE:
                return False
            self._calls.append(now)
        return True

    def steam(self, term: str) -> Optional[List[Suggestion]]:
        """Steam store search. Returns None if Steam could not be asked (rate limit / error)."""
        if not self._steam_allowed():
            return None
        try:
            r = requests.get(STEAM_SEARCH_URL, params={"term": term, "l": "english", "cc": "US"},
                             headers=HEADERS, timeout=4)
            r.raise_for_status()
            items = r.json().get("items", [])
        except (requests.RequestException, ValueError):
            return None
        out = []
        for it in items:
            if it.get("type") != "app" or NON_GAME.search(it.get("name", "")):
                continue
            out.append((it["name"], int(it["id"])))  # name only; the price shows after picking
        return out[:MAX_RESULTS]

    def local(self, term: str) -> List[Suggestion]:
        """Catalog search: names starting with the term first, then names containing it."""
        t = term.lower().strip()
        starts, contains = [], []
        for low, name, appid, _ in self._catalog:
            if low.startswith(t):
                starts.append((name, appid))
            elif t in low:
                contains.append((name, appid))
            if len(starts) >= MAX_RESULTS:
                break
        return [(n, a) for n, a in (starts + contains)[:MAX_RESULTS]]

    def suggest(self, term: str) -> List[Suggestion]:
        term = (term or "").strip()
        if len(term) < 2:
            return []
        res = self.steam(term)
        if not res:  # None (unavailable) or empty: fall back to the local catalog
            res = self.local(term)
        return res
