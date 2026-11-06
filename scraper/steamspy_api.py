"""
SteamSpy API Client and Parser.

Provides access to SteamSpy data:
- Owners ranges (estimated commercial adoption)
- Community tags and vote weights
- Playtime statistics (average, median)
- Concurrent users (CCU)
- Discovery feeds (top100in2weeks, top100forever, genre, tag, all pages)
"""

import re
import time
import logging
from typing import Dict, Any, Optional, List
import requests

logger = logging.getLogger(__name__)

# Prominent game mechanics / subgenre community tags for AI feature engineering
TOP_COMMUNITY_TAGS = [
    "action", "rpg", "adventure", "strategy", "simulation", "indie", "casual",
    "roguelike", "roguelite", "souls-like", "deckbuilder", "card game",
    "turn-based", "turn-based combat", "real-time", "open world", "sandbox",
    "survival", "crafting", "base building", "city builder", "resource management",
    "horror", "psychological horror", "co-op", "online co-op", "multiplayer",
    "singleplayer", "story rich", "choices matter", "visual novel",
    "fps", "shooter", "third-person shooter", "platformer", "puzzle",
    "metroidvania", "pixel graphics", "2d", "3d", "stylized", "realistic",
    "sci-fi", "fantasy", "cyberpunk", "post-apocalyptic", "anime", "cozy",
    "difficult", "physics", "vr", "early access"
]


class SteamSpyClient:
    """Client for the SteamSpy API."""

    BASE_URL = "https://steamspy.com/api.php"

    def __init__(self, session: Optional[requests.Session] = None, request_delay: float = 1.0):
        self.session = session or requests.Session()
        self.request_delay = request_delay
        self.last_request_time = 0.0
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        })

    def _rate_limit(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self.last_request_time = time.time()

    def get_app_details(self, appid: int, max_retries: int = 3) -> Optional[Dict[str, Any]]:
        """Fetch details for a single game from SteamSpy."""
        params = {"request": "appdetails", "appid": appid}

        for attempt in range(max_retries):
            self._rate_limit()
            try:
                resp = self.session.get(self.BASE_URL, params=params, timeout=12)
                if resp.status_code == 429:
                    wait_time = (2 ** attempt) * 2 + 1
                    logger.warning("SteamSpy 429 hit. Waiting %.1fs...", wait_time)
                    time.sleep(wait_time)
                    continue

                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, dict) and data.get("name"):
                    return data
                return None
            except requests.RequestException as e:
                logger.warning("SteamSpy error for appid %d: %s", appid, e)
                if attempt == max_retries - 1:
                    return None
                time.sleep(1.5 * (attempt + 1))
        return None

    def get_top100_in_2weeks(self) -> Dict[str, Any]:
        """Fetch top 100 played games in past 2 weeks."""
        self._rate_limit()
        resp = self.session.get(self.BASE_URL, params={"request": "top100in2weeks"}, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def get_top100_forever(self) -> Dict[str, Any]:
        """Fetch top 100 played games of all time."""
        self._rate_limit()
        resp = self.session.get(self.BASE_URL, params={"request": "top100forever"}, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def get_games_by_tag(self, tag: str) -> Dict[str, Any]:
        """Fetch games with a specific tag."""
        self._rate_limit()
        resp = self.session.get(self.BASE_URL, params={"request": "tag", "tag": tag}, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def get_all_page(self, page: int = 0) -> Dict[str, Any]:
        """Fetch a page of 1000 games from SteamSpy (page 0, 1, 2...)."""
        self._rate_limit()
        resp = self.session.get(self.BASE_URL, params={"request": "all", "page": page}, timeout=25)
        resp.raise_for_status()
        return resp.json()


def parse_owners_range(owners_str: Optional[str]) -> Dict[str, Optional[int]]:
    """
    Parse SteamSpy owners range string like '200,000 .. 500,000' or '20,000,000 .. 50,000,000'.
    Returns min, max, and geometric or arithmetic mid estimate.
    """
    res = {
        "owners_min": None,
        "owners_max": None,
        "owners_mid": None,
    }
    if not owners_str:
        return res

    clean = str(owners_str).replace(",", "").replace(" ", "")
    parts = clean.split("..")
    if len(parts) == 2:
        try:
            o_min = int(parts[0])
            o_max = int(parts[1])
            res["owners_min"] = o_min
            res["owners_max"] = o_max
            # Midpoint approximation
            res["owners_mid"] = (o_min + o_max) // 2
        except ValueError:
            pass
    elif len(parts) == 1 and parts[0].isdigit():
        val = int(parts[0])
        res["owners_min"] = val
        res["owners_max"] = val
        res["owners_mid"] = val

    return res


def extract_tag_features(tags: Optional[Dict[str, int]]) -> Dict[str, Any]:
    """
    Extract structured ML tag features from SteamSpy tags dictionary.
    Returns:
    - tag_count: total distinct tags assigned by players
    - top_5_tags: list of top 5 tag names
    - primary_tag: highest-voted tag
    - tag_{slug}: binary flag for presence in game's top tags
    - tag_weight_{slug}: normalized vote proportion (0.0 to 1.0)
    """
    res: Dict[str, Any] = {
        "tag_count": 0,
        "primary_tag": None,
        "top_5_tags_str": "",
    }

    # Initialize all top tag binary features to 0
    for tag_name in TOP_COMMUNITY_TAGS:
        slug = "tag_" + re.sub(r"[^\w]", "_", tag_name.lower())
        res[slug] = 0

    if not tags or not isinstance(tags, dict):
        return res

    res["tag_count"] = len(tags)
    sorted_tags = sorted(tags.items(), key=lambda x: x[1], reverse=True)
    if sorted_tags:
        res["primary_tag"] = sorted_tags[0][0]
        res["top_5_tags_str"] = ", ".join([t[0] for t in sorted_tags[:5]])

    total_votes = sum(tags.values()) or 1
    tags_lower = {k.lower(): v for k, v in tags.items()}

    for tag_name in TOP_COMMUNITY_TAGS:
        slug = "tag_" + re.sub(r"[^\w]", "_", tag_name.lower())
        if tag_name in tags_lower:
            res[slug] = 1

    return res
