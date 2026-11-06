"""
Steam API and Storefront Scraper Client.

Provides structured access to:
- Steam Storefront API (https://store.steampowered.com/api/appdetails)
- Steam Community XML (https://steamcommunity.com/games/{appid}/memberslistxml/?xml=1)
- Steam User Stats Web API (ISteamUserStats/GetNumberOfCurrentPlayers)
"""

import re
import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Key language markers to check in localization strings
KEY_LANGUAGES = {
    "lang_english": ["english"],
    "lang_schinese": ["simplified chinese", "chinese (simplified)"],
    "lang_tchinese": ["traditional chinese", "chinese (traditional)"],
    "lang_japanese": ["japanese"],
    "lang_korean": ["korean"],
    "lang_german": ["german"],
    "lang_french": ["french"],
    "lang_spanish": ["spanish - spain", "spanish - latin america", "spanish"],
    "lang_russian": ["russian"],
    "lang_portuguese_br": ["portuguese - brazil", "portuguese"],
}

# Standard Steam storefront category ID mappings
CATEGORY_FEATURE_MAP = {
    2: "feat_singleplayer",
    1: "feat_multiplayer",
    9: "feat_coop",
    38: "feat_online_coop",
    23: "feat_steam_cloud",
    22: "feat_steam_achievements",
    28: "feat_full_controller_support",
    18: "feat_partial_controller_support",
    29: "feat_steam_trading_cards",
    30: "feat_steam_workshop",
    31: "feat_vr_support",
}


class SteamApiClient:
    """Client for querying Steam Storefront and Community endpoints with rate limiting."""

    def __init__(self, session: Optional[requests.Session] = None, request_delay: float = 1.0):
        self.session = session or requests.Session()
        self.request_delay = request_delay
        self.last_request_time = 0.0
        # steamcommunity.com throttles much harder than the store API (~60 quick requests
        # trigger HTTP 429 for minutes), so follower requests get their own adaptive pacing.
        self.community_delay = max(3.0, request_delay)
        self.community_min_delay = self.community_delay
        self.community_max_delay = 20.0
        self.last_community_time = 0.0
        self._community_ok_streak = 0
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        })

    def _rate_limit(self):
        """Enforce minimum delay between calls to avoid HTTP 429."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self.last_request_time = time.time()

    def _community_rate_limit(self):
        elapsed = time.time() - self.last_community_time
        if elapsed < self.community_delay:
            time.sleep(self.community_delay - elapsed)
        self.last_community_time = time.time()

    def _community_throttled(self, attempt: int) -> float:
        """Global cooldown after a 429: pause, then keep a slower pace for a while."""
        self.community_delay = min(self.community_delay * 1.5, self.community_max_delay)
        self._community_ok_streak = 0
        return min(60.0 * (2 ** attempt), 600.0)

    def _community_success(self):
        self._community_ok_streak += 1
        if self._community_ok_streak >= 50 and self.community_delay > self.community_min_delay:
            self.community_delay = max(self.community_min_delay, self.community_delay / 1.25)
            self._community_ok_streak = 0

    def get_app_details(self, appid: int, max_retries: int = 3) -> Optional[Dict[str, Any]]:
        """
        Fetch raw application details from Steam storefront API.
        Endpoint: https://store.steampowered.com/api/appdetails?appids={appid}&cc=us&l=en
        """
        url = "https://store.steampowered.com/api/appdetails"
        params = {"appids": appid, "cc": "us", "l": "en"}

        for attempt in range(max_retries):
            self._rate_limit()
            try:
                resp = self.session.get(url, params=params, timeout=12)
                if resp.status_code == 429:
                    wait_time = (2 ** attempt) * 2 + 1
                    logger.warning("Steam Store API 429 hit for appid %d. Backing off for %.1fs...", appid, wait_time)
                    time.sleep(wait_time)
                    continue

                resp.raise_for_status()
                data = resp.json()
                app_key = str(appid)
                if data and app_key in data and data[app_key].get("success"):
                    return data[app_key].get("data")
                return None
            except requests.RequestException as e:
                logger.warning("Attempt %d failed for appid %d: %s", attempt + 1, appid, e)
                if attempt == max_retries - 1:
                    return None
                time.sleep(1.5 * (attempt + 1))
        return None

    def get_community_followers(self, appid: int, max_retries: int = 2) -> Optional[int]:
        """
        Fetch follower / community member count from Steam community XML endpoint:
        https://steamcommunity.com/games/{appid}/memberslistxml/?xml=1
        """
        url = f"https://steamcommunity.com/games/{appid}/memberslistxml/?xml=1"
        max_retries = max(max_retries, 5)
        for attempt in range(max_retries):
            self._community_rate_limit()
            try:
                resp = self.session.get(url, timeout=15)
            except requests.RequestException as e:
                logger.warning("Followers request failed for appid %d (attempt %d): %s", appid, attempt + 1, e)
                time.sleep(5 * (attempt + 1))
                continue

            if resp.status_code == 200:
                match = re.search(r"<memberCount>(\d+)</memberCount>", resp.text)
                if match:
                    self._community_success()
                    return int(match.group(1))
                if "<memberList" in resp.text or "<error>" in resp.text:
                    # Valid XML answer without a count: the game has no community hub (yet)
                    return None
                if "<html" in resp.text[:500].lower():
                    # Steam redirected to an HTML page: no hub for this AppID
                    return None
                logger.warning("Unexpected followers response for appid %d", appid)
                return None

            if resp.status_code in (429, 500, 502, 503, 504):
                # Throttled / transient: pause all community requests, then retry slower
                wait = self._community_throttled(attempt) if resp.status_code == 429 else 10 * (2 ** attempt)
                logger.warning("Followers HTTP %d for appid %d, cooling down %ds (pace now %.1fs/request)",
                               resp.status_code, appid, wait, self.community_delay)
                time.sleep(wait)
                continue

            logger.warning("Followers HTTP %d for appid %d", resp.status_code, appid)
            return None
        logger.warning("Giving up on followers for appid %d after %d attempts", appid, max_retries)
        return None

    def get_current_players(self, appid: int) -> Optional[int]:
        """
        Fetch current active players using Steam Web API:
        https://api.steampowered.com/ISteamUserStats/GetNumberOfCurrentPlayers/v1/?appid={appid}
        """
        url = "https://api.steampowered.com/ISteamUserStats/GetNumberOfCurrentPlayers/v1/"
        params = {"appid": appid}
        self._rate_limit()
        try:
            resp = self.session.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("response", {}).get("result") == 1:
                    return data["response"].get("player_count")
        except requests.RequestException:
            pass
        return None


def parse_languages(html_str: Optional[str]) -> Dict[str, Any]:
    """
    Parse the HTML supported_languages string.
    Extracts total count, full audio count, and market language indicators.
    """
    result = {
        "language_count": 0,
        "audio_language_count": 0,
        **{k: 0 for k in KEY_LANGUAGES.keys()}
    }

    if not html_str:
        return result

    soup = BeautifulSoup(html_str, "html.parser")
    text = soup.get_text()

    # Split languages by commas or clean string
    # Remove footnotes e.g. "languages with full audio support"
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    raw_langs = lines[0] if lines else text
    raw_langs = re.sub(r"languages with full audio support.*", "", raw_langs, flags=re.I)

    # Count asterisk occurrences for audio support
    result["audio_language_count"] = html_str.count("*")

    # Split by comma
    lang_tokens = [tok.strip().strip("*").lower() for tok in raw_langs.split(",") if tok.strip()]
    result["language_count"] = max(len(lang_tokens), 1 if lang_tokens else 0)

    combined_text = html_str.lower()
    for feat_key, synonyms in KEY_LANGUAGES.items():
        if any(syn in combined_text for syn in synonyms):
            result[feat_key] = 1

    return result


def parse_release_date(release_info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Parse release date dictionary from Steam.
    Extracts year, month, quarter, holiday seasonality, day of week, days since release.
    """
    now = datetime.now(timezone.utc)
    res = {
        "release_raw": "",
        "is_coming_soon": 0,
        "is_released": 0,
        "release_year": None,
        "release_month": None,
        "release_quarter": None,
        "release_day": None,
        "release_day_of_week": None,
        "is_q4_holiday_competition": 0,
        "is_summer_window": 0,
        "days_since_release": None,
        "days_until_release": None,
    }

    if not release_info:
        return res

    is_coming = bool(release_info.get("coming_soon", False))
    raw_date = str(release_info.get("date", "")).strip()
    res["release_raw"] = raw_date
    res["is_coming_soon"] = 1 if is_coming else 0
    res["is_released"] = 0 if is_coming else 1

    if not raw_date:
        return res

    parsed_dt = None
    formats = [
        "%b %d, %Y",       # "Aug 3, 2023"
        "%d %b, %Y",       # "3 Aug, 2023"
        "%B %d, %Y",       # "August 3, 2023"
        "%d %B, %Y",       # "3 August, 2023"
        "%b %Y",           # "Aug 2023"
        "%B %Y",           # "August 2023"
        "%Y",              # "2023"
    ]

    for fmt in formats:
        try:
            parsed_dt = datetime.strptime(raw_date, fmt).replace(tzinfo=timezone.utc)
            break
        except ValueError:
            continue

    if not parsed_dt:
        # Fallback regex for 4 digit year
        year_match = re.search(r"\b(19\d\d|20\d\d)\b", raw_date)
        if year_match:
            res["release_year"] = int(year_match.group(1))
        # Month regex
        month_names = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
        }
        for m_str, m_num in month_names.items():
            if re.search(rf"\b{m_str}", raw_date, re.I):
                res["release_month"] = m_num
                res["release_quarter"] = (m_num - 1) // 3 + 1
                if m_num in [10, 11, 12]:
                    res["is_q4_holiday_competition"] = 1
                if m_num in [6, 7]:
                    res["is_summer_window"] = 1
                break
        return res

    # If full datetime was parsed successfully
    m = parsed_dt.month
    q = (m - 1) // 3 + 1
    res["release_year"] = parsed_dt.year
    res["release_month"] = m
    res["release_quarter"] = q
    res["release_day"] = parsed_dt.day
    res["release_day_of_week"] = parsed_dt.weekday()  # 0=Monday, 6=Sunday
    res["is_q4_holiday_competition"] = 1 if m in [10, 11, 12] else 0
    res["is_summer_window"] = 1 if m in [6, 7] else 0

    delta = (now - parsed_dt).days
    if delta >= 0 and not is_coming:
        res["days_since_release"] = delta
        res["days_until_release"] = 0
    else:
        res["days_since_release"] = 0
        res["days_until_release"] = max(0, -delta)

    return res
