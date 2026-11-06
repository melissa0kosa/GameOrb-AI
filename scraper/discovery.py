"""
Game Discovery Engine.

Discovers games across Steam and SteamSpy for data collection:
- Top 100 played games (recent and all-time)
- Genre / Tag specific queries (e.g. Roguelike, RPG, Strategy, Indie)
- Popular upcoming games (pre-launch tracking)
- SteamSpy full catalog pages
- Custom AppID lists
"""

import re
import time
import logging
from typing import List, Set, Optional
import requests
from .steamspy_api import SteamSpyClient

logger = logging.getLogger(__name__)


class GameDiscoverer:
    """Discovers Steam AppIDs to crawl."""

    def __init__(self, steamspy_client: Optional[SteamSpyClient] = None):
        self.steamspy = steamspy_client or SteamSpyClient()
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        })

    def discover_top_games(self, limit: int = 100) -> List[int]:
        """Discover top games from SteamSpy (both 2-week active and all-time)."""
        appids: Set[int] = set()

        try:
            top_2w = self.steamspy.get_top100_in_2weeks()
            for k in top_2w.keys():
                if str(k).isdigit():
                    appids.add(int(k))
        except Exception as e:
            logger.warning("Error fetching top100in2weeks: %s", e)

        try:
            top_forever = self.steamspy.get_top100_forever()
            for k in top_forever.keys():
                if str(k).isdigit():
                    appids.add(int(k))
        except Exception as e:
            logger.warning("Error fetching top100forever: %s", e)

        return list(appids)[:limit]

    def discover_by_tag(self, tag: str, limit: int = 100) -> List[int]:
        """Discover games by community tag (e.g. 'Roguelike', 'Souls-like', 'Simulation')."""
        try:
            data = self.steamspy.get_games_by_tag(tag)
            appids = [int(k) for k in data.keys() if str(k).isdigit()]
            return appids[:limit]
        except Exception as e:
            logger.warning("Error discovering by tag '%s': %s", tag, e)
            return []

    def discover_upcoming_games(self, pages: int = 2) -> List[int]:
        """Discover upcoming/pre-launch games from Steam's popular coming soon list."""
        appids: List[int] = []
        for page in range(1, pages + 1):
            url = f"https://store.steampowered.com/search/results/?filter=popularcomingsoon&page={page}&json=1"
            try:
                resp = self.session.get(url, timeout=12)
                if resp.status_code == 200:
                    data = resp.json()
                    items = data.get("items", [])
                    for item in items:
                        logo = item.get("logo", "")
                        match = re.search(r"/apps/(\d+)/", logo)
                        if match:
                            aid = int(match.group(1))
                            if aid not in appids:
                                appids.append(aid)
            except Exception as e:
                logger.warning("Error fetching upcoming games page %d: %s", page, e)
                break

        return appids

    def discover_coming_soon(self, max_games: int = 500, popular_only: bool = False) -> List[int]:
        """
        Page through Steam's coming-soon search (100 per request).
        popular_only=False uses the full 'comingsoon' list (sorted by release date), which
        includes small unhyped games - needed so a pre-launch model does not only learn
        from games that were already popular before launch.
        """
        flt = "popularcomingsoon" if popular_only else "comingsoon"
        appids: List[int] = []
        for start in range(0, max_games, 100):
            url = (f"https://store.steampowered.com/search/results/?filter={flt}"
                   f"&start={start}&count=100&json=1")
            try:
                resp = self.session.get(url, timeout=15)
                items = resp.json().get("items", []) if resp.status_code == 200 else []
            except Exception as e:
                logger.warning("Error fetching %s (start=%d): %s", flt, start, e)
                break
            if not items:
                break
            for item in items:
                match = re.search(r"/apps/(\d+)/", item.get("logo", ""))
                if match and int(match.group(1)) not in appids:
                    appids.append(int(match.group(1)))
            time.sleep(1.0)
        return appids[:max_games]

    def discover_catalog_pages(self, max_pages: int = 5, limit: Optional[int] = None) -> List[int]:
        """
        Discover thousands of games from SteamSpy catalog pages (1,000 games per page).
        Page 0 has top played, subsequent pages explore the mid-tier and long-tail.
        """
        appids: List[int] = []
        for page in range(max_pages):
            try:
                logger.info("Fetching SteamSpy catalog page %d (1,000 games)...", page)
                data = self.steamspy.get_all_page(page=page)
                if not data or not isinstance(data, dict):
                    break
                for k in data.keys():
                    if str(k).isdigit():
                        aid = int(k)
                        if aid not in appids:
                            appids.append(aid)
                if limit and len(appids) >= limit:
                    break
            except Exception as e:
                logger.warning("Error fetching catalog page %d: %s", page, e)
                break

        return appids[:limit] if limit else appids

    # SteamSpy's "all" catalog is sorted by owners (page 0 = biggest games), so reading
    # pages in order yields only hits. These pages span the head through the long tail.
    BALANCED_PAGES = [0, 1, 2, 4, 6, 9, 13, 18, 24, 32, 42, 55, 70]

    def discover_balanced_training_sample(
        self, target_count: int = 2000, pages: Optional[List[int]] = None, seed: int = 42
    ) -> List[int]:
        """
        Sample games evenly across the four commercial tiers (review-based owner estimate:
        reviews * 35) using SteamSpy catalog pages spread across the whole catalog.
        SteamSpy rate-limits 'all' requests to ~1/minute, so this takes ~1 min per page.
        """
        import random
        pages = pages or self.BALANCED_PAGES
        buckets: dict = {0: [], 1: [], 2: [], 3: []}
        import time
        for i, page in enumerate(pages):
            if i > 0:
                time.sleep(61)  # SteamSpy: max 1 'all' request per minute
            try:
                logger.info("Fetching SteamSpy catalog page %d for balanced sampling...", page)
                data = self.steamspy.get_all_page(page=page)
            except Exception as e:
                logger.warning("Error fetching catalog page %d: %s", page, e)
                continue
            if not data or not isinstance(data, dict):
                logger.info("Page %d empty - reached end of catalog.", page)
                break
            for k, g in data.items():
                if not str(k).isdigit() or not isinstance(g, dict):
                    continue
                reviews = (g.get("positive") or 0) + (g.get("negative") or 0)
                owners = max(reviews, 1) * 35
                try:
                    price = float(g.get("initialprice") or 0) / 100.0
                except (TypeError, ValueError):
                    price = 0.0
                rev = owners * price
                if owners >= 1_000_000 or rev >= 5_000_000:
                    t = 3
                elif owners >= 100_000 or rev >= 250_000:
                    t = 2
                elif owners >= 10_000 or rev >= 25_000:
                    t = 1
                else:
                    t = 0
                buckets[t].append(int(k))

        rng = random.Random(seed)
        per_tier = target_count // 4
        chosen: List[int] = []
        for t, ids in buckets.items():
            ids = sorted(set(ids))
            rng.shuffle(ids)
            chosen.extend(ids[:per_tier])
            logger.info("Tier %d: %d candidates, sampled %d", t, len(ids), min(per_tier, len(ids)))
        # Top up from any tier with spare candidates if one tier was short
        if len(chosen) < target_count:
            spare = [i for ids in buckets.values() for i in ids if i not in set(chosen)]
            rng.shuffle(spare)
            chosen.extend(spare[: target_count - len(chosen)])
        rng.shuffle(chosen)
        return chosen

    def discover_diverse_training_sample(self, target_count: int = 100) -> List[int]:
        """
        Discovers a diverse balance of games for training ML models:
        - Major hits (top played)
        - Indie gems across varied tags (Roguelike, Strategy, Horror, Cozy, RPG)
        - Upcoming games for pre-launch evaluation
        """
        appids: Set[int] = set()

        # 1. Top played
        top = self.discover_top_games(limit=min(50, target_count // 2))
        appids.update(top)

        # 2. Genre representation
        genres_to_sample = ["Roguelike", "RPG", "Simulation", "Strategy", "Horror", "Deckbuilder"]
        per_genre = max(10, (target_count - len(appids)) // (len(genres_to_sample) + 1))
        for tag in genres_to_sample:
            if len(appids) >= target_count:
                break
            genre_ids = self.discover_by_tag(tag, limit=per_genre)
            appids.update(genre_ids)

        # 3. Add upcoming games
        upcoming = self.discover_upcoming_games(pages=1)
        appids.update(upcoming[:15])

        return list(appids)[:target_count]
