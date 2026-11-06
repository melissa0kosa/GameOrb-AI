"""
Scraper Orchestration Pipeline.

Coordinates data fetching across:
1. Steam Storefront API (pricing, release, localization, category polish)
2. SteamSpy API (owners, reviews, playtime, community tags)
3. Steam Community Hub (followers, members)
4. Local SQLite Cache (resumable scraping, offline development)
5. FollowerTracker (velocity & wishlist dynamics over time)
6. Feature Extractor (unified ML dataset export)
"""

import json
import sqlite3
import os
import time
import logging
from typing import List, Dict, Any, Optional
import pandas as pd
from tqdm import tqdm

from .steam_api import SteamApiClient, parse_release_date
from .steamspy_api import SteamSpyClient
from .tracker import FollowerTracker
from .feature_extractor import extract_game_features

logger = logging.getLogger(__name__)


class ScraperPipeline:
    """End-to-end scraper pipeline with caching and dataset export."""

    def __init__(
        self,
        cache_db_path: str = "data/scraper_cache.db",
        velocity_db_path: str = "data/velocity_history.db",
        request_delay: float = 1.0,
        enable_cache: bool = True
    ):
        self.cache_db_path = cache_db_path
        self.enable_cache = enable_cache
        os.makedirs(os.path.dirname(os.path.abspath(cache_db_path)), exist_ok=True)

        self.steam_api = SteamApiClient(request_delay=request_delay)
        self.steamspy = SteamSpyClient(request_delay=request_delay)
        self.tracker = FollowerTracker(db_path=velocity_db_path)

        if self.enable_cache:
            self._init_cache_db()

    def _get_cache_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.cache_db_path)

    def _init_cache_db(self):
        conn = self._get_cache_conn()
        try:
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS response_cache (
                        appid INTEGER NOT NULL,
                        endpoint TEXT NOT NULL,
                        data_json TEXT NOT NULL,
                        cached_at REAL NOT NULL,
                        PRIMARY KEY (appid, endpoint)
                    )
                """)
        finally:
            conn.close()

    def _get_cached_data(self, appid: int, endpoint: str, max_age_hours: float = 48.0) -> Optional[Any]:
        if not self.enable_cache:
            return None
        conn = self._get_cache_conn()
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT data_json, cached_at FROM response_cache
                WHERE appid = ? AND endpoint = ?
            """, (appid, endpoint))
            row = cur.fetchone()
            if row:
                data_json, cached_at = row
                if (time.time() - cached_at) <= (max_age_hours * 3600):
                    return json.loads(data_json)
        finally:
            conn.close()
        return None

    def _set_cached_data(self, appid: int, endpoint: str, data: Any):
        if not self.enable_cache or data is None:
            return
        conn = self._get_cache_conn()
        try:
            with conn:
                conn.execute("""
                    INSERT OR REPLACE INTO response_cache (appid, endpoint, data_json, cached_at)
                    VALUES (?, ?, ?, ?)
                """, (appid, endpoint, json.dumps(data), time.time()))
        finally:
            conn.close()

    def fetch_game(self, appid: int, refresh_cache: bool = False) -> Optional[Dict[str, Any]]:
        """
        Fetch all metadata for a single game across Steam and SteamSpy,
        update tracker snapshots, and return full feature row.
        """
        # 1. Steam Storefront details
        steam_details = None
        if not refresh_cache:
            steam_details = self._get_cached_data(appid, "storefront")
        if steam_details is None:
            steam_details = self.steam_api.get_app_details(appid)
            if steam_details:
                self._set_cached_data(appid, "storefront", steam_details)

        # 2. SteamSpy details
        steamspy_data = None
        if not refresh_cache:
            steamspy_data = self._get_cached_data(appid, "steamspy")
        if steamspy_data is None:
            steamspy_data = self.steamspy.get_app_details(appid)
            if steamspy_data:
                self._set_cached_data(appid, "steamspy", steamspy_data)

        # If neither endpoint returned anything, game doesn't exist or is region-locked
        if not steam_details and not steamspy_data:
            logger.warning("No data found for appid %d on Steam or SteamSpy", appid)
            return None

        # 3. Community followers
        followers = None
        if not refresh_cache:
            followers = self._get_cached_data(appid, "followers", max_age_hours=12.0)
        if followers is None:
            followers = self.steam_api.get_community_followers(appid)
            if followers is not None:
                self._set_cached_data(appid, "followers", followers)

        # 4. Record tracker snapshot
        is_coming_soon = 0
        days_until = None
        if steam_details and steam_details.get("release_date"):
            rel = steam_details.get("release_date", {})
            is_coming_soon = 1 if rel.get("coming_soon") else 0
            days_until = parse_release_date(rel).get("days_until_release")

        if followers is not None:
            self.tracker.record_snapshot(
                appid=appid,
                followers=followers,
                days_until_release=days_until,
                is_released=0 if is_coming_soon else 1
            )

        # 5. Compute velocity metrics
        velocity_metrics = self.tracker.compute_velocity(appid)
        if followers is not None and not velocity_metrics.get("latest_followers"):
            velocity_metrics["latest_followers"] = followers
            velocity_metrics["estimated_wishlists"] = int(followers * 9.6)

        # 6. Extract unified ML features
        features = extract_game_features(
            steam_details=steam_details,
            steamspy_data=steamspy_data,
            follower_metrics=velocity_metrics,
            appid=appid
        )

        return features

    def run(
        self,
        appids: List[int],
        refresh_cache: bool = False,
        output_csv: Optional[str] = "data/games_features.csv",
        output_json: Optional[str] = None,
        append: bool = False
    ) -> pd.DataFrame:
        """
        Scrape a list of AppIDs and compile into a DataFrame.
        """
        records: List[Dict[str, Any]] = []

        logger.info("Beginning scrape for %d games...", len(appids))
        for aid in tqdm(appids, desc="Scraping games"):
            try:
                row = self.fetch_game(aid, refresh_cache=refresh_cache)
                if row:
                    records.append(row)
            except Exception as e:
                logger.error("Failed to process appid %d: %s", aid, e)

        df = pd.DataFrame(records)

        if output_csv and not df.empty:
            os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
            if append and os.path.exists(output_csv):
                try:
                    existing_df = pd.read_csv(output_csv)
                    df = pd.concat([existing_df, df], ignore_index=True)
                    df = df.drop_duplicates(subset=["appid"], keep="last")
                except Exception as e:
                    logger.warning("Failed to merge with existing CSV: %s", e)
            df.to_csv(output_csv, index=False, encoding="utf-8")
            logger.info("Saved %d records to %s", len(df), output_csv)

        if output_json and not df.empty:
            os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
            df.to_json(output_json, orient="records", indent=2)
            logger.info("Saved %d records to %s", len(df), output_json)

        return df
