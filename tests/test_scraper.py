"""
Unit tests for Predictive Game Analytics Scraper.
"""

import unittest
from datetime import datetime, timezone, timedelta
from scraper.publisher_classifier import classify_publisher
from scraper.steam_api import parse_languages, parse_release_date
from scraper.steamspy_api import parse_owners_range, extract_tag_features
from scraper.feature_extractor import assign_success_tier, assign_price_tier, extract_game_features
from scraper.tracker import FollowerTracker, WISHLIST_TO_FOLLOWER_MULTIPLIER


class TestPublisherClassifier(unittest.TestCase):
    def test_aaa_publisher(self):
        res = classify_publisher(["Electronic Arts"], ["EA DICE"])
        self.assertEqual(res["publisher_tier"], "AAA")
        self.assertEqual(res["is_aaa"], 1)
        self.assertEqual(res["is_major_indie"], 0)

        res2 = classify_publisher(["Sony Interactive Entertainment"], ["Naughty Dog"])
        self.assertEqual(res2["publisher_tier"], "AAA")

    def test_major_indie_publisher(self):
        res = classify_publisher(["Devolver Digital"], ["Nomada Studio"])
        self.assertEqual(res["publisher_tier"], "AA_Major_Indie")
        self.assertEqual(res["is_aaa"], 0)
        self.assertEqual(res["is_major_indie"], 1)

    def test_self_published_indie(self):
        res = classify_publisher(["Mega Crit Games"], ["Mega Crit Games"])
        self.assertEqual(res["publisher_tier"], "Indie")
        self.assertEqual(res["is_aaa"], 0)
        self.assertEqual(res["is_self_published"], 1)


class TestParsers(unittest.TestCase):
    def test_release_date_seasonality(self):
        # Holiday Q4 release
        q4_date = {"coming_soon": False, "date": "Nov 15, 2023"}
        res = parse_release_date(q4_date)
        self.assertEqual(res["release_year"], 2023)
        self.assertEqual(res["release_month"], 11)
        self.assertEqual(res["release_quarter"], 4)
        self.assertEqual(res["is_q4_holiday_competition"], 1)
        self.assertEqual(res["is_summer_window"], 0)

        # Summer window release
        summer_date = {"coming_soon": False, "date": "12 Jul, 2022"}
        res2 = parse_release_date(summer_date)
        self.assertEqual(res2["release_month"], 7)
        self.assertEqual(res2["is_summer_window"], 1)
        self.assertEqual(res2["is_q4_holiday_competition"], 0)

    def test_language_parsing(self):
        sample_html = (
            "English<strong>*</strong>, French, German, Simplified Chinese, Japanese"
            "<br><strong>*</strong>languages with full audio support"
        )
        res = parse_languages(sample_html)
        self.assertEqual(res["language_count"], 5)
        self.assertGreaterEqual(res["audio_language_count"], 1)
        self.assertEqual(res["lang_english"], 1)
        self.assertEqual(res["lang_schinese"], 1)
        self.assertEqual(res["lang_japanese"], 1)
        self.assertEqual(res["lang_russian"], 0)

    def test_owners_range_parsing(self):
        res = parse_owners_range("200,000 .. 500,000")
        self.assertEqual(res["owners_min"], 200000)
        self.assertEqual(res["owners_max"], 500000)
        self.assertEqual(res["owners_mid"], 350000)


class TestVelocityTracker(unittest.TestCase):
    def test_velocity_computation(self):
        import tempfile
        import os
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_velocity.db")
            tracker = FollowerTracker(db_path=db_path)

            now = datetime.now(timezone.utc)
            t1 = now - timedelta(days=10)
            t2 = now

            tracker.record_snapshot(12345, followers=1000, timestamp=t1)
            tracker.record_snapshot(12345, followers=2000, timestamp=t2)

            vel = tracker.compute_velocity(12345)
            self.assertEqual(vel["latest_followers"], 2000)
            self.assertEqual(vel["estimated_wishlists"], int(2000 * WISHLIST_TO_FOLLOWER_MULTIPLIER))
            # 1000 followers gained over 10 days = 100 followers/day
            self.assertAlmostEqual(vel["follower_velocity_total"], 100.0, delta=1.0)
            self.assertAlmostEqual(vel["wishlist_velocity_total"], 960.0, delta=10.0)
            self.assertGreater(vel["launch_momentum_score"], 0)


class TestFeatureExtractor(unittest.TestCase):
    def test_feature_extractor_pipeline(self):
        mock_steam = {
            "steam_appid": 1086940,
            "name": "Baldur's Gate 3",
            "is_free": False,
            "price_overview": {"initial": 5999, "final": 5999, "discount_percent": 0},
            "release_date": {"coming_soon": False, "date": "Aug 3, 2023"},
            "publishers": ["Larian Studios"],
            "developers": ["Larian Studios"],
            "supported_languages": "English<strong>*</strong>, French, German, Simplified Chinese",
            "platforms": {"windows": True, "mac": True, "linux": False},
            "categories": [{"id": 2, "description": "Single-player"}, {"id": 1, "description": "Multi-player"}],
            "genres": [{"id": "3", "description": "RPG"}],
            "recommendations": {"total": 500000},
        }
        mock_spy = {
            "appid": 1086940,
            "name": "Baldur's Gate 3",
            "owners": "20,000,000 .. 50,000,000",
            "positive": 500000,
            "negative": 15000,
            "tags": {"RPG": 1000, "Turn-Based Combat": 800, "Story Rich": 900},
            "ccu": 85000,
        }
        row = extract_game_features(mock_steam, mock_spy, follower_metrics={"latest_followers": 1000000})
        self.assertEqual(row["name"], "Baldur's Gate 3")
        self.assertEqual(row["price_usd"], 59.99)
        self.assertEqual(row["price_tier"], 4)
        self.assertEqual(row["target_success_tier"], 3)  # Blockbuster
        self.assertEqual(row["tag_rpg"], 1)
        self.assertEqual(row["is_self_published"], 1)
        self.assertEqual(row["feat_singleplayer"], 1)
        self.assertEqual(row["feat_multiplayer"], 1)

    def test_unreleased_game_targets_none(self):
        mock_steam_unreleased = {
            "steam_appid": 9999999,
            "name": "Upcoming Adventure",
            "is_free": False,
            "release_date": {"coming_soon": True, "date": "Coming Soon"},
            "publishers": ["Indie Dev"],
            "developers": ["Indie Dev"],
        }
        row = extract_game_features(mock_steam_unreleased, steamspy_data={}, follower_metrics={"latest_followers": 5000})
        self.assertIsNone(row["target_success_tier"])
        self.assertIsNone(row["target_is_millionaire_club"])
        self.assertIsNone(row["target_revenue_gross_est"])
        self.assertIsNone(row["postlaunch_follower_to_review_ratio"])
        self.assertEqual(row["is_released"], 0)


if __name__ == "__main__":
    unittest.main()
