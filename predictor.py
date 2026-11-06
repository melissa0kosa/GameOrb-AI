"""
Reusable prediction service (used by predict_game.py and the Streamlit app).

Loads the Stage 1 model, the SteamSpy catalog (studio track record) and, if present,
the Stage 2 hype model ONCE, then predicts for:
  * a Steam game (AppID -> scraped store page), or
  * a manually described game that has no Steam page yet (what-if planning).

All predictions are returned as plain dicts so any front end can render them.
"""

import os
import re
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from scraper.enrichment import CatalogIndex, add_track_record_features, enrich, store_content_features
from scraper.feature_extractor import assign_price_tier
from scraper.publisher_classifier import classify_publisher
from scraper.steam_api import KEY_LANGUAGES
from scraper.steamspy_api import TOP_COMMUNITY_TAGS
from train_model import FeatureBuilder, add_age_feature, owners_to_tier  # noqa: F401  (unpickling)
from train_hype_model import hype_inputs, live_hype_from_tracking

# AppID used for manually described games: larger than any real AppID, so every
# catalog game counts as "earlier" for the studio track record.
MANUAL_APPID = 10 ** 12

STORE_GENRES = ["action", "adventure", "rpg", "strategy", "simulation", "casual", "indie",
                "sports", "racing", "massively multiplayer"]
LANGUAGE_LABELS = {
    "lang_english": "English", "lang_schinese": "Simplified Chinese", "lang_tchinese": "Traditional Chinese",
    "lang_japanese": "Japanese", "lang_korean": "Korean", "lang_german": "German", "lang_french": "French",
    "lang_spanish": "Spanish", "lang_russian": "Russian", "lang_portuguese_br": "Portuguese (Brazil)",
}
FEATURE_LABELS = {
    "feat_singleplayer": "Single-player", "feat_multiplayer": "Multiplayer", "feat_coop": "Co-op",
    "feat_online_coop": "Online co-op", "feat_steam_cloud": "Steam Cloud",
    "feat_steam_workshop": "Steam Workshop", "feat_vr_support": "VR support",
}
TAGS = list(TOP_COMMUNITY_TAGS)
CURVE_YEARS = [0.25, 0.5, 1, 2, 3, 4, 5]


def tag_column(tag: str) -> str:
    return "tag_" + re.sub(r"[^\w]", "_", tag.lower())


def parse_appid(text: str) -> Optional[int]:
    """Accept '1086940' or 'https://store.steampowered.com/app/1086940/...'."""
    text = (text or "").strip()
    m = re.search(r"/app/(\d+)", text) or re.fullmatch(r"(\d+)", text)
    return int(m.group(1)) if m else None


class SuccessPredictor:
    def __init__(self, model_path: str = "data/success_model.joblib",
                 hype_path: str = "data/hype_model.joblib",
                 cache_db: str = "data/scraper_cache.db",
                 tracking_db: str = "data/velocity_history.db"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model '{model_path}' not found. Train it with: python train_model.py")
        self.bundle = joblib.load(model_path)
        self.fb: FeatureBuilder = self.bundle["feature_builder"]
        self.model = self.bundle["model"]
        cat_path = self.bundle.get("catalog_db")
        self.catalog = CatalogIndex(cat_path) if cat_path and os.path.exists(cat_path) else None
        self.hype = joblib.load(hype_path) if os.path.exists(hype_path) else None
        self.cache_db = cache_db
        self.tracking_db = tracking_db
        self._pipeline = None

    # ------------------------------------------------------------------ model info
    @property
    def info(self) -> dict:
        b = self.bundle
        best = b["cv_results"][b["model_name"]]
        return {
            "model_name": b["model_name"], "n_train": b["n_train"],
            "mae_log10": best["mae_log10"], "spearman": best["spearman"],
            "tier_acc": best["tier_acc"], "tier_within1": best["tier_within1"],
            "temporal": b.get("temporal_holdout"), "tier_names": b["tier_names"],
            "has_catalog": self.catalog is not None, "has_hype": self.hype is not None,
            "default_horizon": b.get("default_horizon_years", 3.0),
        }

    # ------------------------------------------------------------------ inputs
    def fetch_appid(self, appid: int) -> Optional[dict]:
        """Scrape (or read from cache) the store page + SteamSpy data for a game."""
        if self._pipeline is None:
            from scraper.pipeline import ScraperPipeline
            self._pipeline = ScraperPipeline(cache_db_path=self.cache_db, request_delay=1.0)
        return self._pipeline.fetch_game(int(appid))

    def frame_from_game(self, game: dict) -> pd.DataFrame:
        return enrich(pd.DataFrame([game]), cache_db=self.cache_db, catalog=self.catalog)

    def frame_from_manual(self, spec: dict) -> pd.DataFrame:
        """Build a full feature row from a manual game description (see app form)."""
        row: Dict[str, object] = {c: 0 for c in self.fb.feature_cols}  # binary flags default off
        is_free = bool(spec.get("is_free"))
        price = 0.0 if is_free else float(spec.get("price", 0) or 0)
        row.update(is_free=int(is_free), initial_price_usd=price, price_usd=price,
                   price_tier=assign_price_tier(price))

        devs = [d.strip() for d in str(spec.get("developer", "")).split(";") if d.strip()]
        pubs = [p.strip() for p in str(spec.get("publisher", "")).split(";") if p.strip()] or devs
        pub = classify_publisher(pubs, devs)
        row.update(is_aaa=pub["is_aaa"], is_major_indie=pub["is_major_indie"],
                   is_self_published=pub["is_self_published"],
                   publisher_count=pub["publisher_count"], developer_count=pub["developer_count"])

        month = int(spec.get("release_month", 3))
        row.update(release_month=month, release_quarter=(month - 1) // 3 + 1,
                   release_day_of_week=int(spec.get("release_day_of_week", 3)),
                   is_q4_holiday_competition=int(month >= 10), is_summer_window=int(month in (6, 7)))

        langs = set(spec.get("languages", ["lang_english"]))
        for k in KEY_LANGUAGES:
            row[k] = int(k in langs)
        row["language_count"] = len(langs) + int(spec.get("other_languages", 0))
        row["audio_language_count"] = int(spec.get("audio_languages", 0))
        row["has_major_asian_langs"] = int(bool(langs & {"lang_schinese", "lang_tchinese", "lang_japanese", "lang_korean"}))
        row["has_figsr"] = int(bool(langs & {"lang_french", "lang_german", "lang_spanish", "lang_russian"}))

        plats = set(spec.get("platforms", ["windows"]))
        row.update(platform_windows=int("windows" in plats), platform_mac=int("mac" in plats),
                   platform_linux=int("linux" in plats), platform_count=len(plats))

        ctrl = spec.get("controller", "none")
        row.update(has_controller_support=int(ctrl != "none"),
                   feat_full_controller_support=int(ctrl == "full"),
                   feat_partial_controller_support=int(ctrl == "partial"))
        ach = int(spec.get("achievements", 0))
        row.update(achievements_count=ach, has_achievements=int(ach > 0), feat_steam_achievements=int(ach > 0))
        row["has_demo"] = int(bool(spec.get("has_demo")))
        mature = bool(spec.get("mature"))
        row.update(required_age=17 if mature else 0, is_mature=int(mature))
        for f in spec.get("features", []):
            row[f] = 1
        for g in spec.get("genres", []):
            row[f"genre_store_{g.replace(' ', '_')}"] = 1
        for t in spec.get("tags", []):
            row[tag_column(t)] = 1

        short = spec.get("short_description", "") or ""
        about = spec.get("about", "") or ""
        store = store_content_features({
            "short_description": short, "about_the_game": about,
            "detailed_description": about + "<img>" * int(spec.get("description_images", 0)),
            "screenshots": [None] * int(spec.get("screenshots", 0)),
            "movies": [None] * int(spec.get("trailers", 0)),
            "website": "x" if spec.get("has_website") else None,
            "support_info": {"email": "x" if spec.get("has_support_email", True) else ""},
        })
        store["store_desc_headers"] = int(spec.get("description_headers", 2))
        store["store_has_recommended_specs"] = int(bool(spec.get("recommended_specs", True)))
        store["req_min_ram_gb"] = spec.get("min_ram_gb")
        store["req_min_storage_gb"] = spec.get("min_storage_gb")
        row.update(store)

        row.update(appid=MANUAL_APPID, name=spec.get("name") or "My game", _developers=devs, _publishers=pubs)
        df = pd.DataFrame([row])
        if self.catalog is not None:
            df = add_track_record_features(df, self.catalog)
        return df

    # ------------------------------------------------------------------ prediction
    def _log_owners(self, df: pd.DataFrame, years: float) -> float:
        X = self.fb.transform(add_age_feature(df, horizon_years=years))
        return float(self.model.predict(X)[0])

    def predict(self, df: pd.DataFrame, horizon_years: float = 3.0, appid: Optional[int] = None) -> dict:
        r = df.iloc[0]
        is_free = bool(r.get("is_free", 0))
        price = float(r.get("initial_price_usd", 0) or 0)
        price_known = is_free or price > 0
        rpo = 0.0 if is_free else price

        log_o = self._log_owners(df, horizon_years)
        lo_q, hi_q = self.bundle["residual_q"]
        owners = 10 ** log_o
        revenue = owners * rpo if (price_known and not is_free) else None
        tier = int(owners_to_tier([owners], [owners * rpo])[0])

        res = {
            "horizon_years": horizon_years,
            "owners": owners, "owners_lo": 10 ** (log_o + lo_q), "owners_hi": 10 ** (log_o + hi_q),
            "revenue": revenue, "revenue_lo": (10 ** (log_o + lo_q)) * rpo if revenue is not None else None,
            "revenue_hi": (10 ** (log_o + hi_q)) * rpo if revenue is not None else None,
            "is_free": is_free, "price": price, "price_known": price_known,
            "tier": tier, "tier_name": self.bundle["tier_names"][tier],
            "curve": [(y, 10 ** self._log_owners(df, y)) for y in CURVE_YEARS],
            "studio": None, "hype": None,
        }
        if self.catalog is not None and "dev_prior_games" in df:
            res["studio"] = {k: (float(r[k]) if pd.notna(r[k]) else 0.0) for k in
                             ["dev_prior_games", "dev_prior_hits", "dev_prior_max_log_owners",
                              "pub_prior_games", "pub_prior_hits", "pub_prior_max_log_owners"]}

        if self.hype is not None and appid is not None:
            live = live_hype_from_tracking(appid, self.tracking_db)
            if live is not None:
                h_days = self.hype["horizon_days"]
                s1 = self._log_owners(df, h_days / 365.25)
                feats = hype_inputs(live["followers"], live["growth"], live["days_tracked"])
                x = np.array([[s1] + [feats[k] for k in self.hype["features"][1:]]])
                log_h = float(self.hype["model"].predict(x)[0])
                hlo, hhi = self.hype["residual_q"]
                res["hype"] = {
                    **live, "horizon_days": h_days, "owners": 10 ** log_h,
                    "owners_lo": 10 ** (log_h + hlo), "owners_hi": 10 ** (log_h + hhi),
                    "factor": 10 ** (log_h - s1), "n_train": self.hype["n_train"],
                }
        return res

    def what_if(self, df: pd.DataFrame, horizon_years: float = 3.0) -> List[dict]:
        """Effect of common changes on predicted owners (only changes the game doesn't have yet)."""
        base = self._log_owners(df, horizon_years)
        r = df.iloc[0]
        scen = []

        def add(label, changes):
            d = df.copy()
            for k, v in changes.items():
                d[k] = v
            scen.append({"change": label, "factor": 10 ** (self._log_owners(d, horizon_years) - base)})

        figs = ["lang_french", "lang_german", "lang_spanish"]
        missing_figs = [l for l in figs if not r.get(l)]
        if missing_figs:
            add("Add French, German & Spanish",
                {**{l: 1 for l in missing_figs}, "has_figsr": 1,
                 "language_count": r["language_count"] + len(missing_figs)})
        if not r.get("lang_schinese"):
            add("Add Simplified Chinese", {"lang_schinese": 1, "has_major_asian_langs": 1,
                                           "language_count": r["language_count"] + 1})
        if not r.get("has_controller_support"):
            add("Add full controller support", {"has_controller_support": 1, "feat_full_controller_support": 1})
        if not r.get("has_achievements"):
            add("Add Steam achievements (30)", {"has_achievements": 1, "feat_steam_achievements": 1,
                                                "achievements_count": 30})
        if not r.get("has_demo"):
            add("Release a demo", {"has_demo": 1})
        if r.get("store_n_trailers", 0) < 2:
            add("Have 2+ trailers", {"store_n_trailers": 2})
        if not r.get("platform_mac") or not r.get("platform_linux"):
            add("Ship on Mac & Linux too", {"platform_mac": 1, "platform_linux": 1, "platform_count": 3})
        if r.get("is_q4_holiday_competition"):
            add("Launch in March instead of Q4", {"release_month": 3, "release_quarter": 1,
                                                 "is_q4_holiday_competition": 0})
        # Price is deliberately not offered: across Steam, bigger productions charge more, so the
        # model associates higher prices with more owners - a confounded pattern, not advice.
        return sorted(scen, key=lambda s: -s["factor"])
