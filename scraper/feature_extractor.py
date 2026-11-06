"""
Feature Extractor and Dataset Builder for Commercial Success AI Models.

Combines and normalizes data from Steam Storefront, SteamSpy, and Follower Tracker
into a clean, numeric and categorical feature row ready for machine learning models
(LightGBM, XGBoost, CatBoost, RandomForest, Neural Networks).
"""

import math
from typing import Dict, Any, Optional, List
from .steam_api import parse_languages, parse_release_date, CATEGORY_FEATURE_MAP, KEY_LANGUAGES
from .steamspy_api import parse_owners_range, extract_tag_features
from .publisher_classifier import classify_publisher
from .tracker import WISHLIST_TO_FOLLOWER_MULTIPLIER


def assign_success_tier(owners_mid: Optional[int], revenue_est: Optional[float], is_released: bool = True) -> Optional[int]:
    """
    Assign commercial tier for classification models:
    - Tier 0: Niche / Micro / Below Commercial Threshold (< 10k owners or < $25k rev)
    - Tier 1: Moderate Indie / Sustainable (10k - 100k owners or $25k - $250k rev)
    - Tier 2: Commercial Success / Indie Hit (100k - 1M owners or $250k - $5M rev)
    - Tier 3: Major Commercial Blockbuster (> 1M owners or > $5M rev)

    Returns None for unreleased games or games with no verified sales/owner data.
    """
    if not is_released or (owners_mid is None and revenue_est is None):
        return None

    rev = revenue_est or 0.0
    own = owners_mid or 0

    if own >= 1_000_000 or rev >= 5_000_000:
        return 3
    elif own >= 100_000 or rev >= 250_000:
        return 2
    elif own >= 10_000 or rev >= 25_000:
        return 1
    else:
        return 0


def assign_price_tier(price_usd: float) -> int:
    """
    Assign price tier:
    0 = Free
    1 = Budget (< $10.00)
    2 = Core Indie ($10.00 - $24.99)
    3 = Mid-Tier / AA ($25.00 - $49.99)
    4 = Premium / AAA ($50.00+)
    """
    if price_usd <= 0.01:
        return 0
    elif price_usd < 10.0:
        return 1
    elif price_usd < 25.0:
        return 2
    elif price_usd < 50.0:
        return 3
    else:
        return 4


def extract_game_features(
    steam_details: Optional[Dict[str, Any]],
    steamspy_data: Optional[Dict[str, Any]] = None,
    follower_metrics: Optional[Dict[str, Any]] = None,
    appid: Optional[int] = None
) -> Dict[str, Any]:
    """
    Extract complete feature vector for a game.

    Args:
        steam_details: Raw dict from store.steampowered.com/api/appdetails
        steamspy_data: Raw dict from steamspy.com/api.php
        follower_metrics: Dict from FollowerTracker.compute_velocity() or manual snapshot
        appid: App ID integer fallback

    Returns:
        Structured dictionary of ML features and target variables.
    """
    s_det = steam_details or {}
    s_spy = steamspy_data or {}
    f_met = follower_metrics or {}

    game_id = appid or s_det.get("steam_appid") or s_spy.get("appid")
    name = s_det.get("name") or s_spy.get("name") or f"AppID_{game_id}"

    # 1. Pricing Features
    is_free = 1 if s_det.get("is_free", False) else 0
    price_overview = s_det.get("price_overview") or {}

    if is_free:
        price_usd = 0.0
        initial_price_usd = 0.0
        discount_percent = 0
    elif price_overview:
        initial_price_usd = price_overview.get("initial", 0) / 100.0
        price_usd = price_overview.get("final", 0) / 100.0
        discount_percent = price_overview.get("discount_percent", 0)
    elif "price" in s_spy:
        # SteamSpy price is in cents as string/int
        try:
            spy_price = float(s_spy.get("initialprice") or s_spy.get("price", 0)) / 100.0
            initial_price_usd = spy_price
            price_usd = float(s_spy.get("price", 0)) / 100.0
            discount_percent = int(s_spy.get("discount", 0))
        except (ValueError, TypeError):
            price_usd = 0.0
            initial_price_usd = 0.0
            discount_percent = 0
    else:
        price_usd = 0.0
        initial_price_usd = 0.0
        discount_percent = 0

    price_tier = assign_price_tier(initial_price_usd if initial_price_usd > 0 else price_usd)

    # 2. Publisher & Developer Classification (AAA vs Indie)
    publishers = s_det.get("publishers") or ([s_spy.get("publisher")] if s_spy.get("publisher") else [])
    developers = s_det.get("developers") or ([s_spy.get("developer")] if s_spy.get("developer") else [])
    pub_info = classify_publisher(publishers, developers)

    # 3. Release Timing & Seasonality Features
    rel_info = parse_release_date(s_det.get("release_date"))

    # 4. Localization Features
    lang_info = parse_languages(s_det.get("supported_languages"))
    has_major_asian_langs = int(
        lang_info["lang_schinese"] or lang_info["lang_tchinese"] or
        lang_info["lang_japanese"] or lang_info["lang_korean"]
    )
    has_figsr = int(
        lang_info["lang_french"] or lang_info["lang_german"] or
        lang_info["lang_spanish"] or lang_info["lang_russian"]
    )

    # 5. Platforms & Technical Features
    platforms = s_det.get("platforms") or {}
    platform_windows = 1 if platforms.get("windows", True) else 0
    platform_mac = 1 if platforms.get("mac", False) else 0
    platform_linux = 1 if platforms.get("linux", False) else 0
    platform_count = platform_windows + platform_mac + platform_linux

    # Categories
    categories = {cat.get("id"): cat.get("description") for cat in s_det.get("categories", [])}
    cat_features = {}
    for cat_id, feat_name in CATEGORY_FEATURE_MAP.items():
        cat_features[feat_name] = 1 if cat_id in categories else 0

    # Controller support
    controller_supp = s_det.get("controller_support", "").lower()
    has_controller = 1 if (controller_supp == "full" or cat_features.get("feat_full_controller_support")) else (
        1 if cat_features.get("feat_partial_controller_support") else 0
    )

    # Achievements & Metacritic
    achievements = s_det.get("achievements", {})
    achievements_count = achievements.get("total", 0) if isinstance(achievements, dict) else 0
    has_achievements = 1 if achievements_count > 0 else 0

    metacritic = s_det.get("metacritic") or {}
    metacritic_score = metacritic.get("score") if isinstance(metacritic, dict) else None
    has_metacritic = 1 if metacritic_score is not None else 0

    recommendations = s_det.get("recommendations", {})
    steam_recommendations = recommendations.get("total", 0) if isinstance(recommendations, dict) else 0

    # Content descriptors & Demos
    raw_age = s_det.get("required_age", 0)
    try:
        req_age = int(str(raw_age).replace("+", "").strip() or 0)
    except (ValueError, TypeError):
        req_age = 0
    is_mature = 1 if (req_age >= 17 or "mature" in str(s_det.get("content_descriptors", "")).lower()) else 0
    dlc_count = len(s_det.get("dlc", []))
    has_demo = 1 if (s_det.get("demos") or len(s_det.get("demos", [])) > 0) else 0

    # 6. Storefront Top-level Genres
    store_genres = [g.get("description", "").lower() for g in s_det.get("genres", [])]
    top_store_genres = ["action", "adventure", "rpg", "strategy", "simulation", "casual", "indie", "sports", "racing", "massively multiplayer"]
    store_genre_feats = {
        f"genre_store_{g.replace(' ', '_')}": 1 if any(g in sg for sg in store_genres) else 0
        for g in top_store_genres
    }

    # 7. SteamSpy Community Tags
    spy_tags = s_spy.get("tags") if isinstance(s_spy.get("tags"), dict) else {}
    tag_feats = extract_tag_features(spy_tags)

    # 8. Follower & Wishlist Pre-launch Velocity Features
    latest_followers = f_met.get("latest_followers")
    est_wishlists = f_met.get("estimated_wishlists")
    if latest_followers is None and "followers" in f_met:
        latest_followers = f_met["followers"]
        est_wishlists = int(latest_followers * WISHLIST_TO_FOLLOWER_MULTIPLIER)

    # If community followers were not tracked before, use community count if provided
    community_followers = latest_followers or 0
    estimated_wishlists = est_wishlists or int(community_followers * WISHLIST_TO_FOLLOWER_MULTIPLIER)

    follower_velocity_7d = f_met.get("follower_velocity_7d", 0.0)
    wishlist_velocity_7d = f_met.get("wishlist_velocity_7d", 0.0)
    follower_velocity_total = f_met.get("follower_velocity_total", 0.0)
    wishlist_velocity_total = f_met.get("wishlist_velocity_total", 0.0)
    launch_momentum_score = f_met.get("launch_momentum_score", 0.0)

    # If game is already released and we have days_since_release, compute follower accumulation per day
    is_game_released = rel_info.get("is_released", 1) == 1
    days_since = rel_info.get("days_since_release")
    if is_game_released and days_since and days_since > 0 and community_followers > 0:
        followers_per_day_since_release = round(community_followers / max(days_since, 1), 3)
    else:
        followers_per_day_since_release = None if not is_game_released else 0.0

    # 9. Target Variables (Commercial Success Metrics)
    owners_parsed = parse_owners_range(s_spy.get("owners"))
    owners_min = owners_parsed.get("owners_min")
    owners_max = owners_parsed.get("owners_max")
    owners_mid = owners_parsed.get("owners_mid")

    pos_reviews = s_spy.get("positive", 0)
    neg_reviews = s_spy.get("negative", 0)
    total_reviews = pos_reviews + neg_reviews
    review_positive_ratio = round(pos_reviews / total_reviews, 4) if total_reviews > 0 else None

    # Gross revenue estimation
    # Price * owners midpoint, or for free games 0.0
    effective_price = price_usd if price_usd > 0 else initial_price_usd
    if not is_game_released:
        estimated_gross_revenue_usd = None
        estimated_revenue_boxleiter = None
        success_tier = None
        is_millionaire_club = None
        follower_to_review_ratio = None
    else:
        if owners_mid is not None and not is_free:
            estimated_gross_revenue_usd = round(owners_mid * effective_price, 2)
        else:
            estimated_gross_revenue_usd = 0.0

        # Boxleiter estimation: total_reviews * 35 multiplier * price
        # Industry standard rule-of-thumb: each Steam review represents 25-50 paid copies
        boxleiter_owners = total_reviews * 35
        estimated_revenue_boxleiter = round(boxleiter_owners * effective_price, 2) if not is_free else 0.0

        success_tier = assign_success_tier(owners_mid, estimated_gross_revenue_usd, is_released=True)
        is_millionaire_club = 1 if (
            (estimated_gross_revenue_usd and estimated_gross_revenue_usd >= 1_000_000) or
            (owners_mid and owners_mid >= 200_000)
        ) else 0

        # Conversion Ratio: reviews per follower
        follower_to_review_ratio = round(total_reviews / community_followers, 4) if community_followers > 0 else None

    # Playtime engagement targets
    avg_playtime_forever = s_spy.get("average_forever", 0) if is_game_released else None
    median_playtime_forever = s_spy.get("median_forever", 0) if is_game_released else None
    ccu = s_spy.get("ccu", 0) if is_game_released else None

    # Construct the unified flat feature record
    record: Dict[str, Any] = {
        # Core identification
        "appid": game_id,
        "name": name,

        # Pricing features
        "is_free": is_free,
        "price_usd": round(price_usd, 2),
        "initial_price_usd": round(initial_price_usd, 2),
        "discount_percent": discount_percent,
        "price_tier": price_tier,

        # Publisher & Developer features (Indie vs AAA)
        "publisher_tier": pub_info["publisher_tier"],
        "is_aaa": pub_info["is_aaa"],
        "is_major_indie": pub_info["is_major_indie"],
        "is_self_published": pub_info["is_self_published"],
        "publisher_count": pub_info["publisher_count"],
        "developer_count": pub_info["developer_count"],

        # Release Timing & Seasonality
        "release_raw": rel_info["release_raw"],
        "is_released": rel_info["is_released"],
        "is_coming_soon": rel_info["is_coming_soon"],
        "release_year": rel_info["release_year"],
        "release_month": rel_info["release_month"],
        "release_quarter": rel_info["release_quarter"],
        "release_day": rel_info["release_day"],
        "release_day_of_week": rel_info["release_day_of_week"],
        "is_q4_holiday_competition": rel_info["is_q4_holiday_competition"],
        "is_summer_window": rel_info["is_summer_window"],
        "days_since_release": rel_info["days_since_release"],
        "days_until_release": rel_info["days_until_release"],

        # Localization Features
        "language_count": lang_info["language_count"],
        "audio_language_count": lang_info["audio_language_count"],
        "has_major_asian_langs": has_major_asian_langs,
        "has_figsr": has_figsr,
        **{k: lang_info[k] for k in KEY_LANGUAGES.keys()},

        # Platforms & Technical Polish
        "platform_windows": platform_windows,
        "platform_mac": platform_mac,
        "platform_linux": platform_linux,
        "platform_count": platform_count,
        "has_controller_support": has_controller,
        "has_achievements": has_achievements,
        "achievements_count": achievements_count,
        "has_metacritic": has_metacritic,
        "metacritic_score": metacritic_score,
        "has_demo": has_demo,
        "dlc_count": dlc_count,
        "required_age": req_age,
        "is_mature": is_mature,
        **cat_features,

        # Pre-Launch / Community Momentum & Wishlist Velocity
        "community_followers": community_followers,
        "estimated_wishlists": estimated_wishlists,
        "follower_velocity_7d": follower_velocity_7d,
        "wishlist_velocity_7d": wishlist_velocity_7d,
        "follower_velocity_total": follower_velocity_total,
        "wishlist_velocity_total": wishlist_velocity_total,
        "launch_momentum_score": launch_momentum_score,
        # Post-Launch Engagement & Conversion (NOT for pre-launch prediction)
        "postlaunch_followers_per_day": followers_per_day_since_release,
        "postlaunch_follower_to_review_ratio": follower_to_review_ratio,
        "followers_per_day_since_release": followers_per_day_since_release,
        "follower_to_review_ratio": follower_to_review_ratio,

        # Storefront Genres
        **store_genre_feats,

        # Community Tags
        **tag_feats,

        # TARGET VARIABLES (Commercial Success for ML training)
        "target_owners_min": owners_min,
        "target_owners_max": owners_max,
        "target_owners_mid": owners_mid,
        "target_positive_reviews": pos_reviews,
        "target_negative_reviews": neg_reviews,
        "target_total_reviews": total_reviews,
        "target_review_positive_ratio": review_positive_ratio,
        "target_steam_recommendations": steam_recommendations,
        "target_revenue_gross_est": estimated_gross_revenue_usd,
        "target_revenue_boxleiter_est": estimated_revenue_boxleiter,
        "target_ccu": ccu,
        "target_playtime_forever_avg": avg_playtime_forever,
        "target_playtime_forever_median": median_playtime_forever,
        "target_success_tier": success_tier,
        "target_is_millionaire_club": is_millionaire_club,
    }

    return record
