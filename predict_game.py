"""
Inference Script: Predict Commercial Success for Any Steam Game.

Usage:
  # Predict success for an upcoming or existing game by AppID:
  python predict_game.py --appid 2320980

  # Use a custom model file (train it with: python train_model.py):
  python predict_game.py --appid 1086940 --model data/success_model.joblib
"""

import os
import sys
import argparse
import joblib
import pandas as pd
import numpy as np

from scraper.pipeline import ScraperPipeline
from scraper.enrichment import CatalogIndex, enrich
from train_model import owners_to_tier, add_age_feature, FeatureBuilder  # noqa: F401  (needed to unpickle bundle)
from train_hype_model import hype_inputs, live_hype_from_tracking

TIER_LABELS = {
    0: "Tier 0: Micro / Niche (< 10k copies, < $25k gross revenue)",
    1: "Tier 1: Sustainable Indie (10k - 100k copies, $25k - $250k revenue)",
    2: "Tier 2: Commercial Hit (100k - 1M copies, $250k - $5M revenue)",
    3: "Tier 3: Major Blockbuster (> 1M copies or > $5M revenue)",
}


def load_inference_bundle(model_path: str = "data/success_model.joblib") -> dict:
    """Load the bundle produced by train_model.py."""
    if not os.path.exists(model_path):
        sys.exit(f"Model '{model_path}' not found. Train it first: python train_model.py")
    return joblib.load(model_path)


def generate_commercial_recommendations(game_dict: dict) -> list:
    """Generate strategic commercial recommendations based on game features."""
    recommendations = []

    # Localization
    lang_count = game_dict.get("language_count", 0)
    has_asian = game_dict.get("has_major_asian_langs", 0)
    if lang_count < 5:
        recommendations.append("Localization: Supporting 5+ major languages (esp. Simplified Chinese, German, Russian) boosts global wishlist conversion significantly.")
    elif not has_asian:
        recommendations.append("Localization: Adding Simplified Chinese/Japanese/Korean localization opens access to >40% of Steam's active purchasing base.")

    # Technical polish
    if not game_dict.get("has_controller_support"):
        recommendations.append("Hardware: Adding full controller support unlocks Steam Deck verification, which drives substantial organic store discoverability.")
    if not game_dict.get("has_demo"):
        recommendations.append("Marketing: Launching a playable demo during Steam Next Fest historically increases pre-launch wishlist velocity by 3x - 10x.")

    # Release timing
    if game_dict.get("is_q4_holiday_competition"):
        recommendations.append("Release Window: Q4 (Oct-Dec) faces intense AAA marketing spend and heavy holiday discounting. Consider Q1/Q2 release window to minimize competition.")

    # Follower base
    followers = game_dict.get("community_followers", 0)
    if followers < 5000 and game_dict.get("is_coming_soon"):
        recommendations.append(f"Pre-Launch Momentum: Game has {followers:,} followers (~{int(followers * 9.6):,} wishlists). Industry benchmarks recommend reaching >= 7,000 wishlists before launch to trigger Steam's 'Popular New Releases' algorithm.")

    return recommendations


def predict_for_appid(appid: int, bundle: dict, horizon: float = 3.0, hype: dict = None):
    """Scrape live features for an AppID and run model inference."""
    print(f"\nFetching live metadata for AppID {appid}...")
    pipeline = ScraperPipeline(request_delay=0.5)
    game_dict = pipeline.fetch_game(appid)

    if not game_dict:
        print(f"Error: Could not retrieve data for AppID {appid}.")
        return

    name = game_dict.get("name", f"AppID_{appid}")
    price = game_dict.get("initial_price_usd", 0.0) or 0.0
    is_free = bool(game_dict.get("is_free"))

    print("\n" + "=" * 65)
    print(f"GAME PROFILE: {name} (AppID: {appid})")
    print("=" * 65)
    print(f"  - Status:              {'Released' if game_dict.get('is_released') else 'Upcoming / Unreleased'}")
    print(f"  - Publisher Tier:      {game_dict.get('publisher_tier', 'Indie')}")
    price_known = is_free or price > 0
    price_str = "Free-to-play" if is_free else (f"${price:.2f}" if price_known else "Not announced")
    print(f"  - Launch Price:        {price_str}")
    print(f"  - Community Followers: {game_dict.get('community_followers', 0):,}")
    print(f"  - Release:             {game_dict.get('release_raw', 'Unknown')}")
    print(f"  - Languages Supported: {game_dict.get('language_count', 0)}")
    print(f"  - Top Tags:            {game_dict.get('top_5_tags_str') or 'N/A'}")
    print("-" * 65)

    catalog = CatalogIndex(bundle["catalog_db"]) if bundle.get("catalog_db") else None
    df_single = enrich(pd.DataFrame([game_dict]), cache_db=pipeline.cache_db_path, catalog=catalog)
    if catalog is not None:
        print(f"  - Studio track record: {int(df_single['dev_prior_games'].iloc[0])} earlier games by developer, "
              f"{int(df_single['pub_prior_games'].iloc[0])} by publisher")
    lo_q, hi_q = bundle["residual_q"]
    rev_per_owner = 0.0 if is_free else price

    def predict_at(years: float) -> float:
        X = bundle["feature_builder"].transform(add_age_feature(df_single, horizon_years=years))
        return float(bundle["model"].predict(X)[0])

    log_owners = predict_at(horizon)
    owners = 10 ** log_owners
    revenue = owners * rev_per_owner
    owners_lo, owners_hi = 10 ** (log_owners + lo_q), 10 ** (log_owners + hi_q)
    tier = int(owners_to_tier([owners], [revenue])[0])

    print(f"\nAI PREDICTION ({bundle['model_name']}, trained on {bundle['n_train']} games):")
    print(f"  Owners after {horizon:g} year{'s' if horizon != 1 else ''}:{' ' if horizon != 1 else '  '}    ~{owners:,.0f}  (80% range: {owners_lo:,.0f} - {owners_hi:,.0f})")
    curve = ", ".join(f"{y:g}y ~{10 ** predict_at(y):,.0f}" for y in (1, 3, 5) if y != horizon)
    print(f"  Over time:                 {curve}")
    if not is_free and price_known:
        print(f"  Estimated gross revenue:   ~${revenue:,.0f}  (before Steam cut, discounts, refunds)")
    print(f"  Commercial tier:           Tier {tier}: {bundle['tier_names'][tier]}")
    if bundle["n_train"] < 300:
        print(f"  NOTE: small training set ({bundle['n_train']} games) - treat as a rough indication.")

    # Stage 2: hype correction, when a hype model exists and the game is being tracked pre-launch
    if hype is not None:
        live = live_hype_from_tracking(appid)
        if live is None:
            print("\n  Hype model: no pre-launch tracking data for this game (only games followed by "
                  "track_upcoming.py before launch get a hype-adjusted prediction).")
        else:
            h_days = hype["horizon_days"]
            s1_h = predict_at(h_days / 365.25)
            x = np.array([[s1_h] + [hype_inputs(live["followers"], live["growth"], live["days_tracked"])[k]
                                    for k in hype["features"][1:]]])
            log_h = float(hype["model"].predict(x)[0])
            lo_h, hi_h = hype["residual_q"]
            factor = 10 ** (log_h - s1_h)
            print(f"\n  WITH PRE-LAUNCH HYPE (Stage 2, trained on {hype['n_train']} tracked launches):")
            print(f"  Followers now: {live['followers']:,}"
                  + (f", growing {live['growth']:,.1f}/day" if live["growth"] is not None else ""))
            print(f"  Owners after {h_days:g} days:   ~{10 ** log_h:,.0f}  "
                  f"(80% range: {10 ** (log_h + lo_h):,.0f} - {10 ** (log_h + hi_h):,.0f})")
            print(f"  Hype adjustment:          x{factor:.2f} vs. the store-page-only prediction")
            print(f"  Owners after {horizon:g} years (adjusted, assumes the same x{factor:.2f}): "
                  f"~{owners * factor:,.0f}")

    recs = generate_commercial_recommendations(game_dict)
    if recs:
        print("\nSTRATEGIC COMMERCIAL RECOMMENDATIONS:")
        for idx, rec in enumerate(recs, 1):
            print(f"  {idx}. {rec}")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Predict commercial success for a game.")
    parser.add_argument("--appid", type=int, required=True, help="Steam AppID to evaluate")
    parser.add_argument("--model", default="data/success_model.joblib", help="Model file from train_model.py")
    parser.add_argument("--hype-model", default="data/hype_model.joblib",
                        help="Stage 2 model from train_hype_model.py (used when present)")
    parser.add_argument("--years", type=float, default=None,
                        help="Prediction horizon: owners after N years on sale (default: 3)")
    args = parser.parse_args()

    bundle = load_inference_bundle(args.model)
    horizon = args.years or bundle.get("default_horizon_years", 3.0)
    hype = joblib.load(args.hype_model) if os.path.exists(args.hype_model) else None
    predict_for_appid(args.appid, bundle, horizon=horizon, hype=hype)


if __name__ == "__main__":
    main()
