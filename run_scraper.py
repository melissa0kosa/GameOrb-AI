"""
CLI Runner for Predictive Game Analytics Web Scraper.

Usage examples:
  # Scrape a quick diverse sample of 15 games (AAA, Indie, upcoming):
  python run_scraper.py --mode sample --limit 15

  # Scrape top 50 games from SteamSpy:
  python run_scraper.py --mode top --limit 50 --output data/top50.csv

  # Scrape 25 upcoming games to track pre-launch follower & wishlist velocity:
  python run_scraper.py --mode upcoming --limit 25 --output data/upcoming.csv

  # Scrape by specific tag (e.g. Roguelike):
  python run_scraper.py --mode tag --tag Roguelike --limit 20

  # Tier-balanced training set (hits AND flops) - recommended for model training:
  python run_scraper.py --mode balanced --limit 2000 --append
  # Same, but from the local catalog, skipping games already in the dataset (no SteamSpy waits):
  python run_scraper.py --mode balanced-local --limit 2000 --append --no-followers

  # Scrape specific AppIDs:
  python run_scraper.py --appids 1086940 1145360 2124490
"""

import argparse
import os
import sys
import logging
import pandas as pd
from scraper import ScraperPipeline, GameDiscoverer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("scraper_cli")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scrape Steam Storefront, SteamSpy, and Community data for Game Commercial Success AI Models."
    )
    parser.add_argument(
        "--mode",
        choices=["sample", "top", "upcoming", "tag", "appids", "catalog", "balanced", "balanced-local"],
        default="sample",
        help="Scrape discovery mode: 'sample' (diverse mix), 'top' (SteamSpy top games), 'upcoming' (coming soon), 'tag', 'catalog' (full pages of 1,000 games each), 'balanced' (equal games per success tier, best for training), or 'appids'"
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=2,
        help="Number of SteamSpy catalog pages (1,000 games/page) to fetch when using --mode catalog (default: 2 = 2,000 games)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of games to scrape (default: 10)"
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="Roguelike",
        help="Tag name when using --mode tag (e.g., Roguelike, Souls-like, Strategy)"
    )
    parser.add_argument(
        "--appids",
        nargs="+",
        type=int,
        help="Specific Steam AppIDs to scrape (e.g. --appids 1086940 1145360)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/game_commercial_dataset.csv",
        help="Output CSV path (default: data/game_commercial_dataset.csv)"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional output JSON path"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Polite request delay in seconds to respect rate limits (default: 1.0s)"
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Force re-fetching data bypassing local SQLite cache"
    )
    parser.add_argument(
        "--no-followers",
        action="store_true",
        help="Skip community follower counts (post-launch data for released games, and the slowest, "
             "most throttled request) - recommended for training scrapes"
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append new scraped games to existing CSV without overwriting"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    discoverer = GameDiscoverer()

    if args.appids:
        appids = args.appids
    elif args.mode == "sample":
        logger.info("Discovering diverse sample of %d games...", args.limit)
        appids = discoverer.discover_diverse_training_sample(target_count=args.limit)
    elif args.mode == "top":
        logger.info("Discovering top %d games...", args.limit)
        appids = discoverer.discover_top_games(limit=args.limit)
    elif args.mode == "upcoming":
        logger.info("Discovering popular upcoming games for pre-launch tracking...")
        pages = max(1, (args.limit // 50) + 1)
        appids = discoverer.discover_upcoming_games(pages=pages)[:args.limit]
    elif args.mode == "catalog":
        logger.info("Discovering up to %d games across %d catalog pages...", args.limit or (args.pages * 1000), args.pages)
        appids = discoverer.discover_catalog_pages(max_pages=args.pages, limit=args.limit)
    elif args.mode == "balanced":
        logger.info("Discovering tier-balanced training sample of %d games (~13 min of catalog fetches)...", args.limit)
        appids = discoverer.discover_balanced_training_sample(target_count=args.limit)
    elif args.mode == "balanced-local":
        exclude = set()
        if args.append and os.path.exists(args.output):
            exclude = set(pd.read_csv(args.output, usecols=["appid"])["appid"].astype(int))
        logger.info("Sampling %d new tier-balanced games from the local catalog (skipping %d already scraped)...",
                    args.limit, len(exclude))
        appids = discoverer.discover_balanced_from_catalog(target_count=args.limit, exclude=exclude)
    elif args.mode == "tag":
        logger.info("Discovering games for tag '%s' (limit %d)...", args.tag, args.limit)
        appids = discoverer.discover_by_tag(args.tag, limit=args.limit)
    else:
        logger.error("No valid discovery target specified.")
        sys.exit(1)

    if not appids:
        logger.error("No AppIDs discovered to scrape.")
        sys.exit(1)

    logger.info("Discovered %d AppIDs: %s", len(appids), appids[:10])

    pipeline = ScraperPipeline(request_delay=args.delay)
    pipeline.fetch_followers = not args.no_followers
    df = pipeline.run(
        appids=appids,
        refresh_cache=args.refresh_cache,
        output_csv=args.output,
        output_json=args.output_json,
        append=args.append
    )

    print("\n" + "=" * 60)
    print("SCRAPE COMPLETE!")
    print(f"Total Games Scraped: {len(df)}")
    print(f"Total Features Extracted: {len(df.columns)}")
    print(f"Dataset Saved To: {args.output}")
    print("=" * 60)

    if not df.empty:
        # Display preview summary
        cols_to_show = [c for c in ["appid", "name", "price_usd", "publisher_tier", "release_year", "language_count", "community_followers", "target_owners_mid", "target_success_tier"] if c in df.columns]
        print("\nDataset Preview:")
        print(df[cols_to_show].head())


if __name__ == "__main__":
    main()
