"""
Predictive Game Analytics Scraper Package.
"""

from .pipeline import ScraperPipeline
from .steam_api import SteamApiClient
from .steamspy_api import SteamSpyClient
from .discovery import GameDiscoverer
from .tracker import FollowerTracker
from .publisher_classifier import classify_publisher
from .feature_extractor import extract_game_features

__all__ = [
    "ScraperPipeline",
    "SteamApiClient",
    "SteamSpyClient",
    "GameDiscoverer",
    "FollowerTracker",
    "classify_publisher",
    "extract_game_features",
]
