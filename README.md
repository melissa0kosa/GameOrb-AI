# GameOrb AI

Predicts the commercial success of Steam games - estimated owners, gross revenue and success tier - from their store page or from a description of a game that is still being planned.

## Data pipeline: game analytics scraper for Steam

A high-performance, modular web scraper and dataset generator designed specifically for **training AI / Machine Learning models that predict the commercial success of video games**.

It aggregates data across:
1. **Steam Storefront API** (`https://store.steampowered.com/api/appdetails`)
2. **SteamSpy API** (`https://steamspy.com/api.php`)
3. **Steam Community XML Endpoints** (`https://steamcommunity.com/games/{appid}/memberslistxml/?xml=1`)
4. **Steam Web API** (`ISteamUserStats/GetNumberOfCurrentPlayers`)

---

## Key Extracted Features for AI Models

The pipeline extracts **152 ML-ready features** and separates them into input predictors (`X`) and commercial ground-truth targets (`y`):

### 1. Pricing & Monetization
- `price_usd`, `initial_price_usd`, `is_free`, `discount_percent`
- `price_tier`: 0 (Free), 1 (Budget <$10), 2 (Core Indie $10-$25), 3 (Mid-Tier AA $25-$50), 4 (Premium AAA $50+)

### 2. Publisher Stature (Indie vs. AAA)
- `publisher_tier`: `"AAA"`, `"AA_Major_Indie"`, `"Indie"`
- `is_aaa`: Flag detecting major conglomerates (EA, Ubisoft, Sony, Microsoft, Square Enix, Konami, Capcom, Bandai Namco, Bethesda, etc.)
- `is_major_indie`: Flag for established boutique publishers (Devolver Digital, Team17, Annapurna, Raw Fury, TinyBuild, Focus Entertainment, etc.)
- `is_self_published`: Flag when the developer matches the publisher (indie self-publishing)
- `publisher_count`, `developer_count`

### 3. Release Timing & Holiday Competition
- `release_year`, `release_month`, `release_quarter`, `release_day_of_week`
- `is_q4_holiday_competition`: Binary flag for October, November, December releases facing peak AAA competition and holiday spending windows
- `is_summer_window`: Binary flag for June & July releases
- `days_since_release`: Normalizes for lifetime age
- `days_until_release`: Days remaining for pre-launch titles

### 4. Follower & Wishlist Velocity Prior to Launch
- `community_followers`: Steam game hub group member count
- `estimated_wishlists`: Pre-launch wishlist projection using empirical industry multiplier (`followers * 9.6`)
- `follower_velocity_7d` & `wishlist_velocity_7d`: Growth rate in followers and wishlists per day
- `follower_velocity_total`: Velocity across all historical snapshots in the SQLite time-series database
- `launch_momentum_score`: Log-scaled composite momentum metric combining follower base and acceleration
- `followers_per_day_since_release`: Lifetime engagement velocity proxy
- `follower_to_review_ratio`: Conversion rate of followers to active reviewers

### 5. Supported Languages & Global Localization
- `language_count`: Total supported languages
- `audio_language_count`: Count of languages with full voice acting
- `has_major_asian_langs`: Simplified Chinese, Traditional Chinese, Japanese, Korean
- `has_figsr`: French, Italian, German, Spanish, Russian
- Binary flags for individual major markets: `lang_english`, `lang_schinese`, `lang_japanese`, `lang_german`, `lang_russian`, etc.

### 6. Genre & Community Mechanic Tags
- Storefront top-level genres: `genre_store_action`, `genre_store_rpg`, `genre_store_strategy`, `genre_store_simulation`, etc.
- 50+ rich SteamSpy community tags: `tag_roguelike`, `tag_deckbuilder`, `tag_souls_like`, `tag_open_world`, `tag_survival`, `tag_horror`, `tag_cozy`, `tag_story_rich`, `tag_early_access`, etc.
- `primary_tag`: Highest-voted community tag

### 7. Technical Polish & Steam Ecosystem Features
- Platform availability: `platform_windows`, `platform_mac`, `platform_linux`, `platform_count`
- Controller support: `has_controller_support`
- Steam achievements: `has_achievements`, `achievements_count`
- Metacritic: `has_metacritic`, `metacritic_score`
- Demo availability: `has_demo`, `dlc_count`
- Gameplay categories: `feat_singleplayer`, `feat_multiplayer`, `feat_coop`, `feat_online_coop`, `feat_steam_cloud`, `feat_steam_workshop`, etc.

### 8. Target Variables (Ground-Truth Commercial Success)
- `target_owners_min`, `target_owners_max`, `target_owners_mid` (From SteamSpy)
- `target_revenue_gross_est` = `target_owners_mid * price_usd`
- `target_revenue_boxleiter_est` = `total_reviews * 35.0 * price_usd` (Boxleiter multiplier method)
- `target_total_reviews`, `target_review_positive_ratio`
- `target_ccu`: Peak concurrent users
- `target_playtime_forever_median`: Player retention
- `target_success_tier`:
  - `0`: Niche / Underperforming (< 10k copies or < $25k revenue)
  - `1`: Moderate / Sustainable Indie (10k - 100k copies)
  - `2`: Commercial Hit (100k - 1M copies)
  - `3`: Major Blockbuster (> 1M copies or > $5M revenue)

---

## Web app

```bash
pip install -r requirements.txt
streamlit run app.py          # opens http://localhost:8501
```

- **Look up a Steam game**: type a game name and pick from live Steam-store suggestions (released or upcoming; falls back to the local catalog if Steam is unreachable). A store link or AppID also works.
- **Plan a new game**: describe a game with no Steam page yet (price, genres, tags,
  languages, platforms, store page, studio) and see its forecast plus "what-if" changes.
- **About the model**: data, metrics and limitations.

Live Steam lookups are cached for 24 h and capped at 10 per minute across all users,
because Steam throttles bursts of requests. The planner needs no network access.

### Deploy a public demo (Streamlit Community Cloud, free)
1. Push this repository to GitHub (public).
2. Go to https://share.streamlit.io, sign in with GitHub, click **Create app**, pick the
   repo, branch `main` and main file `app.py`.
3. Under **Advanced settings** choose Python **3.14** (the model was trained with it).
4. Deploy. Every `git push` redeploys automatically.

After retraining (`python train_model.py`), commit the new `data/success_model.joblib`.
If you upgrade scikit-learn or numpy, retrain before deploying: the model file only
loads reliably with the versions pinned in `requirements.txt`.

## Quickstart

### 1. Installation
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the Scraper (CLI)

```bash
# Scrape a diverse training sample of 50 games (AAA, Indie, upcoming):
python run_scraper.py --mode sample --limit 50 --output data/games.csv

# Scrape top 100 games from SteamSpy:
python run_scraper.py --mode top --limit 100 --output data/top100.csv

# Scrape upcoming games to track pre-launch follower velocity and wishlists:
python run_scraper.py --mode upcoming --limit 30 --output data/upcoming.csv

# Scrape games by tag:
python run_scraper.py --mode tag --tag "Roguelike" --limit 25

# Scrape specific AppIDs (e.g. Baldur's Gate 3, Hades, Silent Hill 2):
python run_scraper.py --appids 1086940 1145360 2124490
```

### 3. Train the Commercial-Success Model (recommended)
```bash
python train_model.py                    # pre-launch features only
python train_model.py --with-followers   # + community followers (optimistic, see below)
python predict_game.py --appid 2320980   # predict owners / revenue / tier for any game
```

`train_model.py` predicts **log10 lifetime owners**, estimated from review count
(Boxleiter: reviews x 35), and derives revenue and the 0-3 commercial tier from that.
It compares Ridge, Random Forest and Gradient Boosting against a median baseline using
repeated 5-fold cross-validation and saves the best one to `data/success_model.joblib`.

Why not SteamSpy owner brackets? They are coarse, inflated for small games, and the
lowest bracket's midpoint (10,000) equals the Tier 1 threshold, so Tier 0 never occurred.

Only pre-launch information is used as input. Post-launch signals (metacritic, DLC count,
current discount, tag-vote count, release year, reviews) are excluded. Follower counts of
released games are measured *today*, so `--with-followers` gives optimistic scores.

**Feature groups** (see `scraper/enrichment.py`):
- *Store page*: screenshot / trailer counts, description length and structure,
  minimum RAM and storage (a proxy for production scale), website / support email.
- *Description text*: TF-IDF of the short + long description compressed into 40 topics.
- *Studio track record*: how the developer's and publisher's **earlier** games (lower
  AppID) sold. Needs the full SteamSpy catalog, downloaded once (~1.5 h, resumable):
  `python build_catalog_index.py`

**Evaluation**: besides random cross-validation, the trainer runs a *time-based test*
(train on games released up to 2022, test on 2023-2024), which mirrors predicting future
games. `--ablation` shows how much each feature group adds.

**Prediction horizon**: game age (time on sale) is a model input, so predictions are
"owners after N years" (`predict_game.py --appid X --years 3`, default 3) instead of
mixing young and old games. Games younger than 90 days are left out of training.

### 4. Pre-launch hype tracking (builds the missing follower/wishlist signal)
Follower counts of released games are measured long after launch, so they cannot be used
as a pre-launch input. The tracker records them *before* launch instead:

```bash
python track_upcoming.py            # daily run: discover upcoming games + snapshot them
python track_upcoming.py --status   # how much data has accumulated
python build_prelaunch_dataset.py   # -> data/prelaunch_dataset.csv once games have launched
```

- Tracks Steam's full "coming soon" list (incl. small, unhyped games) plus "popular coming soon".
- Before launch: snapshots daily within 60 days of launch, every 3 days further out, and
  weekly for vague dates. After launch: weekly follower + review counts for ~400 days.
- steamcommunity.com returns HTTP 429 after ~60 quick requests; the client now pauses
  globally and slows down instead of silently dropping the follower count.
- `build_prelaunch_dataset.py` emits rows with followers **at launch**, follower growth in
  the final month, and reviews 90 days (or `--horizon N`) after launch.

Schedule it daily with Windows Task Scheduler (run once in PowerShell):
```powershell
$a = New-ScheduledTaskAction -Execute "$PWD\run_daily_tracking.bat" -WorkingDirectory "$PWD"
$t = New-ScheduledTaskTrigger -Daily -At 10:00
$s = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 6)
Register-ScheduledTask -TaskName "PredictiveAIGames Daily Tracking" -Action $a -Trigger $t -Settings $s
```
Output goes to `data/tracking.log`. Remove with
`Unregister-ScheduledTask -TaskName "PredictiveAIGames Daily Tracking"`.

### 5. Two-stage model (store page + pre-launch hype)
- **Stage 1** - `train_model.py`: thousands of scraped games, store-page features ->
  "what does a game like this usually sell?" (`data/success_model.joblib`)
- **Stage 2** - `train_hype_model.py`: games tracked before launch; a 4-input ridge model
  (Stage 1 prediction, followers at launch, follower growth, days tracked) with all
  coefficients forced >= 0, so more hype can never lower the prediction. It reports a
  time-based test against Stage 1 alone and Stage 1 recalibrated without hype.
  (`data/hype_model.joblib`, needs >= 30 rows; ~200+ recommended)

```bash
python build_prelaunch_dataset.py    # snapshots -> data/prelaunch_dataset.csv
python train_hype_model.py           # Stage 2
python predict_game.py --appid X     # adds a hype-adjusted prediction for tracked games
```
Retrain both monthly as tracked games launch. Once there are thousands of tracked
launches, a single gradient-boosting model with followers as a regular input is simpler.

**Building a training set**: `python run_scraper.py --mode balanced --limit 2000 --append`
samples equally from all four tiers. SteamSpy's catalog is sorted by owners, so
`--mode catalog` with few pages returns only hits.

### 3b. Legacy Baseline Model
```bash
python train_baseline_model.py --data data/game_commercial_dataset.csv
```

Outputs:
- Out-of-fold test accuracy
- Top 15 most predictive features ranked by Gini / Information Gain

---

## Project Architecture

```
PredictiveAIGames/
├── scraper/
│   ├── __init__.py
│   ├── steam_api.py            # Steam Storefront & Community API client
│   ├── steamspy_api.py         # SteamSpy API client (owners, playtime, tags)
│   ├── discovery.py            # Game discovery (top games, genre tags, upcoming releases)
│   ├── publisher_classifier.py # AAA / AA Major Indie / Small Indie classifier
│   ├── tracker.py              # Follower & wishlist velocity time-series tracker
│   ├── feature_extractor.py    # Merges APIs into 152 normalized ML features
│   └── pipeline.py             # Orchestrator with SQLite caching & rate limiting
├── data/
│   ├── scraper_cache.db        # SQLite response cache (resumable scraping)
│   ├── velocity_history.db     # SQLite snapshot store for velocity calculation
│   └── game_commercial_dataset.csv
├── tests/
│   └── test_scraper.py         # Unit tests
├── run_scraper.py              # Main CLI scraper executable
├── train_baseline_model.py     # Random Forest / Gradient Boosting baseline model
├── requirements.txt
└── README.md
```

## Polite Crawling & Resilience
- **Adaptive Jitter & Backoff**: Automatically respects Steam & SteamSpy rate limits (~1.0s delay by default, exponential backoff on HTTP 429).
- **SQLite Response Cache**: If a scrape is interrupted or rerun, previously fetched games are read instantly from the local database without making duplicate network requests.
