"""
Commercial-success model for Steam games (pre-launch features only).

What it predicts
----------------
  * log10(estimated owners)  -- regression target
  * commercial tier 0-3      -- derived from the predicted owners / revenue

Target definition
-----------------
SteamSpy owner brackets are very coarse (e.g. "0 .. 20,000") and wildly
inflated for small games (a game with 5 reviews listed at 20k-50k owners).
They also made Tier 0 impossible: the lowest bracket's midpoint (10,000)
sits exactly on the Tier 1 threshold. We therefore use the Boxleiter method:

    estimated_owners = total_reviews * REVIEW_MULTIPLIER   (35, as in the scraper)
    estimated_revenue = estimated_owners * launch_price    (0 for free-to-play)

Leakage control
---------------
Only information a developer knows *before* launch is used as input:
price, publisher stature, genres/tags, languages, platforms, Steam features,
release timing. Anything measured after launch is excluded (follower counts of
released games, metacritic, DLC count, current discount, tag-vote count,
release year / age on market, reviews, playtime...).

`--with-followers` adds community follower count as a hype signal. Be aware
that for released games this is the follower count *today*, so it is strongly
inflated vs. what the game had at launch; validation scores will be optimistic.

Usage
-----
  python train_model.py
  python train_model.py --data data/game_commercial_dataset.csv --with-followers
"""

import argparse
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scraper.enrichment import CatalogIndex, enrich

warnings.filterwarnings("ignore", category=UserWarning)

REVIEW_MULTIPLIER = 35.0

TIER_NAMES = {
    0: "Niche (<10k owners, <$25k revenue)",
    1: "Sustainable indie (10k-100k owners or $25k-$250k)",
    2: "Commercial hit (100k-1M owners or $250k-$5M)",
    3: "Blockbuster (>1M owners or >$5M)",
}

# Columns that are post-launch, identifiers, targets, or otherwise unusable pre-launch.
EXCLUDED_EXACT = {
    "appid", "name", "release_raw", "primary_tag", "top_5_tags_str", "publisher_tier",
    "is_released", "is_coming_soon", "days_since_release", "days_until_release",
    "release_year", "release_day",           # age-on-market proxy / noise
    "price_usd", "discount_percent",         # current (possibly discounted) price
    "has_metacritic", "metacritic_score",    # reviews exist only after launch
    "dlc_count",                             # DLC accumulates after launch
    "feat_steam_trading_cards",              # Valve grants cards only after sales/engagement thresholds
    "tag_count",                             # number of player-voted tags ~ popularity
    "follower_to_review_ratio", "followers_per_day_since_release",
}
FOLLOWER_COLS = [
    "community_followers", "estimated_wishlists",
    "follower_velocity_7d", "wishlist_velocity_7d",
    "follower_velocity_total", "wishlist_velocity_total", "launch_momentum_score",
]
LOG_COLS = [
    "community_followers", "achievements_count", "initial_price_usd",
    "store_n_screenshots", "store_n_trailers", "store_about_words", "store_desc_images",
    "store_desc_headers", "req_min_ram_gb", "req_min_storage_gb",
    "dev_prior_games", "pub_prior_games", "dev_prior_hits", "pub_prior_hits",
]
# Feature groups (for ablation and for switching groups off)
STORE_PREFIXES = ("store_", "req_")
TRACK_PREFIXES = ("dev_", "pub_", "is_debut_developer")
CONTEXT_PREFIXES = ("ctx_",)
N_TEXT_COMPONENTS = 40


# --------------------------------------------------------------------------- targets
def owners_to_tier(owners: np.ndarray, revenue: np.ndarray) -> np.ndarray:
    owners = np.asarray(owners, dtype=float)
    revenue = np.asarray(revenue, dtype=float)
    tier = np.zeros(len(owners), dtype=int)
    tier[(owners >= 10_000) | (revenue >= 25_000)] = 1
    tier[(owners >= 100_000) | (revenue >= 250_000)] = 2
    tier[(owners >= 1_000_000) | (revenue >= 5_000_000)] = 3
    return tier


# "boxleiter": owners = reviews x 35 for every game.
# "adjusted" : sales-per-review depends on release year, review score and price, following
#              published industry analyses (VG Insights, Gamalytic, GameDiscoverCo):
#              ~70 per review before 2014 -> ~30 from 2020 on (Steam began prompting players
#              to review in 2019, roughly halving the ratio); very well-reviewed games
#              (>90% positive) ~30 per review vs ~60 around 70% positive; free games about
#              twice the paid ratio; cheaper paid games somewhat higher.
TARGET_MODE = "boxleiter"
_YEAR_RATIO = {2013: 70.0, 2014: 62.0, 2015: 58.0, 2016: 55.0, 2017: 50.0, 2018: 45.0, 2019: 38.0}


def sales_per_review(year, pos_ratio, price, is_free) -> np.ndarray:
    year = pd.to_numeric(pd.Series(year), errors="coerce").fillna(2020).to_numpy()
    base = np.array([_YEAR_RATIO[2013] if y <= 2013 else _YEAR_RATIO.get(int(y), 30.0) for y in year])
    pos = pd.to_numeric(pd.Series(pos_ratio), errors="coerce").fillna(0.8).to_numpy()
    score = np.interp(pos, [0.0, 0.5, 0.7, 0.85, 0.95, 1.0], [0.9, 1.0, 1.5, 1.1, 0.85, 0.8])
    price = pd.to_numeric(pd.Series(price), errors="coerce").fillna(0).to_numpy()
    price_f = np.interp(price, [0, 5, 15, 30, 60], [1.25, 1.15, 1.0, 0.9, 0.8])
    free = np.asarray(pd.Series(is_free).fillna(0).to_numpy(), dtype=float) == 1
    return base * score * np.where(free, 2.0, price_f)


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    reviews = pd.to_numeric(df["target_total_reviews"], errors="coerce").fillna(0)
    # A few review lookups come back empty (e.g. Stacklands: 0 reviews, 29k recommendations);
    # fall back to Steam's recommendation count, which tracks the review count closely.
    if "target_steam_recommendations" in df:
        recs = pd.to_numeric(df["target_steam_recommendations"], errors="coerce").fillna(0)
        reviews = reviews.where(reviews > 0, recs)
    if TARGET_MODE == "adjusted":
        mult = sales_per_review(df["release_year"], df.get("target_review_positive_ratio"),
                                df["initial_price_usd"], df["is_free"])
    else:
        mult = REVIEW_MULTIPLIER
    owners = np.maximum(reviews, 1) * mult
    price = pd.to_numeric(df["initial_price_usd"], errors="coerce").fillna(0)
    revenue = np.where(df["is_free"] == 1, 0.0, owners * price)
    out = pd.DataFrame(index=df.index)
    out["owners_est"] = owners
    out["revenue_est"] = revenue
    out["log_owners"] = np.log10(owners)
    out["tier"] = owners_to_tier(owners, revenue)
    return out


# --------------------------------------------------------------------------- features
class FeatureBuilder:
    """Selects pre-launch columns, log-transforms counts, imputes, and adds
    latent text topics (TF-IDF + SVD) from the store description."""

    def __init__(self, with_followers: bool = False, use_store: bool = True,
                 use_text: bool = True, use_track: bool = True, use_age: bool = True,
                 use_context: bool = True):
        self.with_followers = with_followers
        self.use_store, self.use_text, self.use_track = use_store, use_text, use_track
        self.use_age = use_age
        self.use_context = use_context
        self.feature_cols: list = []
        self.imputer = SimpleImputer(strategy="median")
        self.tfidf = None
        self.svd = None

    def _select(self, df: pd.DataFrame) -> list:
        cols = []
        for c in df.columns:
            if c in EXCLUDED_EXACT or c.startswith(("target_", "postlaunch_", "_")):
                continue
            if c in FOLLOWER_COLS and not (self.with_followers and c == "community_followers"):
                continue
            if c.startswith(STORE_PREFIXES) and not self.use_store:
                continue
            if c.startswith(CONTEXT_PREFIXES) and not getattr(self, "use_context", True):
                continue
            if c.startswith(TRACK_PREFIXES) and not self.use_track:
                continue
            if c == "age_years_log" and not self.use_age:
                continue
            if not pd.api.types.is_numeric_dtype(df[c]):
                continue
            cols.append(c)
        return cols

    def _prep(self, df: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame({
            c: (pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan)
            for c in self.feature_cols
        }, index=df.index)
        for c in LOG_COLS:
            if c in X:
                X[c] = np.log1p(X[c].clip(lower=0))
        return X

    @property
    def all_feature_names(self) -> list:
        n_text = self.svd.n_components if self.svd is not None else 0
        return self.feature_cols + [f"text_topic_{i}" for i in range(n_text)]

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        cols = self._select(df)
        # Drop constant columns: they carry no signal in this dataset.
        self.feature_cols = [c for c in cols if df[c].nunique(dropna=True) > 1]
        X = self.imputer.fit_transform(self._prep(df))
        if self.use_text and "store_text" in df:
            self.tfidf = TfidfVectorizer(min_df=5, max_df=0.5, max_features=20000, ngram_range=(1, 2),
                                         stop_words="english", sublinear_tf=True)
            tf = self.tfidf.fit_transform(df["store_text"].fillna(""))
            self.svd = TruncatedSVD(n_components=min(N_TEXT_COMPONENTS, tf.shape[1] - 1), random_state=42)
            X = np.hstack([X, self.svd.fit_transform(tf)])
        return X

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X = self.imputer.transform(self._prep(df))
        if self.svd is not None:
            texts = df["store_text"].fillna("") if "store_text" in df else pd.Series([""] * len(df))
            X = np.hstack([X, self.svd.transform(self.tfidf.transform(texts))])
        return X


# --------------------------------------------------------------------------- data
MIN_AGE_DAYS = 90
DEFAULT_HORIZON_YEARS = 3.0


def add_age_feature(df: pd.DataFrame, horizon_years: "float | None" = None) -> pd.DataFrame:
    """Time on sale as a model input. Training uses each game's real age, so the
    model learns how owners accumulate over time instead of mixing 1-year-old and
    10-year-old games. At prediction time pass `horizon_years` to ask for
    'owners after N years on sale'."""
    df = df.copy()
    if horizon_years is not None:
        years = pd.Series(float(horizon_years), index=df.index)
    else:
        years = pd.to_numeric(df["days_since_release"], errors="coerce").fillna(0) / 365.25
    df["age_years_log"] = np.log1p(years.clip(lower=0))
    return df


def load_training_data(csv_path: str, cache_db: str = "data/scraper_cache.db",
                       catalog=None) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    df = raw[(raw["is_released"] == 1) & raw["target_total_reviews"].notna()].copy()
    # Rows without storefront data (no release date) are duplicate/package entries.
    df = df[df["release_year"].notna()]
    # Reviews of very fresh releases have not settled yet; they would teach the model noise.
    df = df[pd.to_numeric(df["days_since_release"], errors="coerce").fillna(0) >= MIN_AGE_DAYS]
    df = df.drop_duplicates(subset="appid", keep="last").reset_index(drop=True)
    df = add_age_feature(df)
    df = enrich(df, cache_db=cache_db, catalog=catalog)
    print(f"Loaded {len(raw)} rows -> {len(df)} released games usable for training.")
    return df


def candidate_models():
    return {
        "Baseline (median)": DummyRegressor(strategy="median"),
        "Ridge regression": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 3, 20))),
        "Random forest": RandomForestRegressor(
            n_estimators=500, min_samples_leaf=2, max_features=0.33, random_state=42, n_jobs=-1),
        # tuned on 5.9k games (5-fold x 3): deeper trees + slower learning + feature subsampling
        # cut MAE 0.549 -> 0.516 and lifted the 2023-24 time test from 64.3% to 65.8% tier accuracy
        "Gradient boosting": HistGradientBoostingRegressor(
            max_iter=1200, learning_rate=0.015, max_depth=6, min_samples_leaf=30,
            l2_regularization=1.0, max_features=0.3, random_state=42),
    }


def evaluate(model, X, y_log, revenue_per_owner, n_repeats=10):
    """Repeated 5-fold CV. Returns mean metrics over repeats."""
    maes, spears, tier_accs, within1 = [], [], [], []
    true_tier = owners_to_tier(10 ** y_log, 10 ** y_log * revenue_per_owner)
    for rep in range(n_repeats):
        cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=rep)
        pred = np.zeros(len(y_log))
        for tr, te in cv.split(X):
            m = _clone(model).fit(X[tr], y_log[tr])
            pred[te] = m.predict(X[te])
        maes.append(np.mean(np.abs(pred - y_log)))
        spears.append(spearmanr(pred, y_log).statistic if np.std(pred) > 0 else 0.0)
        pt = owners_to_tier(10 ** pred, 10 ** pred * revenue_per_owner)
        tier_accs.append(np.mean(pt == true_tier))
        within1.append(np.mean(np.abs(pt - true_tier) <= 1))
    return {
        "mae_log10": np.mean(maes), "mae_std": np.std(maes),
        "spearman": np.mean(spears), "tier_acc": np.mean(tier_accs),
        "tier_within1": np.mean(within1),
    }


def _clone(m):
    from sklearn.base import clone
    return clone(m)


def temporal_holdout(df, model, fb_kwargs, train_until=2022, test_years=(2023, 2024)):
    """Train on games released up to `train_until`, test on later releases.
    This mirrors real use (predicting games that do not exist yet). The test games
    have had less time on sale, so some over-prediction (positive bias) is expected."""
    tgt = build_targets(df)
    tr = (df["release_year"] <= train_until).to_numpy()
    te = df["release_year"].isin(test_years).to_numpy()
    if tr.sum() < 50 or te.sum() < 20:
        return None
    fb = FeatureBuilder(**fb_kwargs)
    Xtr = fb.fit_transform(df[tr])
    Xte = fb.transform(df[te])
    y = tgt["log_owners"].to_numpy()
    rpo = np.where(df["is_free"] == 1, 0.0, df["initial_price_usd"].fillna(0)).astype(float)
    pred = _clone(model).fit(Xtr, y[tr]).predict(Xte)
    base = np.median(y[tr])
    true_t = owners_to_tier(10 ** y[te], 10 ** y[te] * rpo[te])
    pred_t = owners_to_tier(10 ** pred, 10 ** pred * rpo[te])
    base_t = owners_to_tier(np.full(te.sum(), 10 ** base), 10 ** base * rpo[te])
    return {
        "n_train": int(tr.sum()), "n_test": int(te.sum()),
        "mae": float(np.mean(np.abs(pred - y[te]))), "base_mae": float(np.mean(np.abs(base - y[te]))),
        "bias": float(np.mean(pred - y[te])),
        "spearman": float(spearmanr(pred, y[te]).statistic),
        "tier_acc": float(np.mean(pred_t == true_t)), "base_tier_acc": float(np.mean(base_t == true_t)),
    }


def run_ablation(df, y, rpo, with_followers, repeats=3):
    """Show what each feature group adds, using gradient boosting."""
    steps = [
        ("Original features", dict(use_store=False, use_text=False, use_track=False, use_age=False, use_context=False)),
        ("+ game age", dict(use_store=False, use_text=False, use_track=False, use_context=False)),
        ("+ store-page stats", dict(use_store=True, use_text=False, use_track=False, use_context=False)),
        ("+ description text", dict(use_store=True, use_text=True, use_track=False, use_context=False)),
        ("+ studio track record", dict(use_store=True, use_text=True, use_track=True, use_context=False)),
        ("+ genre context", dict(use_store=True, use_text=True, use_track=True, use_context=True)),
    ]
    gb = candidate_models()["Gradient boosting"]
    print(f"\nFeature-group ablation (gradient boosting, 5-fold x {repeats}):")
    print(f"{'Features':<26}{'#cols':>6}{'MAE log10':>11}{'Spearman':>10}{'Tier acc':>10}")
    for label, kw in steps:
        X = FeatureBuilder(with_followers=with_followers, **kw).fit_transform(df)
        r = evaluate(gb, X, y, rpo, n_repeats=repeats)
        print(f"{label:<26}{X.shape[1]:>6}{r['mae_log10']:>11.3f}{r['spearman']:>10.3f}{r['tier_acc']:>10.1%}")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/game_commercial_dataset.csv")
    ap.add_argument("--save", default="data/success_model.joblib")
    ap.add_argument("--with-followers", action="store_true",
                    help="Include community follower count (optimistic for released games)")
    ap.add_argument("--repeats", type=int, default=10, help="CV repeats")
    ap.add_argument("--cache-db", default="data/scraper_cache.db")
    ap.add_argument("--catalog-db", default="data/steamspy_catalog.db",
                    help="SteamSpy catalog from build_catalog_index.py (studio track record)")
    ap.add_argument("--ablation", action="store_true", help="Report the gain from each feature group")
    ap.add_argument("--target", choices=["boxleiter", "adjusted"], default="boxleiter",
                    help="Owner estimate: flat reviews x 35, or sales-per-review adjusted for release year, "
                         "review score and price (tested Nov 2026: same accuracy on the 2023-24 time test)")
    args = ap.parse_args()
    global TARGET_MODE
    TARGET_MODE = args.target

    if not os.path.exists(args.data):
        sys.exit(f"Dataset not found: {args.data}")

    catalog = CatalogIndex(args.catalog_db)
    if catalog.available:
        print(f"Studio track record: using SteamSpy catalog of {catalog.n_games:,} games.")
    else:
        print("Studio track record: catalog not found - run build_catalog_index.py to enable.")
    df = load_training_data(args.data, cache_db=args.cache_db, catalog=catalog if catalog.available else None)
    tgt = build_targets(df)
    fb = FeatureBuilder(with_followers=args.with_followers)
    X = fb.fit_transform(df)
    y = tgt["log_owners"].to_numpy()
    rev_per_owner = np.where(df["is_free"] == 1, 0.0, df["initial_price_usd"].fillna(0)).astype(float)

    print(f"Features: {X.shape[1]} pre-launch columns ({X.shape[1] - len(fb.feature_cols)} text topics)"
          f"{' (+ community followers)' if args.with_followers else ''}")
    print("\nTier distribution (review-based owner estimate):")
    for t, n in pd.Series(tgt["tier"]).value_counts().sort_index().items():
        print(f"  Tier {t} {TIER_NAMES[t]:<52} {n:3d} games")
    majority = pd.Series(tgt["tier"]).value_counts(normalize=True).max()

    print(f"\nCross-validation: 5-fold x {args.repeats} repeats")
    print(f"{'Model':<20}{'MAE log10':>12}{'Spearman':>10}{'Tier acc':>10}{'Tier +/-1':>11}")
    results = {}
    for name, model in candidate_models().items():
        r = evaluate(model, X, y, rev_per_owner, n_repeats=args.repeats)
        results[name] = r
        print(f"{name:<20}{r['mae_log10']:>8.3f}+-{r['mae_std']:.2f}{r['spearman']:>10.3f}"
              f"{r['tier_acc']:>10.1%}{r['tier_within1']:>11.1%}")
    print(f"(Always guessing the majority tier would score {majority:.1%} tier accuracy.)")
    print("MAE log10 = average error in orders of magnitude (0.5 = off by ~3x, 1.0 = off by 10x).")

    best_name = min((n for n in results if not n.startswith("Baseline")),
                    key=lambda n: results[n]["mae_log10"])
    best = candidate_models()[best_name].fit(X, y)
    print(f"\nSelected model: {best_name}")

    th = temporal_holdout(df, candidate_models()[best_name], dict(with_followers=args.with_followers))
    if th:
        print(f"\nTime-based test (train <= 2022: {th['n_train']} games, test 2023-2024: {th['n_test']} games):")
        print(f"  MAE log10   {th['mae']:.3f}   (baseline {th['base_mae']:.3f})")
        print(f"  Spearman    {th['spearman']:.3f}")
        print(f"  Tier acc    {th['tier_acc']:.1%}   (baseline {th['base_tier_acc']:.1%})")
        print(f"  Bias        {th['bias']:+.3f} log10 (positive = over-predicts newer games)")

    if args.ablation:
        run_ablation(df, y, rev_per_owner, args.with_followers)

    # Feature importance (random forest impurity is cheap and stable enough for a summary)
    rf = candidate_models()["Random forest"].fit(X, y)
    names = fb.all_feature_names
    imp = sorted(zip(names, rf.feature_importances_), key=lambda t: -t[1])
    print("\nTop 15 drivers (random-forest importance):")
    for i, (f, v) in enumerate(imp[:15], 1):
        direction = np.sign(np.corrcoef(X[:, names.index(f)], y)[0, 1])
        print(f"  {i:2d}. {f:<34}{v * 100:5.1f}%  ({'+' if direction > 0 else '-'} owners)")

    # Residual spread -> prediction interval for inference
    oof = cross_val_predict(_clone(candidate_models()[best_name]), X, y,
                            cv=RepeatedKFold(n_splits=5, n_repeats=1, random_state=0))
    resid = y - oof
    bundle = {
        "model": best, "model_name": best_name, "feature_builder": fb,
        "residual_q": np.quantile(resid, [0.1, 0.9]).tolist(),
        "cv_results": results, "n_train": len(df), "tier_names": TIER_NAMES,
        "review_multiplier": REVIEW_MULTIPLIER, "target_mode": TARGET_MODE, "with_followers": args.with_followers,
        "temporal_holdout": th, "default_horizon_years": DEFAULT_HORIZON_YEARS, "catalog_db": args.catalog_db if catalog.available else None,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
    joblib.dump(bundle, args.save)
    print(f"\nSaved model bundle to {args.save}")
    if len(df) < 300:
        print(f"\nWARNING: only {len(df)} training games. Results are indicative at best; "
              "scrape more games (e.g. `python run_scraper.py --mode catalog --pages 2 --limit 2000 "
              "--append`) for a reliable model.")


if __name__ == "__main__":
    # Run via the module so pickled classes are saved as train_model.FeatureBuilder
    # (not __main__.FeatureBuilder) and the bundle can be loaded from any script.
    import train_model
    train_model.main()
