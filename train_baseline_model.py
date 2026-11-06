"""
Baseline & Production Predictive AI Model for Video Game Commercial Success.

Features:
1. Data Leakage Prevention: Filters out post-launch metrics (reviews, days since release)
   to ensure the model is fit for pre-launch evaluation.
2. Target Filtering: Automatically filters out unreleased games and games with unverified sales.
3. Feature Preprocessing: Median imputation and log1p transforms for heavy-tailed counts.
4. Multi-Algorithm Support: Random Forest and Histogram-based Gradient Boosting (LightGBM equivalent).
5. Comprehensive Evaluation: Stratified validation, classification metrics, confusion matrix,
   and feature importance rankings.
6. Model Persistence: Saves complete inference bundle (model, imputer, transform configs).
"""

import sys
import os
import argparse
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance


METADATA_COLS = [
    "appid", "name", "release_raw", "primary_tag", "top_5_tags_str", "publisher_tier"
]

# Post-launch metrics that must NEVER be used when predicting commercial success prior to release
POSTLAUNCH_LEAK_COLS = [
    "days_since_release",
    "follower_to_review_ratio",
    "followers_per_day_since_release",
    "postlaunch_follower_to_review_ratio",
    "postlaunch_followers_per_day",
]

SKEWED_COUNT_COLS = [
    "community_followers",
    "estimated_wishlists",
    "follower_velocity_7d",
    "wishlist_velocity_7d",
    "follower_velocity_total",
    "wishlist_velocity_total",
    "dlc_count",
    "achievements_count",
]


class GameDataPreprocessor:
    """Handles feature selection, log transforms, and imputation cleanly."""

    def __init__(self, log_cols: Optional[List[str]] = None):
        self.log_cols = log_cols or SKEWED_COUNT_COLS
        self.imputer = SimpleImputer(strategy="median")
        self.feature_cols: List[str] = []
        self.fitted_log_cols: List[str] = []

    def fit_transform(self, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
        self.feature_cols = feature_cols
        self.fitted_log_cols = [c for c in self.log_cols if c in self.feature_cols]

        X_df = df[self.feature_cols].copy()

        # Apply log1p transform to heavy-tailed count columns
        for col in self.fitted_log_cols:
            X_df[col] = np.log1p(np.maximum(0, pd.to_numeric(X_df[col], errors="coerce").fillna(0)))

        # Convert remaining object or boolean columns to float
        for col in self.feature_cols:
            X_df[col] = pd.to_numeric(X_df[col], errors="coerce")

        X_arr = self.imputer.fit_transform(X_df)
        return X_arr

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X_df = df[self.feature_cols].copy()
        for col in self.fitted_log_cols:
            if col in X_df.columns:
                X_df[col] = np.log1p(np.maximum(0, pd.to_numeric(X_df[col], errors="coerce").fillna(0)))

        for col in self.feature_cols:
            if col in X_df.columns:
                X_df[col] = pd.to_numeric(X_df[col], errors="coerce")
            else:
                X_df[col] = 0.0

        return self.imputer.transform(X_df[self.feature_cols])


def load_and_prep_data(csv_path: str, allow_postlaunch: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, List[str]]:
    """
    Load dataset and separate features from targets.
    Filters out unreleased games and rows with missing target tiers.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset '{csv_path}' not found.")

    raw_df = pd.read_csv(csv_path)
    total_raw = len(raw_df)

    # 1. Filter out unreleased games and games with unknown target tiers
    if "is_released" in raw_df.columns:
        released_mask = (raw_df["is_released"] == 1)
    else:
        released_mask = pd.Series(True, index=raw_df.index)

    target_valid_mask = raw_df["target_success_tier"].notna()
    train_mask = released_mask & target_valid_mask

    df = raw_df[train_mask].copy()
    dropped_count = total_raw - len(df)
    if dropped_count > 0:
        print(f"Filtered out {dropped_count} unreleased / unverified games (kept {len(df)} released games with targets).")

    if len(df) == 0:
        raise ValueError("No released games with valid target tiers found in the dataset.")

    # 2. Separate input features X from target and leak columns
    target_cols = [c for c in df.columns if c.startswith("target_")]
    excluded_cols = set(METADATA_COLS + target_cols)

    if not allow_postlaunch:
        excluded_cols.update(POSTLAUNCH_LEAK_COLS)
        for c in df.columns:
            if c.startswith("postlaunch_"):
                excluded_cols.add(c)

    feature_cols = [c for c in df.columns if c not in excluded_cols]

    y = df["target_success_tier"].astype(int)
    return raw_df, df, y, feature_cols


def train_and_evaluate_model(
    df: pd.DataFrame,
    y: pd.Series,
    feature_cols: List[str],
    model_type: str = "rf",
    save_path: Optional[str] = "data/trained_model.joblib"
):
    """
    Train and evaluate classification model.
    """
    preprocessor = GameDataPreprocessor()
    X = preprocessor.fit_transform(df, feature_cols)

    print("\n" + "=" * 65)
    print(f"TRAINING AI MODEL: {model_type.upper()} ({len(X)} games, {len(feature_cols)} input features)")
    print("=" * 65)

    # Class distribution check
    counts = y.value_counts().to_dict()
    print("Commercial Tier Distribution:")
    tier_names = {
        0: "Tier 0 (Niche / <$25k)",
        1: "Tier 1 (Moderate Indie / $25k-$250k)",
        2: "Tier 2 (Commercial Hit / $250k-$5M)",
        3: "Tier 3 (Blockbuster / >$5M)",
    }
    for t in sorted(counts.keys()):
        print(f"  {tier_names.get(t, f'Tier {t}')}: {counts[t]} games")

    # Split strategy
    test_size = 0.25 if len(X) >= 8 else 0.5
    stratify = y if min(counts.values()) >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify
    )

    if model_type == "hgb":
        clf = HistGradientBoostingClassifier(max_iter=100, max_depth=5, random_state=42)
    else:
        clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    print("\n" + "-" * 65)
    print(f"TEST ACCURACY: {test_acc * 100:.2f}% (Tested on {len(y_test)} games)")
    print("-" * 65)

    # Cross-validation if enough data
    if len(X) >= 15 and (stratify is not None):
        cv = StratifiedKFold(n_splits=min(5, min(counts.values())), shuffle=True, random_state=42)
        cv_scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
        print(f"Cross-Validation Accuracy: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")

    # Feature Importance Analysis
    print("\nTOP 15 MOST PREDICTIVE PRE-LAUNCH FEATURES:")
    print("-" * 65)
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    else:
        perm = permutation_importance(clf, X_test, y_test, n_repeats=5, random_state=42)
        importances = perm.importances_mean

    feat_imp = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
    for rank, (feat, imp) in enumerate(feat_imp[:15], 1):
        print(f" {rank:2d}. {feat:<35} {max(0.0, imp) * 100:6.2f}%")
    print("-" * 65)

    # Save model bundle
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        bundle = {
            "model": clf,
            "preprocessor": preprocessor,
            "feature_cols": feature_cols,
            "model_type": model_type,
            "tier_names": tier_names,
        }
        joblib.dump(bundle, save_path)
        print(f"\nTrained model bundle successfully saved to: {save_path}")

    return clf, preprocessor, feat_imp


def main():
    parser = argparse.ArgumentParser(description="Train Predictive AI Model on Commercial Game Dataset.")
    parser.add_argument("--data", default="data/game_commercial_dataset.csv", help="Input dataset CSV")
    parser.add_argument("--model", choices=["rf", "hgb"], default="rf", help="Model: 'rf' (Random Forest) or 'hgb' (HistGradientBoosting)")
    parser.add_argument("--save", default="data/trained_model.joblib", help="Model output file path")
    parser.add_argument("--allow-postlaunch", action="store_true", help="Include post-launch features (warning: causes pre-launch leakage)")
    args = parser.parse_args()

    try:
        raw_df, df_train, y, feat_names = load_and_prep_data(args.data, allow_postlaunch=args.allow_postlaunch)
        train_and_evaluate_model(df_train, y, feat_names, model_type=args.model, save_path=args.save)
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
