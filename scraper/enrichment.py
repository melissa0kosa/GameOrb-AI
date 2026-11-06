"""
Extra pre-launch features computed from data already on disk.

1. Store-page content (from the storefront responses in data/scraper_cache.db):
   screenshot / trailer counts, description length and richness, hardware
   requirements, plus the raw description text for the text model.

2. Developer / publisher track record (from data/steamspy_catalog.db, built by
   build_catalog_index.py): how the studio's *earlier* games sold. "Earlier"
   means a lower AppID - Steam assigns AppIDs sequentially, so this is a good
   proxy for "existed before this game" and never includes the game itself.

Both are pure functions of a DataFrame with an `appid` column, so training and
inference share exactly the same code path.
"""

import json
import os
import re
import sqlite3
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

REVIEW_MULTIPLIER = 35.0

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_CORP_SUFFIXES = {
    "inc", "inc.", "ltd", "ltd.", "llc", "llc.", "co.", "co", "corp", "corp.", "gmbh",
    "s.a.", "sa", "s.r.o.", "srl", "s.l.", "ab", "oy", "bv", "b.v.", "pty", "limited",
    "co., ltd.", "k.k.", "plc", "ltda", "s.a.s.", "sas", "as", "a/s", "aps",
}


# --------------------------------------------------------------------------- helpers
def strip_html(html: Optional[str]) -> str:
    if not html or not isinstance(html, str):
        return ""
    return _WS_RE.sub(" ", _TAG_RE.sub(" ", html)).strip()


def _size_gb(text: str, labels: str) -> Optional[float]:
    m = re.search(rf"(?:{labels})\s*:?\s*([\d.,]+)\s*(GB|MB|TB)", text, re.I)
    if not m:
        return None
    try:
        val = float(m.group(1).replace(",", "."))
    except ValueError:
        return None
    unit = m.group(2).upper()
    return val / 1024 if unit == "MB" else val * 1024 if unit == "TB" else val


def normalize_company(name: str) -> str:
    n = name.lower().strip().strip(",").strip()
    n = re.sub(r"[™®©]", "", n)
    n = re.sub(r"\s+", " ", n)
    return n


def split_companies(raw) -> List[str]:
    """Split SteamSpy's comma-joined company string (or a store list) into names,
    without breaking 'Studio, Inc.' style suffixes into separate companies."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    joined = ",".join(str(x) for x in raw) if isinstance(raw, list) else str(raw)
    parts = joined.split(",")
    out: List[str] = []
    for p in parts:
        p = normalize_company(p).rstrip(".")
        if not p:
            continue
        if (p in _CORP_SUFFIXES or p + "." in _CORP_SUFFIXES) and out:
            continue  # suffix of the previous company ("Studio, Inc.")
        p = re.sub(r"\s+(inc|ltd|llc|co|corp|gmbh|limited|s\.a|s\.r\.o|ab|oy)\.?$", "", p).strip(" ,.")
        if p and p not in out:
            out.append(p)
    return out


# --------------------------------------------------------------------------- store content
def _load_storefront(cache_db: str, appids: List[int]) -> Dict[int, dict]:
    if not os.path.exists(cache_db):
        return {}
    conn = sqlite3.connect(cache_db)
    try:
        out = {}
        ids = [int(a) for a in appids]
        for i in range(0, len(ids), 500):
            chunk = ids[i:i + 500]
            q = f"SELECT appid, data_json FROM response_cache WHERE endpoint='storefront' AND appid IN ({','.join('?' * len(chunk))})"
            for aid, js in conn.execute(q, chunk):
                try:
                    out[int(aid)] = json.loads(js)
                except json.JSONDecodeError:
                    pass
        return out
    finally:
        conn.close()


def store_content_features(d: Optional[dict]) -> dict:
    d = d or {}
    about = strip_html(d.get("about_the_game") or d.get("detailed_description"))
    short = strip_html(d.get("short_description"))
    detailed_html = d.get("detailed_description") or ""
    req = d.get("pc_requirements")
    req_text = strip_html(req.get("minimum", "")) if isinstance(req, dict) else ""
    rec_text = strip_html(req.get("recommended", "")) if isinstance(req, dict) else ""
    return {
        "store_n_screenshots": len(d.get("screenshots") or []),
        "store_n_trailers": len(d.get("movies") or []),
        "store_about_words": len(about.split()),
        "store_short_desc_words": len(short.split()),
        "store_desc_images": detailed_html.count("<img"),
        "store_desc_gifs": detailed_html.lower().count(".gif"),
        "store_desc_headers": len(re.findall(r"<h[1-3]|bb_tag", detailed_html)),
        "store_has_website": int(bool(d.get("website"))),
        "store_has_support_email": int(bool((d.get("support_info") or {}).get("email"))),
        "store_has_recommended_specs": int(bool(rec_text)),
        "req_min_ram_gb": _size_gb(req_text, r"Memory|RAM"),
        "req_min_storage_gb": _size_gb(req_text, r"Storage|Hard Drive|Hard Disk Space|Disk Space|HDD|Space"),
        "store_text": f"{short} {about}",
    }


def add_store_content_features(df: pd.DataFrame, cache_db: str = "data/scraper_cache.db") -> pd.DataFrame:
    store = _load_storefront(cache_db, df["appid"].tolist())
    feats = pd.DataFrame([store_content_features(store.get(int(a))) for a in df["appid"]], index=df.index)
    feats["_developers"] = [store.get(int(a), {}).get("developers") or [] for a in df["appid"]]
    feats["_publishers"] = [store.get(int(a), {}).get("publishers") or [] for a in df["appid"]]
    return pd.concat([df.drop(columns=[c for c in feats.columns if c in df.columns]), feats], axis=1)


# --------------------------------------------------------------------------- track record
class CatalogIndex:
    """Company -> sorted list of (appid, log10 owners) from the SteamSpy catalog."""

    def __init__(self, catalog_db: str = "data/steamspy_catalog.db"):
        self.available = os.path.exists(catalog_db)
        self.dev: Dict[str, list] = defaultdict(list)
        self.pub: Dict[str, list] = defaultdict(list)
        self.n_games = 0
        if not self.available:
            return
        conn = sqlite3.connect(catalog_db)
        try:
            rows = conn.execute("SELECT appid, developer, publisher, positive, negative FROM catalog").fetchall()
        finally:
            conn.close()
        self.n_games = len(rows)
        for appid, dev, pub, pos, neg in rows:
            lo = float(np.log10(max((pos or 0) + (neg or 0), 1) * REVIEW_MULTIPLIER))
            for c in split_companies(dev):
                self.dev[c].append((appid, lo))
            for c in split_companies(pub):
                self.pub[c].append((appid, lo))
        for idx in (self.dev, self.pub):
            for k in idx:
                idx[k].sort()

    @staticmethod
    def _history(index: Dict[str, list], companies: List[str], appid: int) -> dict:
        prior = {}
        for c in companies:
            for aid, lo in index.get(c, []):
                if aid >= appid:
                    break
                prior[aid] = lo
        vals = np.array(list(prior.values()))
        if len(vals) == 0:
            return {"n": 0, "max": 0.0, "mean": 0.0, "hits": 0, "last": 0.0}
        last = prior[max(prior)]
        return {
            "n": len(vals), "max": float(vals.max()), "mean": float(vals.mean()),
            "hits": int((vals >= 5).sum()),  # >= 100k estimated owners
            "last": last,
        }

    def features(self, appid: int, developers: List[str], publishers: List[str]) -> dict:
        devs = split_companies(developers)
        pubs = split_companies(publishers)
        d = self._history(self.dev, devs, int(appid))
        p = self._history(self.pub, pubs, int(appid))
        return {
            "dev_prior_games": d["n"], "dev_prior_max_log_owners": d["max"],
            "dev_prior_mean_log_owners": d["mean"], "dev_prior_hits": d["hits"],
            "dev_last_game_log_owners": d["last"],
            "pub_prior_games": p["n"], "pub_prior_max_log_owners": p["max"],
            "pub_prior_mean_log_owners": p["mean"], "pub_prior_hits": p["hits"],
            "is_debut_developer": int(d["n"] == 0),
        }


def add_track_record_features(df: pd.DataFrame, index: CatalogIndex) -> pd.DataFrame:
    if not index.available:
        return df
    feats = pd.DataFrame(
        [index.features(a, devs, pubs) for a, devs, pubs in zip(df["appid"], df["_developers"], df["_publishers"])],
        index=df.index,
    )
    return pd.concat([df.drop(columns=[c for c in feats.columns if c in df.columns]), feats], axis=1)


def enrich(df: pd.DataFrame, cache_db: str = "data/scraper_cache.db",
           catalog: Optional[CatalogIndex] = None) -> pd.DataFrame:
    df = add_store_content_features(df, cache_db)
    if catalog is not None:
        df = add_track_record_features(df, catalog)
    return df
