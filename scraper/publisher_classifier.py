"""
Publisher and Developer Classification Module.

Classifies game publishers and developers into industry tiers:
- AAA (Major global publishers / multinational corporations)
- AA / Major Indie (Established boutique publishers with substantial marketing budgets)
- Indie (Independent self-published creators or small indie publishers)
"""

import re
from typing import List, Dict, Any, Tuple

# Curated list of AAA publishers and major multinational corporate entities
AAA_PUBLISHERS = {
    "electronic arts", "ea", "ea sports", "ea games", "ubisoft",
    "sony interactive entertainment", "playstation pc llc", "playstation publishing llc",
    "microsoft", "xbox game studios", "bethesda softworks", "bethesda",
    "activision", "activision blizzard", "blizzard entertainment",
    "take-two interactive", "2k", "2k games", "rockstar games", "private division",
    "square enix", "capcom", "bandai namco entertainment", "bandai namco",
    "sega", "atlus", "konami", "koei tecmo", "koei tecmo games",
    "warner bros. games", "warner bros. interactive entertainment", "wb games",
    "cd projekt red", "cd projekt", "valve",
    "tencent", "netease games", "epic games", "riot games",
    "krafton", "ncsoft", "nexon", "pearl abyss",
    "deep silver", "plaion", "embracer group",
}

# Curated list of AA & Major Indie boutique publishers
MAJOR_INDIE_PUBLISHERS = {
    "devolver digital", "annapurna interactive", "team17", "team17 digital",
    "raw fury", "tinybuild", "focus entertainment", "focus home interactive",
    "paradox interactive", "thq nordic", "505 games", "curve games",
    "curve digital", "fellow traveller", "hooded horse", "kepler interactive",
    "playstack", "chucklefish", "finji", "humble games", "frontier developments",
    "daedalic entertainment", "coffee stain publishing", "rebellion",
    "plug in digital", "dear villagers", "newzoo", "headup", "headup games",
    "graffiti games", "modio", "merge games", "crunchyroll games",
    "kakehashi games", "playway s.a.", "shunka", "bigmode",
    "sabotage studio", "ghost ship publishing", "superhot team",
    "panic", "no more robots", "future friends games", "whitethorn games",
    "akupara games", "secret mode", "serenity forge", "modus games",
    "silver lining interactive", "wired productions", "neon doctrine",
}


def normalize_name(name: str) -> str:
    """Normalize company name for fuzzy matching."""
    if not name:
        return ""
    clean = name.lower()
    # Strip common suffixes like inc, ltd, llc, co, gmbh, corp
    clean = re.sub(r"\b(inc|ltd|llc|co|corp|gmbh|s\.a\.|corporation|entertainment|interactive|games|studios|studio)\b", "", clean)
    clean = re.sub(r"[^\w\s]", " ", clean)
    clean = " ".join(clean.split())
    return clean


def classify_publisher(
    publishers: List[str],
    developers: List[str]
) -> Dict[str, Any]:
    """
    Classify publisher tier and attributes for a game.

    Returns:
        Dict with:
            - publisher_tier: "AAA", "AA_Major_Indie", or "Indie"
            - is_aaa: int (0 or 1)
            - is_major_indie: int (0 or 1)
            - is_self_published: int (0 or 1)
            - publisher_count: int
            - developer_count: int
    """
    pubs = [p.strip() for p in (publishers or []) if p and p.strip()]
    devs = [d.strip() for d in (developers or []) if d and d.strip()]

    pubs_lower = [p.lower() for p in pubs]
    pubs_norm = [normalize_name(p) for p in pubs]

    devs_lower = [d.lower() for d in devs]
    devs_norm = [normalize_name(d) for d in devs]

    # Check AAA
    is_aaa = False
    for p_raw, p_norm in zip(pubs_lower, pubs_norm):
        if p_raw in AAA_PUBLISHERS:
            is_aaa = True
            break
        for aaa in AAA_PUBLISHERS:
            if aaa in p_raw or (len(p_norm) >= 3 and normalize_name(aaa) == p_norm):
                is_aaa = True
                break
        if is_aaa:
            break

    # If developer is a known AAA studio (e.g. Rockstar North, EA DICE)
    if not is_aaa:
        for d_raw, d_norm in zip(devs_lower, devs_norm):
            for aaa in AAA_PUBLISHERS:
                if aaa in d_raw or (len(d_norm) >= 3 and normalize_name(aaa) == d_norm):
                    is_aaa = True
                    break
            if is_aaa:
                break

    # Check AA / Major Indie
    is_major_indie = False
    if not is_aaa:
        for p_raw, p_norm in zip(pubs_lower, pubs_norm):
            if p_raw in MAJOR_INDIE_PUBLISHERS:
                is_major_indie = True
                break
            for mi in MAJOR_INDIE_PUBLISHERS:
                if mi in p_raw or (len(p_norm) >= 3 and normalize_name(mi) == p_norm):
                    is_major_indie = True
                    break
            if is_major_indie:
                break

    # Determine self-published flag
    is_self_published = False
    if not pubs:
        is_self_published = True
    else:
        # Check intersection between developer and publisher names
        pub_set = set(pubs_norm)
        dev_set = set(devs_norm)
        if pub_set and dev_set and (pub_set.intersection(dev_set) or any(d in p or p in d for d in devs_norm for p in pubs_norm if d and p)):
            is_self_published = True

    # Assign overall tier
    if is_aaa:
        tier = "AAA"
    elif is_major_indie:
        tier = "AA_Major_Indie"
    else:
        tier = "Indie"

    return {
        "publisher_tier": tier,
        "is_aaa": 1 if is_aaa else 0,
        "is_major_indie": 1 if is_major_indie else 0,
        "is_self_published": 1 if is_self_published else 0,
        "publisher_count": len(pubs),
        "developer_count": len(devs),
    }
