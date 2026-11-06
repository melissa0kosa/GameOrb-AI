"""
Steam Game Success Predictor - Streamlit front end.

Run locally:   streamlit run app.py
"""

import threading
import time
from collections import deque

import altair as alt
import pandas as pd
import streamlit as st

from predictor import (FEATURE_LABELS, LANGUAGE_LABELS, STORE_GENRES, TAGS, SuccessPredictor,
                       parse_appid, tag_column)

st.set_page_config(page_title="Steam Game Success Predictor", page_icon="🎮", layout="wide")

# Chart colors: validated reference palette (blue / red diverging pair, light surface)
BLUE, BLUE_LIGHT, RED, GRID = "#2a78d6", "#cde2fb", "#e34948", "#e4e3df"
LOOKUPS_PER_MINUTE = 10          # global cap on live Steam lookups (Steam throttles hard)
SESSION_COOLDOWN_S = 5


# ---------------------------------------------------------------------------- resources
@st.cache_resource(show_spinner="Loading model...")
def get_predictor() -> SuccessPredictor:
    return SuccessPredictor()


@st.cache_resource
def lookup_limiter():
    return {"lock": threading.Lock(), "times": deque()}


def allow_lookup() -> bool:
    lim = lookup_limiter()
    now = time.time()
    with lim["lock"]:
        while lim["times"] and now - lim["times"][0] > 60:
            lim["times"].popleft()
        if len(lim["times"]) >= LOOKUPS_PER_MINUTE:
            return False
        lim["times"].append(now)
    return True


@st.cache_data(ttl=24 * 3600, show_spinner=False, max_entries=2000)
def fetch_game_cached(appid: int):
    return get_predictor().fetch_appid(appid)


# ---------------------------------------------------------------------------- formatting
def _sig2(x: float) -> float:
    """Round to 2 significant figures - the model is not precise enough for more."""
    if x <= 0:
        return 0.0
    from math import floor, log10
    return round(x, -int(floor(log10(x))) + 1)


def fmt_count(x: float) -> str:
    if x is None:
        return "-"
    x = _sig2(x)
    for unit, suffix in ((1e9, "B"), (1e6, "M")):
        if x >= unit:
            v = x / unit
            return f"{v:.0f}{suffix}" if v >= 10 else f"{v:.1f}{suffix}"
    if x >= 1e4:
        return f"{x / 1e3:.0f}k"
    return f"{x:,.0f}"


def fmt_money(x: float) -> str:
    return "-" if x is None else "$" + fmt_count(x)


TIER_SHORT = {0: "Niche", 1: "Sustainable indie", 2: "Commercial hit", 3: "Blockbuster"}


# ---------------------------------------------------------------------------- result view
def curve_chart(res: dict) -> alt.Chart:
    lo_f = res["owners_lo"] / res["owners"]
    hi_f = res["owners_hi"] / res["owners"]
    df = pd.DataFrame([{"Years on sale": y, "Owners": o, "Low (10%)": o * lo_f, "High (90%)": o * hi_f}
                       for y, o in res["curve"]])
    lo, hi = df["Low (10%)"].min(), df["High (90%)"].max()
    ticks = [10 ** k for k in range(0, 11) if lo / 10 <= 10 ** k <= hi * 10]
    base = alt.Chart(df).encode(x=alt.X("Years on sale:Q", axis=alt.Axis(grid=False, tickCount=6)))
    band = base.mark_area(color=BLUE_LIGHT, opacity=0.6).encode(
        y=alt.Y("Low (10%):Q", scale=alt.Scale(type="log"), title="Estimated owners (log scale)",
                axis=alt.Axis(gridColor=GRID, format="~s", values=ticks)),
        y2="High (90%):Q")
    line = base.mark_line(color=BLUE, strokeWidth=2).encode(y="Owners:Q")
    pts = base.mark_circle(color=BLUE, size=70, opacity=1).encode(
        y="Owners:Q",
        tooltip=[alt.Tooltip("Years on sale:Q"), alt.Tooltip("Owners:Q", format=",.0f"),
                 alt.Tooltip("Low (10%):Q", format=",.0f"), alt.Tooltip("High (90%):Q", format=",.0f")])
    return (band + line + pts).properties(height=280).configure_view(stroke=None)


def what_if_chart(rows: list) -> alt.Chart:
    df = pd.DataFrame(rows)
    df["pct"] = (df["factor"] - 1) * 100
    df["Direction"] = df["pct"].map(lambda v: "Higher" if v >= 0 else "Lower")
    y = alt.Y("change:N", sort=None, title=None,
              scale=alt.Scale(paddingInner=0.35, paddingOuter=0.2), axis=alt.Axis(labelLimit=320, labelFontSize=12))
    bars = alt.Chart(df).mark_bar(cornerRadiusEnd=4).encode(
        y=y,
        x=alt.X("pct:Q", title="Change in predicted owners (%)", axis=alt.Axis(gridColor=GRID, format="+.0f")),
        color=alt.Color("Direction:N", scale=alt.Scale(domain=["Higher", "Lower"], range=[BLUE, RED]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("change:N", title="Change"), alt.Tooltip("pct:Q", title="Owners", format="+.0f")])
    zero = alt.Chart(pd.DataFrame({"z": [0]})).mark_rule(color="#52514e", strokeWidth=1).encode(
        x=alt.X("z:Q", title="Change in predicted owners (%)"))
    return (bars + zero).properties(height=40 * len(df) + 20).configure_view(stroke=None)


def show_result(res: dict, df: pd.DataFrame, name: str):
    p = get_predictor()
    h = res["horizon_years"]
    st.subheader(f"Forecast for {name}")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Owners after {h:g} year{'s' if h != 1 else ''}", fmt_count(res["owners"]),
              help="Estimated lifetime owners at this point (reviews x 35 method).")
    c1.caption(f"80% range: {fmt_count(res['owners_lo'])} - {fmt_count(res['owners_hi'])}")
    if res["is_free"]:
        c2.metric("Gross revenue", "Free-to-play")
        c2.caption("Revenue from in-game purchases is not modeled.")
    elif res["revenue"] is None:
        c2.metric("Gross revenue", "Price not set")
    else:
        c2.metric("Gross revenue (before Steam's cut)", fmt_money(res["revenue"]))
        c2.caption(f"80% range: {fmt_money(res['revenue_lo'])} - {fmt_money(res['revenue_hi'])}")
    c3.metric("Commercial tier", f"Tier {res['tier']}: {TIER_SHORT[res['tier']]}")
    c3.caption(res["tier_name"])

    st.markdown("**Owners over time** - line = best estimate, band = 80% range")
    st.altair_chart(curve_chart(res), use_container_width=True)
    with st.expander("Show as table"):
        st.dataframe(pd.DataFrame([{"Years on sale": y, "Estimated owners": round(o)} for y, o in res["curve"]]),
                     hide_index=True)

    if res["studio"]:
        s = res["studio"]
        st.markdown("**Studio track record** (earlier games in the SteamSpy catalog)")
        a, b = st.columns(2)
        a.write(f"Developer: **{int(s['dev_prior_games'])}** earlier games, "
                f"**{int(s['dev_prior_hits'])}** with 100k+ owners"
                + ("" if s["dev_prior_games"] else " - first game (or name not found)"))
        b.write(f"Publisher: **{int(s['pub_prior_games'])}** earlier games, "
                f"**{int(s['pub_prior_hits'])}** with 100k+ owners")

    if res["hype"]:
        hy = res["hype"]
        st.markdown(f"**With pre-launch hype** (Stage 2, trained on {hy['n_train']} tracked launches)")
        st.write(f"{hy['followers']:,} followers now. Owners after {hy['horizon_days']:g} days: "
                 f"**{fmt_count(hy['owners'])}** (80% range {fmt_count(hy['owners_lo'])} - "
                 f"{fmt_count(hy['owners_hi'])}), x{hy['factor']:.2f} vs. the store-page-only estimate.")

    rows = [r for r in p.what_if(df, h) if abs(r["factor"] - 1) >= 0.03]
    st.markdown("**What-if: changes associated with different sales**")
    if rows:
        st.altair_chart(what_if_chart(rows), use_container_width=True)
        st.caption("These are patterns the model learned from existing games, not guaranteed effects. "
                   "Bigger games tend to have more languages and achievements, for example, so part of "
                   "the effect is correlation.")
        with st.expander("Show as table"):
            st.dataframe(pd.DataFrame([{"Change": r["change"], "Change in owners": f"{(r['factor'] - 1) * 100:+.0f}%"}
                                       for r in rows]), hide_index=True)
    else:
        st.caption("None of the standard changes moves this prediction by 3% or more.")


# ---------------------------------------------------------------------------- page
p = get_predictor()
info = p.info

st.title("🎮 Steam Game Success Predictor")
st.write("Estimate how many people will own a Steam game, its gross revenue and its commercial tier - "
         "from its store page, or from a description of a game you are still planning.")
st.info(f"Typical error is about {10 ** info['mae_log10']:.1f}x either way, so read results as a rough "
        "order of magnitude, not a sales promise. The 80% range matters more than the single number.",
        icon="ℹ️")

horizon = st.sidebar.slider("Forecast horizon (years on sale)", 0.5, 5.0, float(info["default_horizon"]), 0.5)
st.sidebar.markdown("---")
st.sidebar.caption(f"Model: {info['model_name']}, trained on {info['n_train']:,} Steam games.")

tab_lookup, tab_plan, tab_about = st.tabs(["🔎 Look up a Steam game", "🛠️ Plan a new game", "📊 About the model"])

# ---- Tab 1: Steam lookup
with tab_lookup:
    st.write("Paste a Steam AppID or store link. Works for released and upcoming games.")
    examples = {"Baldur's Gate 3": "1086940", "Terraria": "105600", "Drawquarium (upcoming)": "4045480"}
    def _pick_example():
        choice = st.session_state.get("example", "-")
        if choice != "-":
            st.session_state["lookup_text"] = examples[choice]
            st.session_state["trigger_lookup"] = True

    cols = st.columns([3, 1])
    text = cols[0].text_input("AppID or store URL", key="lookup_text",
                              placeholder="e.g. 1086940 or https://store.steampowered.com/app/1086940/")
    cols[1].selectbox("...or try an example", ["-"] + list(examples), key="example", on_change=_pick_example)
    clicked = st.button("Predict", type="primary", key="lookup_btn")
    if clicked or st.session_state.pop("trigger_lookup", False):
        appid = parse_appid(text)
        if appid is None:
            st.error("That doesn't look like a Steam AppID or store link.")
        else:
            last = st.session_state.get("last_lookup", 0)
            if time.time() - last < SESSION_COOLDOWN_S:
                st.warning("Please wait a few seconds between lookups.")
            elif not allow_lookup():
                st.warning("Too many lookups right now (Steam limits how fast we can ask). Try again in a minute.")
            else:
                st.session_state["last_lookup"] = time.time()
                with st.spinner("Reading the Steam store page..."):
                    try:
                        game = fetch_game_cached(appid)
                    except Exception as e:  # network trouble, Steam outage...
                        game = None
                        st.error(f"Could not reach Steam: {e}")
                if not game:
                    st.error("No Steam data found for that AppID (it may be region-locked, removed or not a game).")
                else:
                    df = p.frame_from_game(game)
                    res = p.predict(df, horizon, appid=appid)
                    st.session_state["lookup_result"] = (res, df, game)
    if "lookup_result" in st.session_state:
        res, df, game = st.session_state["lookup_result"]
        if res["horizon_years"] != horizon:  # slider moved: recompute without re-fetching
            res = p.predict(df, horizon, appid=int(game["appid"]))
        g = game
        st.caption(f"{'Released' if g.get('is_released') else 'Upcoming'} · {g.get('release_raw') or 'date TBA'} · "
                   f"{g.get('publisher_tier', '')} publisher · {g.get('language_count', 0)} languages · "
                   f"tags: {g.get('top_5_tags_str') or 'n/a'}")
        if g.get("is_released"):
            st.caption("This game is already out: the forecast uses only what was knowable before launch, "
                       "so it may differ a lot from its real sales.")
        show_result(res, df, g.get("name", "this game"))

# ---- Tab 2: manual planner
with tab_plan:
    st.write("Describe a game that isn't on Steam yet. Every field is optional; defaults are typical indie values.")
    model_cols = set(p.fb.feature_cols)
    tag_options = [t for t in TAGS if tag_column(t) in model_cols]
    genre_options = [g for g in STORE_GENRES if f"genre_store_{g.replace(' ', '_')}" in model_cols]
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
              "October", "November", "December"]
    with st.form("plan"):
        a, b, c = st.columns(3)
        name = a.text_input("Game name", "My game")
        developer = b.text_input("Developer", help="Separate several with ';'. Used to look up the studio's earlier games.")
        publisher = c.text_input("Publisher", help="Leave empty if self-published.")
        a, b, c = st.columns(3)
        is_free = a.checkbox("Free-to-play")
        price = a.number_input("Launch price (USD)", 0.0, 100.0, 14.99, 1.0, disabled=False)
        month = b.selectbox("Release month", months, index=2)
        dow = c.selectbox("Release weekday", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                                              "Saturday", "Sunday"], index=3)

        st.markdown("**Genre & tags**")
        a, b = st.columns(2)
        genres = a.multiselect("Store genres", genre_options, default=["indie"] if "indie" in genre_options else [])
        tags = b.multiselect("Community tags", tag_options, default=["singleplayer"] if "singleplayer" in tag_options else [],
                             help="Only tags the model has learned from are listed.")

        st.markdown("**Reach & features**")
        a, b, c = st.columns(3)
        langs = a.multiselect("Languages", list(LANGUAGE_LABELS), default=["lang_english"],
                              format_func=LANGUAGE_LABELS.get)
        other_langs = a.number_input("Other languages (count)", 0, 30, 0)
        audio = a.number_input("Languages with full voice-over", 0, 30, 0)
        platforms = b.multiselect("Platforms", ["windows", "mac", "linux"], default=["windows"],
                                  format_func=str.capitalize)
        controller = b.selectbox("Controller support", ["none", "partial", "full"], index=0, format_func=str.capitalize)
        feats = b.multiselect("Steam features", list(FEATURE_LABELS), default=["feat_singleplayer"],
                              format_func=FEATURE_LABELS.get)
        achievements = c.number_input("Steam achievements", 0, 500, 0)
        has_demo = c.checkbox("Has a demo")
        mature = c.checkbox("Mature content (17+)")

        st.markdown("**Store page**")
        a, b, c = st.columns(3)
        screenshots = a.number_input("Screenshots", 0, 50, 8)
        trailers = a.number_input("Trailers", 0, 20, 1)
        images = b.number_input("Images/GIFs in description", 0, 50, 2)
        website = b.checkbox("Has a website", value=True)
        ram = c.number_input("Minimum RAM (GB)", 0.0, 64.0, 4.0, 1.0)
        storage = c.number_input("Minimum storage (GB)", 0.0, 300.0, 2.0, 1.0,
                                 help="A strong signal: bigger installs usually mean bigger productions.")
        short_desc = st.text_area("Short description (the one-liner under the capsule)", max_chars=300)
        about = st.text_area("About this game (long description)", height=150, max_chars=8000)
        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        spec = dict(name=name, developer=developer, publisher=publisher, is_free=is_free, price=price,
                    release_month=months.index(month) + 1,
                    release_day_of_week=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                                         "Saturday", "Sunday"].index(dow),
                    genres=genres, tags=tags, languages=langs, other_languages=other_langs,
                    audio_languages=audio, platforms=platforms or ["windows"], controller=controller,
                    features=feats, achievements=achievements, has_demo=has_demo, mature=mature,
                    screenshots=screenshots, trailers=trailers, description_images=images,
                    has_website=website, min_ram_gb=ram or None, min_storage_gb=storage or None,
                    short_description=short_desc, about=about)
        st.session_state["plan_result"] = (p.frame_from_manual(spec), name or "your game")
    if "plan_result" in st.session_state:
        df, nm = st.session_state["plan_result"]
        show_result(p.predict(df, horizon), df, nm)

# ---- Tab 3: about
with tab_about:
    st.subheader("How it works")
    st.markdown(f"""
- **Data:** {info['n_train']:,} released Steam games, sampled evenly across four success tiers from
  SteamSpy's catalog, with store pages from the Steam Storefront API.
- **What it predicts:** estimated owners (reviews x 35, the "Boxleiter" rule of thumb) after a chosen
  number of years on sale; revenue = owners x launch price; tier from owners/revenue.
- **Inputs (pre-launch only):** price, publisher type, genres and tags, languages, platforms, Steam
  features, release timing, store-page content (screenshots, trailers, description text, system
  requirements) and the studio's track record from {'82,500' if info['has_catalog'] else 'no'} catalog games.
  Anything that only exists after launch (reviews, Metacritic, follower counts of released games,
  DLC) is excluded.
- **Model:** {info['model_name']}, chosen by repeated 5-fold cross-validation.
""")
    a, b, c, d = st.columns(4)
    a.metric("Typical error", f"{10 ** info['mae_log10']:.1f}x", help="Median-ish factor between predicted and true owners")
    b.metric("Ranking quality", f"{info['spearman']:.2f}", help="Spearman correlation, 1 = perfect ordering")
    c.metric("Correct tier", f"{info['tier_acc']:.0%}", help="Random guessing: ~25%")
    d.metric("Within one tier", f"{info['tier_within1']:.0%}")
    t = info["temporal"]
    if t:
        st.markdown(f"**Time-based test** (trained on games up to 2022, tested on {t['n_test']} games from "
                    f"2023-2024): correct tier {t['tier_acc']:.0%} vs. {t['base_tier_acc']:.0%} for a naive "
                    f"baseline; it still over-predicts newer games by about {10 ** t['bias']:.1f}x.")
    st.subheader("Limitations")
    st.markdown("""
- The biggest driver of sales - how good and how hyped a game is - is mostly invisible on a store page.
  A pre-launch hype model (follower tracking) is being built and will be added when enough tracked
  games have launched.
- Owner counts are estimates derived from reviews, not real sales data.
- Breakout hits and very large franchises are usually under-predicted.
""")
    st.caption("Data: Steam Storefront API and SteamSpy. Not affiliated with Valve.")
