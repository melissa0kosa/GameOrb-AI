"""
GameOrb AI - Streamlit front end (Steam game commercial-success predictor).

Run locally:   streamlit run app.py
"""

import html
import threading
import time
from collections import deque

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit_searchbox import st_searchbox

from game_search import GameSearch
from train_model import owners_to_tier
from predictor import (FEATURE_LABELS, LANGUAGE_LABELS, STORE_GENRES, TAGS, SuccessPredictor,
                       parse_appid, tag_column)

st.set_page_config(page_title="GameOrb AI", page_icon=":material/sports_esports:", layout="wide",
                   initial_sidebar_state="collapsed")

# Chart colors for the dark violet theme, validated with the dataviz palette checker against the
# glass-panel surface (#1a1230): violet/red diverging pair + 4-step sequential violet tier ramp.
PALETTE = dict(blue="#8f6df5", band="#3a2a78", red="#e8524f", grid="#2d2346", rule="#b3a6d4",
               tier=["#5238a8", "#6e52d6", "#957af5", "#cbbcff"],
               tier_ink=["#ffffff", "#ffffff", "#140b26", "#140b26"])


def pal() -> dict:
    return PALETTE


# Line icons (Lucide, ISC license) drawn in the current text color
_ICON_PATHS = {
    "gamepad": '<line x1="6" x2="10" y1="11" y2="11"/><line x1="8" x2="8" y1="9" y2="13"/>'
               '<line x1="15" x2="15.01" y1="12" y2="12"/><line x1="18" x2="18.01" y1="10" y2="10"/>'
               '<path d="M17.32 5H6.68a4 4 0 0 0-3.978 3.59c-.006.052-.01.101-.017.152C2.604 9.416 2 14.456 2 16'
               'a3 3 0 0 0 3 3c1 0 1.5-.5 2-1l1.414-1.414A2 2 0 0 1 9.828 16h4.344a2 2 0 0 1 1.414.586L17 18'
               'c.5.5 1 1 2 1a3 3 0 0 0 3-3c0-1.545-.604-6.584-.685-7.258-.007-.05-.011-.1-.017-.151A4 4 0 0 0 17.32 5z"/>',
    "trending": '<polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/>',
    "bulb": '<path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5'
            '.7.7 1.3 1.5 1.5 2.5"/><path d="M9 18h6"/><path d="M10 22h4"/>',
    "building": '<rect width="16" height="20" x="4" y="2" rx="2"/><path d="M9 22v-4h6v4"/>'
                '<path d="M8 6h.01M16 6h.01M12 6h.01M12 10h.01M12 14h.01M16 10h.01M16 14h.01M8 10h.01M8 14h.01"/>',
    "users": '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/>'
             '<path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
    "dollar": '<line x1="12" x2="12" y1="2" y2="22"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>',
    "trophy": '<path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6"/><path d="M18 9h1.5a2.5 2.5 0 0 0 0-5H18"/><path d="M4 22h16"/>'
              '<path d="M10 14.66V17c0 .55-.47.98-.97 1.21C7.85 18.75 7 20.24 7 22"/>'
              '<path d="M14 14.66V17c0 .55.47.98.97 1.21C16.15 18.75 17 20.24 17 22"/>'
              '<path d="M18 2H6v7a6 6 0 0 0 12 0V2Z"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/>'
                '<path d="M3 12A9 3 0 0 0 21 12"/>',
    "eyeoff": '<path d="M9.88 9.88a3 3 0 1 0 4.24 4.24"/>'
              '<path d="M10.73 5.08A10.43 10.43 0 0 1 12 5c7 0 10 7 10 7a13.16 13.16 0 0 1-1.67 2.68"/>'
              '<path d="M6.61 6.61A13.526 13.526 0 0 0 2 12s3 7 10 7a9.74 9.74 0 0 0 5.39-1.61"/>'
              '<line x1="2" x2="22" y1="2" y2="22"/>',
    "eye": '<path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/>',
    "cpu": '<rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/>'
           '<path d="M15 2v2M15 20v2M2 15h2M2 9h2M20 15h2M20 9h2M9 2v2M9 20v2"/>',
    "gauge": '<path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/>',
    "check": '<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "shield": '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1'
              'c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/>'
              '<path d="M12 8v4"/><path d="M12 16h.01"/>',
    "search": '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
    "sliders": '<line x1="4" x2="4" y1="21" y2="14"/><line x1="4" x2="4" y1="10" y2="3"/>'
               '<line x1="12" x2="12" y1="21" y2="12"/><line x1="12" x2="12" y1="8" y2="3"/>'
               '<line x1="20" x2="20" y1="21" y2="16"/><line x1="20" x2="20" y1="12" y2="3"/>'
               '<line x1="2" x2="6" y1="14" y2="14"/><line x1="10" x2="14" y1="8" y2="8"/>'
               '<line x1="18" x2="22" y1="16" y2="16"/>',
    "globe": '<circle cx="12" cy="12" r="10"/><path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20"/>'
             '<path d="M2 12h20"/>',
    "layers": '<path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0'
              'l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"/>'
              '<path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/>',
    "flame": '<path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5'
             ' 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5z"/>',
}


def badge(name: str, size: int = 18) -> str:
    """Glossy gradient icon tile."""
    return f'<span class="badge">{icon(name, size)}</span>'


def icon(name: str, size: int = 20) -> str:
    return (f'<svg class="ico" xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
            f'viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" '
            f'stroke-linejoin="round" aria-hidden="true">{_ICON_PATHS[name]}</svg>')


LOOKUPS_PER_MINUTE = 10          # global cap on live Steam lookups (Steam throttles hard)
SESSION_COOLDOWN_S = 5
HORIZONS = {"6 months": 0.5, "1 year": 1.0, "2 years": 2.0, "3 years": 3.0, "5 years": 5.0}
DEFAULT_HORIZON = "3 years"

TIER_SHORT = {0: "Niche", 1: "Sustainable indie", 2: "Commercial hit", 3: "Blockbuster"}
# owners OR gross revenue, whichever gives the higher tier
TIER_RANGE = {0: "<10k / <$25k", 1: "10k+ / $25k+", 2: "100k+ / $250k+", 3: "1M+ / $5M+"}
TIER_BLURB = {
    0: "Most Steam games land here; it would struggle to fund a full-time team.",
    1: "Typically enough to sustain a small indie studio.",
    2: "Well beyond what most indie games reach.",
    3: "Only a few percent of Steam games get here.",
}

CSS = """
<style>
/* ---- base: near-black violet; content sits above the animated background ---- */
html, body, .stApp {background: #0b0614;}
[data-testid="stAppViewContainer"] {background: transparent; position: relative; z-index: 1;}
[data-testid="stHeader"] {background: transparent;}
.block-container {padding-top: 3.5rem; max-width: 1200px;}

/* ---- lava-lamp background: big blurred blobs that drift and change shape ---- */
.lava {position: fixed; inset: 0; z-index: -1; overflow: hidden; pointer-events: none;}
.lava span {position: absolute; display: block; filter: blur(80px); mix-blend-mode: screen; will-change: transform;}
.lava .b1 {width: 46vw; height: 42vw; left: -6vw; top: -18vh; background: #e3d8ff; opacity: .75;
           animation: lava1 28s ease-in-out infinite alternate, morph 17s ease-in-out infinite;}
.lava .b2 {width: 52vw; height: 46vw; right: -14vw; top: 4vh; background: #7a4cf2; opacity: .8;
           animation: lava2 34s ease-in-out infinite alternate, morph 21s ease-in-out infinite reverse;}
.lava .b3 {width: 58vw; height: 50vw; left: 14vw; bottom: -34vh; background: #4a22b5; opacity: .85;
           animation: lava3 40s ease-in-out infinite alternate, morph 25s ease-in-out infinite;}
.lava .b4 {width: 30vw; height: 30vw; right: 12vw; bottom: 2vh; background: #9b78ff; opacity: .55;
           animation: lava4 24s ease-in-out infinite alternate, morph 15s ease-in-out infinite reverse;}
.lava .shade {position: absolute; inset: 0; filter: none; mix-blend-mode: normal;
              background: radial-gradient(ellipse at 70% 85%, rgba(8,4,16,.7), transparent 60%),
                          radial-gradient(ellipse at 0% 100%, rgba(8,4,16,.8), transparent 55%);}
@keyframes lava1 {0% {transform: translate(0,0) scale(1);} 50% {transform: translate(16vw,14vh) scale(1.2);}
                  100% {transform: translate(4vw,30vh) scale(.9);}}
@keyframes lava2 {0% {transform: translate(0,0) scale(1);} 50% {transform: translate(-18vw,22vh) scale(.85);}
                  100% {transform: translate(-6vw,-8vh) scale(1.15);}}
@keyframes lava3 {0% {transform: translate(0,0) scale(1);} 50% {transform: translate(-14vw,-26vh) scale(1.15);}
                  100% {transform: translate(18vw,-10vh) scale(.95);}}
@keyframes lava4 {0% {transform: translate(0,0) scale(1);} 50% {transform: translate(-26vw,-30vh) scale(1.3);}
                  100% {transform: translate(-8vw,-50vh) scale(.8);}}
@keyframes morph {0%, 100% {border-radius: 42% 58% 70% 30% / 45% 45% 55% 55%;}
                  33% {border-radius: 70% 30% 46% 54% / 30% 39% 61% 70%;}
                  66% {border-radius: 32% 68% 38% 62% / 58% 34% 66% 42%;}}
.grain {position: fixed; inset: 0; z-index: -1; pointer-events: none; opacity: .16; mix-blend-mode: overlay;
        background-image: url("data:image/svg+xml,%3Csvg%20xmlns%3D%27http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%27%20width%3D%27220%27%20height%3D%27220%27%3E%3Cfilter%20id%3D%27n%27%3E%3CfeTurbulence%20type%3D%27fractalNoise%27%20baseFrequency%3D%27.9%27%20numOctaves%3D%273%27%20stitchTiles%3D%27stitch%27%2F%3E%3C%2Ffilter%3E%3Crect%20width%3D%27100%25%27%20height%3D%27100%25%27%20filter%3D%27url%28%23n%29%27%2F%3E%3C%2Fsvg%3E");}
@media (prefers-reduced-motion: reduce) {.lava span {animation: none !important;}}

/* ---- dark violet glass for everything with text ---- */
[class*="st-key-panel"], div[data-testid="stExpander"] details {
    background: rgba(18,10,36,.72) !important; border: 1px solid rgba(190,165,255,.16) !important;
    border-radius: 16px !important; backdrop-filter: blur(16px) saturate(140%);
    box-shadow: 0 10px 30px rgba(0,0,0,.25);}
.hero {background: linear-gradient(135deg, rgba(92,56,196,.62) 0%, rgba(44,20,100,.66) 55%, rgba(20,10,44,.72) 100%);
       border: 1px solid rgba(214,196,255,.22); border-radius: 20px; padding: 2rem 2.2rem; color: #fff;
       margin-bottom: 1.2rem; backdrop-filter: blur(18px) saturate(150%); box-shadow: 0 16px 40px rgba(0,0,0,.3);}
.hero h1 {color: #fff; font-size: 2.1rem; margin: 0 0 .4rem 0; padding: 0; display: flex; align-items: center; gap: .6rem;}
.hero h1 .ico {color: #d9ccff;}
.hero p {color: #e6ddff; text-shadow: 0 1px 2px rgba(0,0,0,.35); font-size: 1.05rem; margin: 0 0 1rem 0; max-width: 760px;}
.chip {display: inline-block; background: rgba(255,255,255,.1); border: 1px solid rgba(255,255,255,.2);
       color: #fff; border-radius: 999px; padding: .2rem .75rem; margin: 0 .4rem .4rem 0; font-size: .85rem;}
.card {background: rgba(18,10,36,.72); border: 1px solid rgba(190,165,255,.16); border-radius: 16px;
       padding: 1.1rem 1.25rem; min-height: 172px; backdrop-filter: blur(16px) saturate(140%);
       box-shadow: 0 10px 30px rgba(0,0,0,.25);}
.card .label {color: #b3a6d4; font-size: .8rem; text-transform: uppercase; letter-spacing: .05em;}
.card .value {color: #f3eeff; font-size: 2rem; font-weight: 700; line-height: 1.2; margin: .25rem 0;}
.card .sub {color: #cfc4ec; font-size: .88rem;}
.game {display: flex; gap: 1.6rem; align-items: center; background: rgba(18,10,36,.72);
       border: 1px solid rgba(190,165,255,.16); border-radius: 18px; padding: 1.2rem; margin: .8rem 0 1rem 0;
       backdrop-filter: blur(16px) saturate(140%); box-shadow: 0 10px 30px rgba(0,0,0,.25);}
.game img {width: 440px; max-width: 48%; border-radius: 12px; flex-shrink: 0; box-shadow: 0 8px 24px rgba(0,0,0,.4);}
.game h3 {margin: 0 0 .6rem 0; padding: 0; font-size: 2rem; color: #f3eeff;}
.game a {color: #c6b6ff;}
.pill {display: inline-block; background: rgba(255,255,255,.08); color: #ddd3ff; border-radius: 999px;
       padding: .18rem .7rem; margin: 0 .35rem .4rem 0; font-size: .85rem; border: 1px solid rgba(255,255,255,.08);}
.pill.blue {background: rgba(143,109,245,.24); color: #e2d8ff;}
.pill.price {background: #7c5cf0; color: #fff; font-weight: 600; font-size: .95rem; border-color: #9b80ff;}
.verdict {font-size: 1.15rem; color: #ffffff; margin: .2rem 0 1rem 0; text-shadow: 0 1px 3px rgba(0,0,0,.65);}
/* text that sits directly on the moving background */
[data-testid="stCaptionContainer"], [data-testid="stMarkdownContainer"] > p,
[data-testid="stWidgetLabel"] {text-shadow: 0 1px 3px rgba(0,0,0,.6);}
.ladder {display: flex; gap: 4px; margin-top: .6rem;}
.ladder div {flex: 1; border-radius: 6px; padding: .45rem .3rem; text-align: center; font-size: .72rem;
             color: #b3a6d4; background: rgba(255,255,255,.06); line-height: 1.25;}
.ladder div.on {font-weight: 600; outline: 2px solid #efe9ff; outline-offset: 1px;}
.ladder div b {display: block; font-size: .78rem;}
.section-title {font-size: 1.1rem; font-weight: 650; margin: .2rem 0 .1rem 0; color: #f3eeff;
                display: flex; align-items: center; gap: .5rem;}
.section-title .ico {color: #b9a2ff;}
.muted {color: #b3a6d4; font-size: .88rem;}
/* tabs as a glass bar; the selected tab in white so it reads on any part of the background */
[data-testid="stTabs"] [role="tablist"] {background: rgba(18,10,36,.72); border: 1px solid rgba(190,165,255,.16);
    border-radius: 14px; padding: .3rem .6rem; backdrop-filter: blur(16px) saturate(140%); gap: .6rem;}
[data-testid="stTab"] {color: #cfc4ec !important;}
[data-testid="stTab"] * {color: inherit !important;}
[data-testid="stTab"][aria-selected="true"] {color: #ffffff !important;}
/* example buttons as glass pills */
.st-key-example button {background: rgba(255,255,255,.07) !important; border-color: rgba(214,196,255,.22) !important;
    color: #ece6ff !important;}
.st-key-example button:hover {background: rgba(143,109,245,.3) !important; border-color: #9b80ff !important;}
@media (max-width: 800px) {.game {flex-direction: column; align-items: flex-start;}
                           .game img {width: 100%; max-width: 100%;}}

.stars {position: fixed; top: 0; left: 0; width: 1px; height: 1px; border-radius: 50%; z-index: -1;
         pointer-events: none; background: transparent;}
.stars.s1 {box-shadow: 32.4vw 15.1vh 0 0px rgba(235,228,255,0.71), 7.2vw 53.6vh 0 0px rgba(235,228,255,0.55), 5.8vw 50.7vh 0 0px rgba(235,228,255,0.37), 43.4vw 7.0vh 0 0px rgba(235,228,255,0.40), 42.5vw 82.7vh 0 0px rgba(235,228,255,0.42), 22.3vw 62.7vh 0 0px rgba(235,228,255,0.87), 57.7vw 39.7vh 0 0px rgba(235,228,255,0.89), 4.7vw 85.8vh 0 0px rgba(235,228,255,0.51), 14.4vw 11.8vh 0 0px rgba(235,228,255,0.52), 81.6vw 18.1vh 0 0px rgba(235,228,255,0.67), 63.9vw 37.2vh 0 0px rgba(235,228,255,0.65), 6.3vw 6.0vh 0 0px rgba(235,228,255,0.46), 68.0vw 42.8vh 0 0px rgba(235,228,255,0.52), 58.6vw 45.3vh 0 0px rgba(235,228,255,0.51), 79.4vw 69.9vh 0 0px rgba(235,228,255,0.48), 57.4vw 52.5vh 0 0px rgba(235,228,255,0.83), 72.9vw 28.8vh 0 0px rgba(235,228,255,0.89), 11.8vw 41.8vh 0 0px rgba(235,228,255,0.77), 15.2vw 48.9vh 0 0px rgba(235,228,255,0.37), 66.8vw 76.5vh 0 0px rgba(235,228,255,0.67), 87.5vw 31.4vh 0 0px rgba(235,228,255,0.73), 59.4vw 58.0vh 0 0px rgba(235,228,255,0.60), 84.0vw 94.5vh 0 0px rgba(235,228,255,0.61), 66.4vw 6.1vh 0 0px rgba(235,228,255,0.74), 64.7vw 99.3vh 0 0px rgba(235,228,255,0.80), 28.5vw 38.6vh 0 0px rgba(235,228,255,0.72), 2.3vw 46.2vh 0 0px rgba(235,228,255,0.44), 11.7vw 5.9vh 0 0px rgba(235,228,255,0.77), 12.9vw 24.8vh 0 0px rgba(235,228,255,0.57), 87.1vw 8.1vh 0 0px rgba(235,228,255,0.60), 54.9vw 88.3vh 0 0px rgba(235,228,255,0.80), 86.4vw 27.8vh 0 0px rgba(235,228,255,0.58), 35.9vw 88.4vh 0 0px rgba(235,228,255,0.88), 15.1vw 17.6vh 0 0px rgba(235,228,255,0.48), 23.3vw 48.5vh 0 0px rgba(235,228,255,0.67), 26.3vw 0.4vh 0 0px rgba(235,228,255,0.58), 36.9vw 56.6vh 0 0px rgba(235,228,255,0.87), 69.0vw 51.5vh 0 0px rgba(235,228,255,0.69), 67.6vw 5.4vh 0 0px rgba(235,228,255,0.84), 78.0vw 87.5vh 0 0px rgba(235,228,255,0.79), 39.2vw 39.9vh 0 0px rgba(235,228,255,0.41), 63.4vw 6.2vh 0 0px rgba(235,228,255,0.39), 20.9vw 16.2vh 0 0px rgba(235,228,255,0.54), 5.3vw 0.0vh 0 0px rgba(235,228,255,0.43), 10.1vw 36.4vh 0 0px rgba(235,228,255,0.36), 87.4vw 61.4vh 0 0px rgba(235,228,255,0.43), 25.2vw 34.7vh 0 0px rgba(235,228,255,0.55), 12.3vw 84.9vh 0 0px rgba(235,228,255,0.90), 46.6vw 48.4vh 0 0px rgba(235,228,255,0.40), 10.2vw 34.3vh 0 0px rgba(235,228,255,0.50), 82.9vw 16.1vh 0 0px rgba(235,228,255,0.36), 95.1vw 52.8vh 0 0px rgba(235,228,255,0.43), 54.3vw 2.7vh 0 0px rgba(235,228,255,0.64), 97.9vw 86.3vh 0 0px rgba(235,228,255,0.73), 26.1vw 36.7vh 0 0px rgba(235,228,255,0.44), 77.2vw 53.3vh 0 0px rgba(235,228,255,0.78), 33.0vw 22.3vh 0 0px rgba(235,228,255,0.80), 98.5vw 85.3vh 0 0px rgba(235,228,255,0.79), 81.8vw 74.0vh 0 0px rgba(235,228,255,0.47), 51.8vw 35.6vh 0 0px rgba(235,228,255,0.37), 2.8vw 27.9vh 0 0px rgba(235,228,255,0.49), 69.3vw 95.7vh 0 0px rgba(235,228,255,0.60), 93.7vw 98.8vh 0 0px rgba(235,228,255,0.88), 36.5vw 22.0vh 0 0px rgba(235,228,255,0.47), 19.7vw 20.4vh 0 0px rgba(235,228,255,0.69), 90.0vw 84.0vh 0 0px rgba(235,228,255,0.61), 65.3vw 80.0vh 0 0px rgba(235,228,255,0.40), 66.1vw 91.0vh 0 0px rgba(235,228,255,0.78), 75.0vw 47.8vh 0 0px rgba(235,228,255,0.45), 78.9vw 33.3vh 0 0px rgba(235,228,255,0.79); animation: twinkle 5s ease-in-out infinite alternate;}
.stars.s2 {box-shadow: 97.2vw 39.6vh 0 .5px rgba(235,228,255,0.57), 94.7vw 72.5vh 0 .5px rgba(235,228,255,0.44), 12.7vw 15.1vh 0 .5px rgba(235,228,255,0.85), 80.7vw 14.6vh 0 .5px rgba(235,228,255,0.80), 98.0vw 65.7vh 0 .5px rgba(235,228,255,0.54), 54.9vw 13.1vh 0 .5px rgba(235,228,255,0.36), 97.1vw 65.0vh 0 .5px rgba(235,228,255,0.64), 93.4vw 43.4vh 0 .5px rgba(235,228,255,0.83), 82.6vw 21.1vh 0 .5px rgba(235,228,255,0.49), 29.3vw 24.1vh 0 .5px rgba(235,228,255,0.67), 25.9vw 41.9vh 0 .5px rgba(235,228,255,0.42), 91.0vw 35.4vh 0 .5px rgba(235,228,255,0.60), 58.3vw 90.4vh 0 .5px rgba(235,228,255,0.58), 91.8vw 50.2vh 0 .5px rgba(235,228,255,0.64), 52.4vw 1.9vh 0 .5px rgba(235,228,255,0.59), 18.3vw 0.4vh 0 .5px rgba(235,228,255,0.79), 17.2vw 47.3vh 0 .5px rgba(235,228,255,0.75), 55.6vw 32.6vh 0 .5px rgba(235,228,255,0.64), 55.5vw 78.4vh 0 .5px rgba(235,228,255,0.41), 56.0vw 24.8vh 0 .5px rgba(235,228,255,0.50), 77.2vw 50.8vh 0 .5px rgba(235,228,255,0.66), 76.0vw 91.2vh 0 .5px rgba(235,228,255,0.59), 61.3vw 50.6vh 0 .5px rgba(235,228,255,0.63), 69.3vw 45.2vh 0 .5px rgba(235,228,255,0.64), 47.8vw 94.2vh 0 .5px rgba(235,228,255,0.73), 87.7vw 94.2vh 0 .5px rgba(235,228,255,0.49), 56.0vw 94.3vh 0 .5px rgba(235,228,255,0.81), 13.7vw 12.2vh 0 .5px rgba(235,228,255,0.59), 7.3vw 24.1vh 0 .5px rgba(235,228,255,0.39), 66.9vw 78.4vh 0 .5px rgba(235,228,255,0.84), 15.4vw 71.6vh 0 .5px rgba(235,228,255,0.71), 14.3vw 88.3vh 0 .5px rgba(235,228,255,0.88), 22.0vw 95.3vh 0 .5px rgba(235,228,255,0.57), 48.7vw 99.0vh 0 .5px rgba(235,228,255,0.81), 16.1vw 43.2vh 0 .5px rgba(235,228,255,0.63), 33.9vw 19.6vh 0 .5px rgba(235,228,255,0.53), 72.2vw 1.9vh 0 .5px rgba(235,228,255,0.65), 44.0vw 1.8vh 0 .5px rgba(235,228,255,0.53), 62.4vw 51.2vh 0 .5px rgba(235,228,255,0.39), 98.5vw 78.8vh 0 .5px rgba(235,228,255,0.88); animation: twinkle 7s ease-in-out 1.5s infinite alternate;}
.stars.s3 {width: 2px; height: 2px; box-shadow: 10.5vw 26.6vh 0 1px rgba(235,228,255,0.37), 77.9vw 27.0vh 0 1px rgba(235,228,255,0.42), 42.2vw 91.1vh 0 1px rgba(235,228,255,0.80), 25.9vw 14.9vh 0 1px rgba(235,228,255,0.86), 57.1vw 70.0vh 0 1px rgba(235,228,255,0.40), 5.8vw 68.8vh 0 1px rgba(235,228,255,0.58), 7.2vw 93.8vh 0 1px rgba(235,228,255,0.70), 80.2vw 8.4vh 0 1px rgba(235,228,255,0.82), 6.7vw 86.3vh 0 1px rgba(235,228,255,0.60), 33.9vw 55.3vh 0 1px rgba(235,228,255,0.86), 26.8vw 12.9vh 0 1px rgba(235,228,255,0.64), 23.8vw 10.9vh 0 1px rgba(235,228,255,0.44), 5.0vw 20.2vh 0 1px rgba(235,228,255,0.52), 30.5vw 75.9vh 0 1px rgba(235,228,255,0.51), 50.0vw 17.8vh 0 1px rgba(235,228,255,0.54), 1.8vw 25.0vh 0 1px rgba(235,228,255,0.36), 73.3vw 55.1vh 0 1px rgba(235,228,255,0.45), 47.5vw 93.5vh 0 1px rgba(235,228,255,0.41);
           filter: drop-shadow(0 0 3px rgba(200,180,255,.9)); animation: twinkle 3.5s ease-in-out .7s infinite alternate;}
@keyframes twinkle {0% {opacity: .25;} 100% {opacity: 1;}}

/* ---- entrances ---- */
@keyframes rise {from {opacity: 0; transform: translateY(26px) rotate(-1.5deg) scale(.97);}
                 to {opacity: 1; transform: none;}}
@keyframes wordUp {from {opacity: 0; transform: translateY(.55em); filter: blur(8px);}
                   to {opacity: 1; transform: none; filter: blur(0);}}
@keyframes fadeUp {from {opacity: 0; transform: translateY(12px);} to {opacity: 1; transform: none;}}
@keyframes shimmer {to {background-position: 300% 0;}}
@keyframes beam {0% {transform: translateX(-60%) rotate(18deg); opacity: 0;} 20% {opacity: .9;}
                 60% {opacity: .6;} 100% {transform: translateX(260%) rotate(18deg); opacity: 0;}}
@keyframes glowPulse {0%, 100% {box-shadow: 0 0 0 0 rgba(170,140,255,.0), 0 0 14px rgba(143,109,245,.35);}
                      50% {box-shadow: 0 0 0 3px rgba(170,140,255,.18), 0 0 26px rgba(143,109,245,.6);}}
@keyframes reveal {from {opacity: 0; transform: translateY(40px) scale(.97);} to {opacity: 1; transform: none;}}

/* ---- hero ---- */
.hero {position: relative; overflow: hidden;}
.hero::before {content: ""; position: absolute; top: -60%; left: 0; width: 22%; height: 220%;
               background: linear-gradient(90deg, transparent, rgba(220,205,255,.22), transparent);
               filter: blur(6px); animation: beam 7s ease-in-out 1.2s infinite; pointer-events: none;}
.hero h1 {font-size: 2.5rem; gap: .25rem .55rem; flex-wrap: wrap;}
.hero h1 .w {display: inline-block; opacity: 0; animation: wordUp .8s cubic-bezier(.2,.7,.2,1) forwards;}
.grad {background: linear-gradient(90deg, #d9ccff, #9d7bff, #f0eaff, #b9a2ff, #d9ccff); background-size: 300% 100%;
       -webkit-background-clip: text; background-clip: text; color: transparent !important;
       animation: shimmer 8s linear infinite;}
.hero h1 .w .grad {animation: shimmer 8s linear infinite;}
.hero p {opacity: 0; animation: fadeUp .8s ease .75s forwards;}
.hero .chip {opacity: 0; animation: fadeUp .7s ease forwards; border-color: rgba(214,196,255,.35);
             box-shadow: 0 0 14px rgba(143,109,245,.25), inset 0 1px 0 rgba(255,255,255,.12);}
.hero .chip:nth-of-type(1) {animation-delay: .95s;} .hero .chip:nth-of-type(2) {animation-delay: 1.05s;}
.hero .chip:nth-of-type(3) {animation-delay: 1.15s;}

/* ---- flat icon tiles ---- */
.badge {display: inline-flex; align-items: center; justify-content: center; flex-shrink: 0;
        width: 34px; height: 34px; border-radius: 10px;
        background: rgba(143,109,245,.18); border: 1px solid rgba(185,160,255,.38);
        transition: background .3s, border-color .3s, transform .3s;}
.badge .ico {color: #cbbcff !important;}
.badge.big {width: 52px; height: 52px; border-radius: 14px; margin-right: .35rem; opacity: 0;
            animation: fadeUp .7s ease .05s forwards;}
.card:hover .badge, .tile:hover .badge {background: rgba(143,109,245,.32); border-color: rgba(203,188,255,.7);
                                        transform: scale(1.06);}
/* ---- cards: glowing glass, float in one after another, lift on hover ---- */
.card, .tile {background: linear-gradient(180deg, rgba(48,33,96,.8) 0%, rgba(15,9,32,.88) 100%) !important;
        border: 1px solid rgba(170,150,255,.2) !important;
        box-shadow: inset 0 1px 0 rgba(255,255,255,.09), 0 18px 40px rgba(0,0,0,.35) !important;
        transition: transform .4s cubic-bezier(.2,.7,.2,1), box-shadow .4s, border-color .4s;}
.card:hover, .tile:hover {transform: translateY(-5px);
        border-color: rgba(185,160,255,.55) !important;
        box-shadow: inset 0 1px 0 rgba(255,255,255,.12), 0 22px 48px rgba(0,0,0,.4),
                    0 0 34px rgba(143,109,245,.32) !important;}
[data-testid="stColumn"]:nth-child(2) .card {animation-delay: .12s;}
[data-testid="stColumn"]:nth-child(3) .card {animation-delay: .24s;}
[data-testid="stColumn"]:nth-child(4) .card {animation-delay: .36s;}
.card-head {display: flex; align-items: center; gap: .6rem;}
.card .value {background: linear-gradient(180deg, #ffffff 35%, #cbbcff 100%);
              -webkit-background-clip: text; background-clip: text; color: transparent;
              font-size: 2.2rem; margin: .45rem 0 .2rem 0;}
.ladder div.on {animation: glowPulse 2.6s ease-in-out infinite;}

/* ---- panels: same glass, glow on hover, reveal while scrolling ---- */
[class*="st-key-panel"], .game {
    background: linear-gradient(180deg, rgba(40,27,82,.72) 0%, rgba(13,8,28,.84) 100%) !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,.08), 0 18px 40px rgba(0,0,0,.35) !important;
    transition: box-shadow .45s, border-color .45s;}
[class*="st-key-panel"]:hover, .game:hover {border-color: rgba(185,160,255,.4) !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,.1), 0 22px 48px rgba(0,0,0,.4), 0 0 30px rgba(143,109,245,.22) !important;}
.game img {transition: transform .6s cubic-bezier(.2,.7,.2,1);}
.game:hover img {transform: scale(1.03) rotate(-.6deg);}
/* selected time-period button: white text on violet */
button[aria-checked="true"] {background: #7c5cf0 !important; border-color: #9b80ff !important;
                              color: #ffffff !important;}
button[aria-checked="true"] * {color: #ffffff !important;}
/* ---- "How could this game sell more?" list ---- */
.wi-list {display: flex; flex-direction: column; gap: .7rem; margin-top: .8rem;}
.wi-row {background: rgba(255,255,255,.04); border: 1px solid rgba(190,165,255,.14); border-radius: 12px;
         padding: .7rem .85rem; transition: border-color .3s, background .3s;}
.wi-row:hover {background: rgba(143,109,245,.1); border-color: rgba(190,165,255,.35);}
.wi-top {display: flex; justify-content: space-between; align-items: baseline; gap: .8rem;}
.wi-name {color: #f3eeff; font-weight: 600; font-size: .95rem;}
.wi-pct {color: #cbbcff; font-weight: 700; font-size: 1.05rem; white-space: nowrap;}
.wi-row.down .wi-pct {color: #ff8a87;}
.wi-tag {display: inline-block; margin-left: .5rem; padding: .05rem .5rem; border-radius: 999px; font-size: .7rem;
         font-weight: 600; letter-spacing: .03em; color: #fff; background: #7c5cf0; vertical-align: 2px;}
.wi-bar {height: 6px; border-radius: 999px; background: rgba(255,255,255,.07); margin: .45rem 0 .4rem 0; overflow: hidden;}
.wi-bar span {display: block; height: 100%; border-radius: 999px;
              background: linear-gradient(90deg, #6e52d6, #a58bff); transform-origin: left;
              animation: growBar 1s cubic-bezier(.2,.7,.2,1) .2s both;}
.wi-row.down .wi-bar span {background: #e8524f;}
.wi-result {color: #cfc4ec; font-size: .85rem;}
.wi-row.flat .wi-pct {color: #9d93bd; font-weight: 600; font-size: .85rem;}
.wi-row.flat .wi-bar span {background: rgba(203,188,255,.35);}
.wi-done-row {display: flex; flex-wrap: wrap; align-items: center; gap: .4rem; margin-top: .8rem;}
.wi-done-label {color: #b3a6d4; font-size: .8rem; text-transform: uppercase; letter-spacing: .05em; margin-right: .2rem;}
.wi-done {display: inline-flex; align-items: center; gap: .3rem; padding: .15rem .6rem; border-radius: 999px;
          font-size: .82rem; color: #e2d8ff; background: rgba(143,109,245,.16); border: 1px solid rgba(185,160,255,.28);}
.wi-done .ico {color: #a8f0c6;}
.wi-result b {color: #ffffff;}
@keyframes growBar {from {transform: scaleX(0);} to {transform: scaleX(1);}}
/* the zero-height script frame takes no space */
[data-testid="stElementContainer"]:has(iframe[title="st.iframe"]) {display: none;}
/* the search box frame sets a fixed pixel width once while the page is still settling;
   always make it fill its panel instead */
iframe[title*="searchbox"] {width: 100% !important; max-width: 100% !important; display: block;}
/* ---- pop-up when a section scrolls into view (classes set by the script at the end) ---- */
.pop-ready {opacity: 0; transform: translateY(38px) scale(.9);}
.pop-in {animation: popIn .75s cubic-bezier(.34,1.56,.64,1) both;}
@keyframes popIn {0% {opacity: 0; transform: translateY(38px) scale(.9);}
                  60% {opacity: 1; transform: translateY(-5px) scale(1.025);}
                  100% {opacity: 1; transform: none;}}

/* ---- feature tiles (How it works) ---- */
.title-wrap {width: fit-content; margin: 0 auto 1.2rem auto; padding: 1rem 2.4rem; position: relative; isolation: isolate;}
.title-wrap::before {content: ""; position: absolute; inset: 4px -10px; border-radius: 50%; z-index: -1;
                     background: rgba(11,6,22,.62); filter: blur(26px);}
.title-wrap .big-sub {margin: 0; text-shadow: 0 1px 3px rgba(0,0,0,.7);}
/* info/warning boxes in the violet style */
[data-testid="stAlertContainer"] {background: rgba(143,109,245,.16) !important; border: 1px solid rgba(185,160,255,.3) !important;
    border-radius: 12px !important;}
[data-testid="stAlertContainer"] * {color: #e8e0ff !important;}
.big-title {text-align: center; font-size: 2.2rem; font-weight: 700; color: #fff; margin: .5rem 0 .2rem 0;
            text-shadow: 0 2px 12px rgba(0,0,0,.4);}
.big-sub {text-align: center; color: #cfc4ec; margin: 0 0 1.4rem 0;}
.tiles {display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem;}
.tile {border-radius: 18px; padding: 1.3rem 1.3rem 1.1rem;}
.tile .badge {width: 46px; height: 46px; border-radius: 13px;}
.tile h4 {color: #fff; margin: .9rem 0 .35rem 0; padding: 0; font-size: 1.08rem;}
.tile p {color: #cfc4ec; font-size: .9rem; margin: 0; line-height: 1.5;}
@media (max-width: 900px) {.tiles {grid-template-columns: 1fr 1fr;}}
@media (max-width: 600px) {.tiles {grid-template-columns: 1fr;}}

/* ---- glowing primary buttons with a light sweep ---- */
[data-testid="stBaseButton-primaryFormSubmit"], [data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #a084ff 0%, #6c48ea 100%) !important; border: 1px solid rgba(225,212,255,.45) !important;
    box-shadow: 0 8px 26px rgba(110,70,240,.45), inset 0 1px 0 rgba(255,255,255,.35) !important;
    position: relative; overflow: hidden; transition: transform .25s, box-shadow .25s;}
[data-testid="stBaseButton-primaryFormSubmit"]:hover, [data-testid="stBaseButton-primary"]:hover {
    transform: translateY(-2px); box-shadow: 0 12px 34px rgba(120,80,255,.6), inset 0 1px 0 rgba(255,255,255,.4) !important;}
[data-testid="stBaseButton-primaryFormSubmit"]::after, [data-testid="stBaseButton-primary"]::after {
    content: ""; position: absolute; top: 0; left: -40%; width: 30%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,.35), transparent); transform: skewX(-20deg);
    transition: left .7s ease;}
[data-testid="stBaseButton-primaryFormSubmit"]:hover::after, [data-testid="stBaseButton-primary"]:hover::after {left: 120%;}

@media (prefers-reduced-motion: reduce) {
  .hero::before, .stars, .ladder div.on {animation: none !important;}
  .hero h1 .w, .hero p, .hero .chip, .badge.big, .pop-ready, .pop-in {
      animation: none !important; opacity: 1 !important; transform: none !important;}
}
</style>
"""

BACKGROUND = """<div class="lava"><span class="b1"></span><span class="b2"></span><span class="b3"></span><span class="b4"></span>
<span class="shade"></span></div><div class="grain"></div>
<div class="stars s1"></div><div class="stars s2"></div><div class="stars s3"></div>
"""


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


@st.cache_resource(show_spinner=False)
def get_search() -> GameSearch:
    return GameSearch()


@st.cache_data(ttl=6 * 3600, show_spinner=False, max_entries=5000)
def search_cached(term: str):
    return get_search().suggest(term)


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


def esc(s) -> str:
    return html.escape(str(s or ""))


def card(label: str, value: str, sub: str = "", extra: str = "", ico: str = None) -> str:
    head = f'<div class="card-head">{badge(ico) if ico else ""}<div class="label">{label}</div></div>'
    return (f'<div class="card">{head}<div class="value">{value}</div>'
            f'<div class="sub">{sub}</div>{extra}</div>')


def tier_ladder(tier: int, highlight: bool = True) -> str:
    """4-step tier scale; steps up to the predicted tier are filled with the sequential blue ramp."""
    cells, c = [], pal()
    for t in range(4):
        style = ""
        if t <= tier:
            style = f' style="background:{c["tier"][t]};color:{c["tier_ink"][t]}"'
        cls = ' class="on"' if highlight and t == tier else ""
        cells.append(f"<div{cls}{style}><b>{TIER_SHORT[t]}</b>{TIER_RANGE[t]}</div>")
    return f'<div class="ladder">{"".join(cells)}</div>'


# ---------------------------------------------------------------------------- charts
def curve_chart(res: dict) -> alt.Chart:
    c = pal()
    lo_f = res["owners_lo"] / res["owners"]
    hi_f = res["owners_hi"] / res["owners"]
    df = pd.DataFrame([{"Years on sale": y, "Owners": o, "Low (10%)": o * lo_f, "High (90%)": o * hi_f}
                       for y, o in res["curve"]])
    lo, hi = df["Low (10%)"].min(), df["High (90%)"].max()
    ticks = [10 ** k for k in range(0, 11) if lo / 10 <= 10 ** k <= hi * 10]
    base = alt.Chart(df).encode(x=alt.X("Years on sale:Q", axis=alt.Axis(grid=False, tickCount=6)))
    band = base.mark_area(color=c["band"], opacity=0.6).encode(
        y=alt.Y("Low (10%):Q", scale=alt.Scale(type="log"), title="Estimated owners (log scale)",
                axis=alt.Axis(gridColor=c["grid"], format="~s", values=ticks)),
        y2="High (90%):Q")
    line = base.mark_line(color=c["blue"], strokeWidth=2).encode(y="Owners:Q")
    pts = base.mark_circle(color=c["blue"], size=70, opacity=1).encode(
        y="Owners:Q",
        tooltip=[alt.Tooltip("Years on sale:Q"), alt.Tooltip("Owners:Q", format=",.0f"),
                 alt.Tooltip("Low (10%):Q", format=",.0f"), alt.Tooltip("High (90%):Q", format=",.0f")])
    now = alt.Chart(pd.DataFrame({"x": [res["horizon_years"]]})).mark_rule(
        color=c["rule"], strokeDash=[4, 4], strokeWidth=1).encode(x=alt.X("x:Q", title="Years on sale"))
    return (band + line + pts + now).properties(height=290).configure_view(stroke=None).configure(background="transparent")


# Every change the what-if analysis can test (labels must match SuccessPredictor.what_if)
WHAT_IF_CHECKS = ["Add French, German & Spanish", "Add Simplified Chinese", "Add full controller support",
                  "Add Steam achievements (30)", "Release a demo", "Have 2+ trailers", "Ship on Mac & Linux too"]
DONE_LABELS = {"Add French, German & Spanish": "French, German & Spanish", "Add Simplified Chinese": "Chinese",
               "Add full controller support": "Controller support", "Add Steam achievements (30)": "Achievements",
               "Release a demo": "Demo", "Have 2+ trailers": "2+ trailers", "Ship on Mac & Linux too": "Mac & Linux"}


def what_if_html(rows: list, base_owners: float) -> str:
    """Ranked list: each row = one change, the owners it leads to, and a bar for the size of the effect."""
    biggest = max(max(abs(r["factor"] - 1) for r in rows), 1e-9)
    out = []
    for i, r in enumerate(rows):
        pct = (r["factor"] - 1) * 100
        small = abs(pct) < 3
        cls = " flat" if small else (" down" if pct < 0 else "")
        width = max(3.0, abs(r["factor"] - 1) / biggest * 100)
        tag = '<span class="wi-tag">Biggest boost</span>' if i == 0 and pct >= 3 else ""
        pct_txt = "little effect" if small else f"{pct:+.0f}%"
        out.append(
            f'<div class="wi-row{cls}">'
            f'<div class="wi-top"><span class="wi-name">{esc(r["change"])}{tag}</span>'
            f'<span class="wi-pct">{pct_txt}</span></div>'
            f'<div class="wi-bar"><span style="width:{width:.0f}%"></span></div>'
            f'<div class="wi-result">{fmt_count(base_owners)} &rarr; <b>{fmt_count(base_owners * r["factor"])}</b>'
            f' owners</div></div>')
    return '<div class="wi-list">' + "".join(out) + "</div>"


def done_html(done: list) -> str:
    chips = "".join(f'<span class="wi-done">{icon("check", 14)} {esc(DONE_LABELS[d])}</span>' for d in done)
    return f'<div class="wi-done-row"><span class="wi-done-label">Already has</span>{chips}</div>'


# ---------------------------------------------------------------------------- result view
def price_label(game: dict) -> str:
    if game.get("is_free"):
        return "Free to play"
    price = float(game.get("initial_price_usd") or 0)
    return f"${price:.2f}" if price > 0 else "Price not announced"


def game_header(game: dict):
    appid = int(game["appid"])
    status = "Released" if game.get("is_released") else "Upcoming"
    pills = [f'<span class="pill price">{price_label(game)}</span>',
             f'<span class="pill blue">{status}</span>',
             f'<span class="pill">{esc(game.get("release_raw") or "Date TBA")}</span>']
    if game.get("publisher_tier"):
        pills.append(f'<span class="pill">{esc(str(game["publisher_tier"]).capitalize())} publisher</span>')
    if game.get("language_count"):
        pills.append(f'<span class="pill">{int(game["language_count"])} languages</span>')
    tags = [t.strip() for t in str(game.get("top_5_tags_str") or "").split(",") if t.strip()]
    pills += [f'<span class="pill">{esc(t)}</span>' for t in tags[:5]]
    img = esc(game.get("header_image")
              or f"https://shared.akamai.steamstatic.com/store_item_assets/steam/apps/{appid}/header.jpg")
    st.markdown(
        f'<div class="game"><img src="{img}" alt=""><div><h3>{esc(game.get("name") or "Unknown game")}</h3>'
        f'{"".join(pills)}<div class="muted" style="margin-top:.3rem">'
        f'<a href="https://store.steampowered.com/app/{appid}/" target="_blank">View on Steam &#8599;</a></div>'
        f'</div></div>', unsafe_allow_html=True)
    if game.get("is_released"):
        st.caption("This game is already out. The forecast only uses what was knowable before launch, "
                   "so it can differ a lot from its real sales - a good way to see how the model behaves.")


def show_result(df: pd.DataFrame, name: str, key: str, appid=None):
    p = get_predictor()
    period = st.segmented_control("Forecast after", list(HORIZONS), default=DEFAULT_HORIZON, key=f"h_{key}",
                                  help="How long the game has been on sale when we estimate its owners.")
    period = period or DEFAULT_HORIZON
    h = HORIZONS[period]
    res = p.predict(df, h, appid=appid)
    tier = res["tier"]

    # --- headline
    owners_tier = int(owners_to_tier([res["owners"]], [0])[0])
    by_revenue = (f' and <b>{fmt_money(res["revenue"])}</b> in gross revenue'
                  if res["revenue"] is not None and tier > owners_tier else "")
    st.markdown(
        f'<div class="verdict">After <b>{period}</b> on sale, <b>{esc(name)}</b> is likely to reach about '
        f'<b>{fmt_count(res["owners"])} owners</b>{by_revenue}: a <b>{TIER_SHORT[tier].lower()}</b> result'
        f'{" by revenue" if by_revenue else ""}. {TIER_BLURB[tier]}</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1, 1.4])
    c1.markdown(card("Estimated owners", fmt_count(res["owners"]),
                     f"Likely range {fmt_count(res['owners_lo'])} - {fmt_count(res['owners_hi'])}", ico="users"),
                unsafe_allow_html=True)
    if res["is_free"]:
        rev = card("Gross revenue", "Free", "In-game purchases aren't modeled.", ico="dollar")
    elif res["revenue"] is None:
        rev = card("Gross revenue", "-", "No price set yet.", ico="dollar")
    else:
        rev = card("Gross revenue", fmt_money(res["revenue"]),
                   f"Range {fmt_money(res['revenue_lo'])} - {fmt_money(res['revenue_hi'])}<br>"
                   f"Before Steam's 30% cut, refunds and discounts", ico="dollar")
    c2.markdown(rev, unsafe_allow_html=True)
    c3.markdown(card("Commercial tier", TIER_SHORT[tier], "", tier_ladder(tier), ico="trophy"), unsafe_allow_html=True)
    st.write("")

    # --- charts
    left, right = st.columns(2, gap="medium")
    with left.container(border=True, key=f"panel_curve_{key}"):
        st.markdown(f'<div class="section-title">{badge("trending")} Owners over time</div>'
                    '<div class="muted">Line = estimate · band = likely range · '
                    'dashed = horizon</div>', unsafe_allow_html=True)
        st.altair_chart(curve_chart(res), use_container_width=True)
        with st.expander("Show as table"):
            st.dataframe(pd.DataFrame([{"Years on sale": y, "Estimated owners": fmt_count(o)}
                                       for y, o in res["curve"]]), hide_index=True, use_container_width=True)

    all_rows = p.what_if(df, h)
    tested = {r["change"] for r in all_rows}
    done = [c for c in WHAT_IF_CHECKS if c not in tested]
    with right.container(border=True, key=f"panel_whatif_{key}"):
        st.markdown(f'<div class="section-title">{badge("bulb")} What moves the needle</div>'
                    '<div class="muted">One change at a time, everything else kept the same.</div>',
                    unsafe_allow_html=True)
        if all_rows:
            st.markdown(what_if_html(all_rows, res["owners"]), unsafe_allow_html=True)
        if done:
            st.markdown(done_html(done), unsafe_allow_html=True)
        st.caption("Patterns from similar games, not guarantees.")

    # --- studio + hype
    extras = []
    if res["studio"]:
        s = res["studio"]
        dev = (f"**{int(s['dev_prior_games'])}** earlier games, **{int(s['dev_prior_hits'])}** with 100k+ owners"
               if s["dev_prior_games"] else "First game on Steam (or name not found)")
        pub = (f"**{int(s['pub_prior_games'])}** earlier games, **{int(s['pub_prior_hits'])}** with 100k+ owners"
               if s["pub_prior_games"] else "No earlier games found")
        extras.append(("building", "Studio track record", f"- Developer: {dev}\n- Publisher: {pub}\n\n"
                       "A publisher's past hits are the strongest single signal the model uses."))
    if res["hype"]:
        hy = res["hype"]
        extras.append(("flame", "Pre-launch hype", f"**{hy['followers']:,}** Steam followers right now. With hype "
                       f"included, owners after {hy['horizon_days']:g} days: **{fmt_count(hy['owners'])}** "
                       f"(range {fmt_count(hy['owners_lo'])} - {fmt_count(hy['owners_hi'])}), "
                       f"x{hy['factor']:.2f} vs. the store-page-only estimate."))
    if extras:
        cols = st.columns(len(extras), gap="medium")
        for i, (col, (ico, title, body)) in enumerate(zip(cols, extras)):
            with col.container(border=True, key=f"panel_extra_{key}_{i}"):
                st.markdown(f'<div class="section-title">{badge(ico)} {title}</div>', unsafe_allow_html=True)
                st.markdown(body)


# ---------------------------------------------------------------------------- page
st.html(CSS)  # style-only: applied to the whole page
st.markdown(BACKGROUND, unsafe_allow_html=True)
# Pop-up on scroll: a tiny script (in a zero-height component iframe, same origin) watches the page
# and gives each section a bouncy entrance the first time it scrolls into view.
components.html("""<script>
(() => {
  const w = window.parent, d = w.document;
  if (w.__popInstalled) return; w.__popInstalled = true;
  if (w.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  const SEL = '[class*="st-key-panel"], .card, .tile, .game, .verdict, .title-wrap';
  const io = new w.IntersectionObserver(entries => {
    let i = 0;
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      const el = e.target; io.unobserve(el);
      el.style.animationDelay = (i++ * 0.09) + 's';
      el.classList.add('pop-in');
      el.addEventListener('animationend', () => {
        el.classList.remove('pop-ready', 'pop-in'); el.style.animationDelay = '';
      }, {once: true});
    }
  }, {threshold: 0.12, rootMargin: '0px 0px -40px 0px'});
  const scan = () => {
   // a frame can be inserted after its section was prepared: undo the pop for those right away
   d.querySelectorAll('.pop-ready').forEach(el => {
     if (el.querySelector('iframe')) { io.unobserve(el); el.classList.remove('pop-ready', 'pop-in'); }
   });
   d.querySelectorAll(SEL).forEach(el => {
    if (el.dataset.pop) return;
    el.dataset.pop = '1';
    // never animate sections that contain an embedded frame (the search box): it measures its own
    // width while the section is mid-bounce and would keep that wrong, oversized width
    if (el.querySelector('iframe')) return;
    el.classList.add('pop-ready'); io.observe(el);
   });
  };
  scan();
  new w.MutationObserver(scan).observe(d.body, {childList: true, subtree: true});
})();
</script>""", height=0)
p = get_predictor()
info = p.info

_words = ["GameOrb", "<span class='grad'>AI</span>"]
hero_words = " ".join(f'<span class="w" style="animation-delay:{.15 + i * .12:.2f}s">{w}</span>'
                      for i, w in enumerate(_words))
st.markdown(f"""
<div class="hero">
  <h1><span class="badge big">{icon('gamepad', 30)}</span>{hero_words}</h1>
  <p>How many people will own a Steam game, and how much could it earn? Look up any game on Steam,
  or describe one you're still planning, and get an estimate in seconds.</p>
  <span class="chip">Trained on {info['n_train']:,} Steam games</span>
  <span class="chip">Correct tier {info['tier_acc']:.0%} of the time</span>
  <span class="chip">Within one tier {info['tier_within1']:.0%}</span>
</div>
""", unsafe_allow_html=True)

tab_lookup, tab_plan, tab_about = st.tabs([":material/search: Look up a Steam game", ":material/construction: Plan a new game",
                                         ":material/insights: How it works"])

# ---- Tab 1: Steam lookup
def suggest_games(term: str):
    """Called on every keystroke by the search box (debounced): returns (label, appid) pairs."""
    term = (term or "").strip()
    appid = parse_appid(term)  # power users can still paste a store link or AppID
    if appid is not None:
        return [(f"Steam AppID {appid}", appid)]
    return search_cached(term.lower())


with tab_lookup:
    examples = {"Baldur's Gate 3": 1086940, "Terraria": 105600, "Stardew Valley": 413150,
                "Drawquarium (upcoming)": 4045480}

    def _pick_example():
        choice = st.session_state.get("example")
        if choice:
            st.session_state["pending_appid"] = examples[choice]
            st.session_state["example"] = None

    with st.container(border=True, key="panel_search"):
        st.markdown(f'<div class="section-title">{badge("search")} Search for a game on Steam</div>',
                    unsafe_allow_html=True)
        picked = st_searchbox(suggest_games, key="game_search",
                              placeholder="Start typing a game name, e.g. Hollow Knight",
                              debounce=300, clear_on_submit=False, edit_after_submit="option")
        st.markdown('<div class="muted" style="margin-top:-.4rem">Suggestions come from the Steam store, '
                    'including upcoming games. A store link or AppID works too.</div>', unsafe_allow_html=True)
        st.pills("Or try one of these", list(examples), key="example", on_change=_pick_example)

    # A new pick from the search box or an example pill starts a lookup (once per pick)
    appid = st.session_state.pop("pending_appid", None)
    if appid is None and picked and picked != st.session_state.get("searched_appid"):
        appid = picked
        st.session_state["searched_appid"] = picked

    if appid is not None:
        last = st.session_state.get("last_lookup", 0)
        if time.time() - last < SESSION_COOLDOWN_S:
            st.warning("Please wait a few seconds between lookups.")
        elif not allow_lookup():
            st.warning("Lots of people are looking up games right now, and Steam limits how fast we can "
                       "ask. Please try again in a minute.")
        else:
            st.session_state["last_lookup"] = time.time()
            with st.spinner("Reading the Steam store page..."):
                try:
                    game = fetch_game_cached(int(appid))
                except Exception as e:  # network trouble, Steam outage...
                    game = None
                    st.error(f"Couldn't reach Steam right now ({e}). Please try again shortly.")
            if not game:
                st.error("No Steam data found for that game. It may be region-locked, removed, or not a "
                         "game (e.g. a soundtrack or DLC).")
            else:
                st.session_state["lookup_result"] = (p.frame_from_game(game), game)
                st.toast(f"**{game.get('name', 'Game')}** · {price_label(game)}", icon=":material/sports_esports:")

    if "lookup_result" in st.session_state:
        df, game = st.session_state["lookup_result"]
        game_header(game)
        show_result(df, game.get("name", "this game"), key="lookup", appid=int(game["appid"]))
    else:
        st.markdown('<div class="muted" style="text-align:center;padding:2.5rem 0">Search for a game above '
                    'or pick an example to see a forecast.</div>', unsafe_allow_html=True)

# ---- Tab 2: manual planner
with tab_plan:
    st.markdown("Describe a game that isn't on Steam yet. **The first section matters most** - everything "
                "else is optional and starts with typical indie values.")
    model_cols = set(p.fb.feature_cols)
    tag_options = [t for t in TAGS if tag_column(t) in model_cols]
    genre_options = [g for g in STORE_GENRES if f"genre_store_{g.replace(' ', '_')}" in model_cols]
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
              "October", "November", "December"]
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    with st.form("plan", border=False):
        with st.container(border=True, key="panel_basics"):
            st.markdown(f'<div class="section-title">{badge("sliders")} 1 · The basics</div>', unsafe_allow_html=True)
            a, b, c = st.columns(3)
            name = a.text_input("Game name", "My game")
            developer = b.text_input("Developer", help="Separate several with ';'. Used to look up the "
                                                       "studio's earlier games on Steam.")
            publisher = c.text_input("Publisher", help="Leave empty if self-published.")
            a, b, c = st.columns(3)
            price = a.number_input("Launch price (USD)", 0.0, 100.0, 14.99, 1.0)
            is_free = a.checkbox("Free-to-play")
            genres = b.multiselect("Store genres", genre_options, format_func=str.title,
                                   default=["indie"] if "indie" in genre_options else [])
            tags = c.multiselect("Community tags", tag_options,
                                 default=["singleplayer"] if "singleplayer" in tag_options else [],
                                 help="Only tags the model has learned from are listed.")

        with st.container(border=True, key="panel_reach"):
            st.markdown(f'<div class="section-title">{badge("globe")} 2 · Reach & features</div>', unsafe_allow_html=True)
            a, b, c = st.columns(3)
            langs = a.multiselect("Languages", list(LANGUAGE_LABELS), default=["lang_english"],
                                  format_func=LANGUAGE_LABELS.get)
            platforms = b.multiselect("Platforms", ["windows", "mac", "linux"], default=["windows"],
                                      format_func=str.capitalize)
            feats = c.multiselect("Steam features", list(FEATURE_LABELS), default=["feat_singleplayer"],
                                  format_func=FEATURE_LABELS.get)
            a, b, c = st.columns(3)
            controller = a.selectbox("Controller support", ["none", "partial", "full"], format_func=str.capitalize)
            achievements = b.number_input("Steam achievements", 0, 500, 0)
            month = c.selectbox("Release month", months, index=2)
            a, b, _ = st.columns(3)
            has_demo = a.checkbox("Has a demo")
            mature = b.checkbox("Mature content (17+)")

        with st.expander("3 · Store page & description (optional, improves accuracy)"):
            a, b, c = st.columns(3)
            screenshots = a.number_input("Screenshots", 0, 50, 8)
            trailers = a.number_input("Trailers", 0, 20, 1)
            images = b.number_input("Images/GIFs in description", 0, 50, 2)
            website = b.checkbox("Has a website", value=True)
            ram = c.number_input("Minimum RAM (GB)", 0.0, 64.0, 4.0, 1.0)
            storage = c.number_input("Minimum storage (GB)", 0.0, 300.0, 2.0, 1.0,
                                     help="A strong signal: bigger installs usually mean bigger productions.")
            a, b, c = st.columns(3)
            other_langs = a.number_input("Other languages (count)", 0, 30, 0)
            audio = b.number_input("Languages with full voice-over", 0, 30, 0)
            dow = c.selectbox("Release weekday", weekdays, index=3)
            short_desc = st.text_area("Short description (the one-liner under the capsule)", max_chars=300)
            about = st.text_area("About this game (long description)", height=150, max_chars=8000)

        submitted = st.form_submit_button("Predict my game", icon=":material/auto_awesome:", type="primary", use_container_width=True)

    if submitted:
        spec = dict(name=name, developer=developer, publisher=publisher, is_free=is_free,
                    price=0.0 if is_free else price,
                    release_month=months.index(month) + 1, release_day_of_week=weekdays.index(dow),
                    genres=genres, tags=tags, languages=langs, other_languages=other_langs,
                    audio_languages=audio, platforms=platforms or ["windows"], controller=controller,
                    features=feats, achievements=achievements, has_demo=has_demo, mature=mature,
                    screenshots=screenshots, trailers=trailers, description_images=images,
                    has_website=website, min_ram_gb=ram or None, min_storage_gb=storage or None,
                    short_description=short_desc, about=about)
        st.session_state["plan_result"] = (p.frame_from_manual(spec), name or "your game")
    if "plan_result" in st.session_state:
        df, nm = st.session_state["plan_result"]
        st.divider()
        show_result(df, nm, key="plan")

# ---- Tab 3: about
with tab_about:
    t = info["temporal"]
    a, b, c, d = st.columns(4)
    a.markdown(card("Typical error", f"{10 ** info['mae_log10']:.1f}x", "either way, on a typical game", ico="gauge"),
               unsafe_allow_html=True)
    b.markdown(card("Ranking quality", f"{info['spearman']:.2f}", "Spearman correlation (1 = perfect order)", ico="trending"),
               unsafe_allow_html=True)
    c.markdown(card("Correct tier", f"{info['tier_acc']:.0%}", "random guessing gets ~25%", ico="check"),
               unsafe_allow_html=True)
    d.markdown(card("Within one tier", f"{info['tier_within1']:.0%}", "of all games tested", ico="layers"),
               unsafe_allow_html=True)
    st.write("")

    st.write("")
    st.markdown('<div class="title-wrap"><h2 class="big-title">How the <span class="grad">prediction</span> works</h2>'
                '<p class="big-sub">Everything the model sees is available before launch day.</p></div>',
                unsafe_allow_html=True)
    tiles = [
        ("database", "The data", f"{info['n_train']:,} released Steam games, sampled evenly across the four "
                                 "success tiers, with their store pages."),
        ("target", "What it predicts", "Estimated owners (reviews x 35, a common industry rule of thumb) after "
                                       "a chosen time on sale. Revenue = owners x launch price."),
        ("eye", "What it looks at", "Price, genres, tags, languages, platforms, Steam features, release timing, "
                                    "store-page content and the studio's track record across "
                                    f"{'82,500' if info['has_catalog'] else 'no'} catalog games."),
        ("eyeoff", "What it ignores", "Anything that only exists after launch - reviews, Metacritic, DLC - so "
                                      "it works for games that aren't out yet."),
        ("cpu", "The model", f"{info['model_name']}, chosen out of several candidates by repeated "
                             "5-fold cross-validation."),
        ("clock", "Tested on the future",
         (f"Trained on games up to 2022, tested on {t['n_test']} games from 2023-2024: correct tier "
          f"{t['tier_acc']:.0%} vs. {t['base_tier_acc']:.0%} for a naive guess.") if t else
         "Checked on games released after the training period."),
    ]
    st.markdown('<div class="tiles">' + "".join(
        f'<div class="tile" style="animation-delay:{.1 + i * .09:.2f}s">{badge(ic, 22)}'
        f'<h4>{esc(title)}</h4><p>{esc(body)}</p></div>' for i, (ic, title, body) in enumerate(tiles))
        + '</div>', unsafe_allow_html=True)
    st.write("")

    left, right = st.columns(2, gap="medium")
    with left.container(border=True, key="panel_tiers"):
        st.markdown(f'<div class="section-title">{badge("layers")} The four tiers</div>', unsafe_allow_html=True)
        st.markdown(tier_ladder(3, highlight=False), unsafe_allow_html=True)
        st.markdown("Each step shows owners / gross revenue. A game reaches a tier by owners **or** "
                    "revenue, whichever is higher.")
    with right.container(border=True, key="panel_limits"):
        st.markdown(f'<div class="section-title">{badge("shield")} Limitations</div>', unsafe_allow_html=True)
        st.markdown("""
- The biggest driver of sales - how good and how hyped a game is - is mostly invisible on a store
  page. A hype model based on daily follower tracking is being built.
- Owner counts are estimates derived from reviews, not real sales data.
- Breakout hits and big franchises are usually under-predicted.
- Read results as an order of magnitude, not a sales promise.
""")
    st.caption("Data: Steam Storefront API and SteamSpy. Not affiliated with Valve.")
