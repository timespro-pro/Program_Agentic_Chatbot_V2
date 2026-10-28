import base64
import html
import json
import re
import textwrap
from pathlib import Path

import pandas as pd
import streamlit as st
from openai import OpenAI

from agent import ask_agent
from source_analysis import analyze_sources, format_source_analysis_markdown
from batch_analysis import analyze_batch_history, format_batch_analysis_markdown
from overall_program_analysis import build_overall_program_analysis, format_overall_program_fallback
from analysis_cache import make_cache_key, get_cached_report, get_cache_diagnostic, put_cached_report, get_process_lock, cache_is_configured

from analysis_engine import (
    apply_source_filter,
    get_visual_report_data,
    _current_batch_codes,
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Program Performance Analysis Agent",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# SESSION STATE
# =========================================================

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "analysis_report" not in st.session_state:
    st.session_state.analysis_report = None

if "analysis_text" not in st.session_state:
    st.session_state.analysis_text = None

if "selected_program" not in st.session_state:
    st.session_state.selected_program = None

if "selected_old_batch" not in st.session_state:
    st.session_state.selected_old_batch = None

if "source_analysis_result" not in st.session_state:
    st.session_state.source_analysis_result = None

if "source_popup_batches" not in st.session_state:
    st.session_state.source_popup_batches = []

if "source_popup_new_batch" not in st.session_state:
    st.session_state.source_popup_new_batch = None


# =========================================================
# BACKGROUND IMAGE LOADER
# =========================================================
# Keep these two PNG files in the same folder as app.py
# (or inside an assets/ folder) when deploying.
# The exact supplied images are used as the UI background.

APP_DIR = Path(__file__).resolve().parent


def _load_background_data_uri(filename):
    candidates = [
        APP_DIR / filename,
        APP_DIR / "assets" / filename,
    ]

    for image_path in candidates:
        try:
            if image_path.exists():
                encoded = base64.b64encode(
                    image_path.read_bytes()
                ).decode("utf-8")

                return f"data:image/png;base64,{encoded}"
        except Exception:
            continue

    return None


LIGHT_BACKGROUND_IMAGE = _load_background_data_uri(
    "light_mode_background.png"
)

DARK_BACKGROUND_IMAGE = _load_background_data_uri(
    "dark_mode_background.png"
)

# =========================================================
# THEME
# =========================================================

if st.session_state.dark_mode:

    TEXT = "#F7F9FF"
    TEXT_SECONDARY = "#B4C6ED"

    PAGE_BACKGROUND = (
        f'url("{DARK_BACKGROUND_IMAGE}") center center / cover no-repeat fixed'
        if DARK_BACKGROUND_IMAGE
        else """
        radial-gradient(
            ellipse 75% 72% at 50% 48%,
            #0D2B73 0%,
            #08215B 40%,
            #061845 70%,
            #061331 100%
        )
        """
    )

    CARD_BACKGROUND = "rgba(23, 48, 105, 0.42)"
    CARD_BORDER = "rgba(161, 187, 238, 0.32)"

    INPUT_BACKGROUND = "rgba(75, 96, 148, 0.42)"
    INPUT_BORDER = "rgba(178, 198, 236, 0.48)"

    ICON_BACKGROUND = "rgba(34, 77, 166, 0.55)"

    SOURCE_BUTTON_BACKGROUND = "rgba(8, 29, 72, 0.82)"

    CHAT_BACKGROUND = "rgba(52, 76, 132, 0.42)"
    CHAT_BORDER = "rgba(169, 193, 235, 0.40)"
    CHAT_TEXT = "#FFFFFF"
    CHAT_PREVIEW = "rgba(235, 242, 255, 0.68)"
    CHAT_BUTTON = "#FFFFFF"

else:

    TEXT = "#09255D"
    TEXT_SECONDARY = "#526A98"

    PAGE_BACKGROUND = (
        f'url("{LIGHT_BACKGROUND_IMAGE}") center center / cover no-repeat fixed'
        if LIGHT_BACKGROUND_IMAGE
        else """
        radial-gradient(
            ellipse 31% 29% at 7% 100%,
            rgba(48, 91, 207, 0.96) 0%,
            rgba(48, 91, 207, 0.68) 42%,
            rgba(48, 91, 207, 0.00) 78%
        ),

        linear-gradient(
            153deg,
            transparent 0%,
            transparent 60%,
            rgba(205, 185, 212, 0.86) 60%,
            rgba(205, 185, 212, 0.86) 100%
        ),

        linear-gradient(
            45deg,
            transparent 0%,
            transparent 83%,
            rgba(226, 55, 82, 0.95) 83%,
            rgba(226, 55, 82, 0.95) 100%
        ),

        radial-gradient(
            ellipse 58% 42% at 46% 3%,
            rgba(194, 204, 225, 0.92) 0%,
            rgba(219, 225, 239, 0.72) 55%,
            rgba(255, 255, 255, 0) 100%
        ),

        #FFFFFF
        """
    )

    CARD_BACKGROUND = "rgba(255, 255, 255, 0.58)"
    CARD_BORDER = "rgba(90, 119, 180, 0.32)"

    INPUT_BACKGROUND = "rgba(255, 255, 255, 0.64)"
    INPUT_BORDER = "rgba(91, 120, 179, 0.42)"

    ICON_BACKGROUND = "rgba(211, 225, 253, 0.86)"

    SOURCE_BUTTON_BACKGROUND = "rgba(7, 27, 70, 0.90)"

    CHAT_BACKGROUND = "rgba(255, 255, 255, 0.48)"
    CHAT_BORDER = "rgba(80, 110, 175, 0.30)"
    CHAT_TEXT = "#09255D"
    CHAT_PREVIEW = "rgba(35, 62, 112, 0.68)"
    CHAT_BUTTON = "#1768DA"


# =========================================================
# GLOBAL CSS
# =========================================================

st.markdown(
    f"""
<style>

#MainMenu {{
    visibility: hidden;
}}

footer {{
    visibility: hidden;
}}

header {{
    background: transparent !important;
}}

[data-testid="stHeader"] {{
    background: transparent !important;
}}

[data-testid="stToolbar"] {{
    background: transparent !important;
}}

html,
body,
[data-testid="stApp"],
[data-testid="stAppViewContainer"] {{
    background:
        {PAGE_BACKGROUND}
        !important;

    background-attachment: fixed !important;
    background-position: center center !important;
    background-repeat: no-repeat !important;
    background-size: cover !important;

    min-height: 100vh !important;
}}

[data-testid="stAppViewContainer"] > .main {{
    background: transparent !important;
}}

[data-testid="stAppViewContainer"] > .main > div {{
    background: transparent !important;
}}

.block-container {{
    max-width: 1350px !important;

    padding-top: 5px !important;
    padding-bottom: 70px !important;
}}


/* =====================================================
   LOGO
===================================================== */

.top-logo {{
    display: flex;
    align-items: center;
    height: 58px;
}}

.top-logo img {{
    width: 205px !important;
    max-width: 205px !important;
    height: auto !important;
}}

.logo-fallback {{
    color: {TEXT};
    font-size: 1.8rem;
    font-weight: 900;
    letter-spacing: -0.04em;
}}

.logo-fallback span {{
    color: #FF1D2D;
}}


/* =====================================================
   THEME TOGGLE
===================================================== */

[data-testid="stToggle"] {{
    width: 178px !important;
    min-height: 48px !important;

    display: flex !important;
    align-items: center !important;

    padding: 4px 9px !important;

    border-radius: 30px !important;

    background:
        rgba(255,255,255,0.16) !important;

    border:
        1px solid
        rgba(255,255,255,0.48) !important;

    box-shadow:
        0 8px 24px
        rgba(0,0,0,0.08) !important;

    backdrop-filter:
        blur(18px) !important;

    -webkit-backdrop-filter:
        blur(18px) !important;
}}

[data-testid="stToggle"] label {{
    width: 100% !important;

    min-height: 40px !important;

    display: flex !important;

    flex-direction: row-reverse !important;

    align-items: center !important;

    justify-content: space-between !important;

    gap: 10px !important;

    margin: 0 !important;

    color: {TEXT} !important;

    font-size: 0.88rem !important;

    font-weight: 750 !important;
}}

[data-testid="stToggle"] label p {{
    color: {TEXT} !important;

    font-size: 0.88rem !important;

    font-weight: 750 !important;

    margin: 0 !important;
}}


/* =====================================================
   TITLE
===================================================== */

.main-title {{
    color: {TEXT} !important;

    font-size: 2.72rem !important;

    line-height: 1.06 !important;

    font-weight: 900 !important;

    letter-spacing: -0.043em !important;

    margin-top: 8px !important;

    margin-bottom: 4px !important;
}}

.main-subtitle {{
    color: {TEXT_SECONDARY} !important;

    font-size: 1.12rem;

    line-height: 1.35;

    font-weight: 500;

    margin-top: 0;

    margin-bottom: 10px;
}}

.brand-accent {{
    width: 225px;

    height: 4px;

    border-radius: 10px;

    background:
        linear-gradient(
            90deg,
            #087BFF 0%,
            #087BFF 30%,
            #6B2BC7 58%,
            #FF202D 100%
        );

    margin-top: 9px;

    margin-bottom: 25px;
}}


/* =====================================================
   BUSINESS OBJECTIVE
===================================================== */

.objective-card {{
    display: flex;

    align-items: center;

    gap: 22px;

    min-height: 98px;

    padding: 15px 24px;

    border-radius: 16px;

    background:
        {CARD_BACKGROUND};

    border:
        1px solid
        {CARD_BORDER};

    box-shadow:
        0 12px 38px
        rgba(0,25,80,0.07);

    backdrop-filter:
        blur(20px);

    -webkit-backdrop-filter:
        blur(20px);

    margin-bottom: 24px;
}}

.objective-icon {{
    width: 63px;
    height: 63px;
    min-width: 63px;

    display: flex;
    align-items: center;
    justify-content: center;

    border-radius: 14px;

    background:
        {ICON_BACKGROUND};

    font-size: 34px;
}}

.objective-content {{
    display: flex;
    flex-direction: column;
    gap: 5px;
}}

.objective-title {{
    color: #1768DA !important;

    font-size: 0.79rem;

    font-weight: 900;

    letter-spacing: 0.08em;

    text-transform: uppercase;
}}

.objective-text {{
    color: {TEXT} !important;

    font-size: 0.95rem;

    line-height: 1.4;

    font-weight: 500;
}}


/* =====================================================
   PROGRAM SELECTION
===================================================== */

.selection-header {{
    display: flex;
    align-items: center;

    gap: 17px;

    margin-top: 2px;
    margin-bottom: 4px;
}}

.selection-bar {{
    width: 5px;
    min-width: 5px;

    height: 31px;

    border-radius: 8px;

    background:
        linear-gradient(
            180deg,
            #FF1727,
            #FF3944
        );
}}

.selection-title {{
    color: {TEXT} !important;

    font-size: 0.94rem;

    font-weight: 900;

    letter-spacing: 0.075em;

    text-transform: uppercase;
}}

.selection-description {{
    color: {TEXT_SECONDARY} !important;

    font-size: 0.91rem;

    margin-left: 22px;

    margin-bottom: 14px;
}}


/* =====================================================
   BATCH HEADINGS
===================================================== */

.batch-heading {{
    display: flex;
    align-items: center;

    gap: 10px;

    margin-bottom: 7px;
}}

.batch-icon {{
    width: 43px;
    height: 43px;
    min-width: 43px;

    border-radius: 12px;

    display: flex;
    align-items: center;
    justify-content: center;

    background:
        {ICON_BACKGROUND};

    font-size: 20px;
}}

.batch-title {{
    color: {TEXT} !important;

    font-size: 1rem;

    font-weight: 850 !important;
}}


/* =====================================================
   SELECTORS — GLASS STYLE IN LIGHT AND DARK MODE
===================================================== */

/* Widget containers remain clear */
div[data-testid="stSelectbox"],
div[data-testid="stMultiSelect"],
div[data-testid="stSelectbox"] > div,
div[data-testid="stMultiSelect"] > div {{
    background: transparent !important;
    background-color: transparent !important;
}}

/* Main visible dropdown surface — translucent blue glass */
div[data-testid="stSelectbox"] [data-baseweb="select"],
div[data-testid="stMultiSelect"] [data-baseweb="select"] {{
    min-height: 57px !important;
    width: 100% !important;
    border-radius: 13px !important;
    background: rgba(115, 157, 238, 0.16) !important;
    background-color: rgba(115, 157, 238, 0.16) !important;
    border: 1px solid rgba(220, 233, 255, 0.42) !important;
    box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.12),
        0 8px 24px rgba(0, 10, 45, 0.16) !important;
    backdrop-filter: blur(22px) saturate(155%) !important;
    -webkit-backdrop-filter: blur(22px) saturate(155%) !important;
    transition: background 0.18s ease, border-color 0.18s ease,
                box-shadow 0.18s ease !important;
}}

/* Clear BaseWeb inner layers that otherwise appear opaque */
div[data-testid="stSelectbox"] [data-baseweb="select"] > div,
div[data-testid="stSelectbox"] [data-baseweb="select"] [role="combobox"],
div[data-testid="stMultiSelect"] [data-baseweb="select"] > div,
div[data-testid="stMultiSelect"] [data-baseweb="select"] [role="combobox"] {{
    background: transparent !important;
    background-color: transparent !important;
    box-shadow: none !important;
}}

/* Hover/focus preserve translucent glass */
div[data-testid="stSelectbox"] [data-baseweb="select"]:hover,
div[data-testid="stMultiSelect"] [data-baseweb="select"]:hover {{
    background: rgba(135, 173, 245, 0.22) !important;
    border-color: rgba(225, 237, 255, 0.68) !important;
}}

div[data-testid="stSelectbox"] [data-baseweb="select"]:focus-within,
div[data-testid="stMultiSelect"] [data-baseweb="select"]:focus-within {{
    border-color: rgba(205, 222, 255, 0.95) !important;
    box-shadow: 0 0 0 1px rgba(205, 222, 255, 0.28),
                0 10px 25px rgba(0, 15, 60, 0.13) !important;
}}

/* Readable labels, values, icons */
div[data-testid="stSelectbox"] [data-baseweb="select"] span,
div[data-testid="stSelectbox"] [data-baseweb="select"] input,
div[data-testid="stMultiSelect"] [data-baseweb="select"] span,
div[data-testid="stMultiSelect"] [data-baseweb="select"] input {{
    background: transparent !important;
    color: inherit !important;
    -webkit-text-fill-color: currentColor !important;
}}

div[data-testid="stSelectbox"] [data-baseweb="select"] svg,
div[data-testid="stMultiSelect"] [data-baseweb="select"] svg {{
    background: transparent !important;
    fill: currentColor !important;
}}
/* =====================================================
   DROPDOWN
===================================================== */

[data-baseweb="popover"] {{
    background: rgba(10, 30, 78, 0.88) !important;
    border: 1px solid rgba(210, 228, 255, 0.28) !important;
    border-radius: 13px !important;
    box-shadow: 0 18px 42px rgba(0, 8, 38, 0.32) !important;
    backdrop-filter: blur(22px) saturate(150%) !important;
    -webkit-backdrop-filter: blur(22px) saturate(150%) !important;
    overflow: hidden !important;
}}

[data-baseweb="popover"] [role="option"] {{
    color: #F4F7FF !important;
    background: transparent !important;
}}

[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="popover"] [role="option"][aria-selected="true"] {{
    color: #FFFFFF !important;
    background: rgba(112, 151, 235, 0.28) !important;
}}


/* =====================================================
   ACTION BUTTONS
===================================================== */

div[data-testid="stButton"] {{
    width: 100% !important;
}}

div[data-testid="stButton"]
> button {{
    min-height: 65px !important;

    width: 100% !important;

    border-radius: 13px !important;

    font-size: 0.94rem !important;

    font-weight: 800 !important;

    transition:
        transform 0.15s ease,
        filter 0.15s ease !important;
}}

div[data-testid="stButton"]
> button[kind="secondary"] {{
    background:
        {SOURCE_BUTTON_BACKGROUND} !important;

    color: #FFFFFF !important;

    border:
        1px solid
        rgba(157,184,237,0.65) !important;

    box-shadow:
        0 10px 25px
        rgba(0,15,60,0.13) !important;
}}

div[data-testid="stButton"]
> button[kind="secondary"] p {{
    color: #FFFFFF !important;
}}

div[data-testid="stButton"]
> button[kind="primary"] {{
    background:
        linear-gradient(
            105deg,
            #FF2938 0%,
            #FF3042 50%,
            #FF2533 100%
        ) !important;

    color: #FFFFFF !important;

    border: none !important;

    box-shadow:
        0 11px 28px
        rgba(255,31,49,0.24) !important;
}}

div[data-testid="stButton"]
> button[kind="primary"] p {{
    color: #FFFFFF !important;
}}

div[data-testid="stButton"]
> button:hover {{
    transform: translateY(-1px) !important;
}}


/* =====================================================
   STRUCTURED PERFORMANCE OUTPUT
===================================================== */

.comparison-header {{
    margin-top: 30px;
    margin-bottom: 17px;
}}

.comparison-title {{
    color: {TEXT};

    font-size: 1.18rem;

    font-weight: 900;

    line-height: 1.3;
}}

.comparison-title span {{
    color: {TEXT_SECONDARY};

    font-weight: 650;
}}

.comparison-subtitle {{
    color: {TEXT_SECONDARY};

    font-size: 0.78rem;

    margin-top: 8px;

    font-weight: 550;
}}


/* =====================================================
   KPI CARDS
===================================================== */

.metric-card {{
    min-height: 120px;

    padding: 17px 16px;

    border-radius: 15px;

    border: 1px solid
        rgba(255,255,255,0.32);

    backdrop-filter:
        blur(18px);

    -webkit-backdrop-filter:
        blur(18px);

    box-shadow:
        0 8px 25px
        rgba(0,25,80,0.06);

    margin-bottom: 12px;
}}

.metric-card.neutral {{
    background:
        rgba(255,255,255,0.34);
}}

.metric-card.danger {{
    background:
        rgba(255,238,241,0.78);

    border-color:
        rgba(255,70,80,0.35);
}}

.metric-card.success {{
    background:
        rgba(233,249,241,0.78);

    border-color:
        rgba(30,170,105,0.30);
}}

.metric-label {{
    color: {TEXT};

    font-size: 0.72rem;

    font-weight: 900;

    letter-spacing: 0.075em;

    margin-bottom: 10px;
}}

.metric-values {{
    display: flex;

    align-items: baseline;

    gap: 7px;

    white-space: nowrap;
}}

.metric-current {{
    color: {TEXT};

    font-size: 1.02rem;

    font-weight: 900;
}}

.metric-vs {{
    color: {TEXT_SECONDARY};

    font-size: 0.68rem;

    font-weight: 650;
}}

.metric-delta {{
    margin-top: 9px;

    font-size: 0.75rem;

    font-weight: 800;
}}

.metric-delta.danger {{
    color: #FF1727;
}}

.metric-delta.success {{
    color: #008B62;
}}

.metric-delta.neutral {{
    color: {TEXT_SECONDARY};
}}


/* =====================================================
   PERFORMANCE PANELS
===================================================== */

.performance-panel {{
    min-height: 145px;

    padding: 19px 20px;

    border-radius: 16px;

    margin-top: 23px;

    margin-bottom: 16px;

    backdrop-filter:
        blur(18px);

    -webkit-backdrop-filter:
        blur(18px);

    box-shadow:
        0 10px 30px
        rgba(0,25,80,0.06);
}}

.decline-panel {{
    background:
        rgba(255,230,234,0.84);

    border:
        1px solid
        rgba(255,72,90,0.34);
}}

.decline-label {{
    color: #FF1727 !important;
}}

.positive-panel {{
    background:
        rgba(229,248,239,0.84);

    border:
        1px solid
        rgba(0,153,100,0.25);
}}

.positive-label {{
    color: #008B62 !important;
}}

.neutral-panel {{
    background:
        rgba(255,255,255,0.42);

    border:
        1px solid
        rgba(100,130,180,0.25);
}}

.panel-label {{
    font-size: 0.73rem;

    font-weight: 900;

    letter-spacing: 0.075em;

    margin-bottom: 12px;
}}

.panel-title {{
    color: {TEXT};

    font-size: 0.95rem;

    font-weight: 850;

    line-height: 1.45;

    margin-bottom: 10px;
}}

.decline-panel .panel-title {{
    color: #A30F1D;
}}

.positive-panel .panel-title {{
    color: #007654;
}}

.panel-point {{
    color: #A30F1D;

    font-size: 0.75rem;

    line-height: 1.65;

    font-weight: 600;
}}


/* =====================================================
   AI BUSINESS DIAGNOSIS
===================================================== */

.ai-diagnosis-wrap {{
    margin-top: 26px;

    margin-bottom: 28px;
}}

.ai-diagnosis-header {{
    display: flex;

    align-items: center;

    gap: 11px;

    margin-bottom: 14px;
}}

.ai-diagnosis-icon {{
    width: 39px;

    height: 39px;

    min-width: 39px;

    display: flex;

    align-items: center;

    justify-content: center;

    border-radius: 11px;

    background:
        linear-gradient(
            135deg,
            rgba(31,111,235,0.20),
            rgba(123,45,190,0.20)
        );

    font-size: 20px;
}}

.ai-diagnosis-title {{
    color: {TEXT};

    font-size: 1.08rem;

    font-weight: 900;
}}

.ai-diagnosis-subtitle {{
    color: {TEXT_SECONDARY};

    font-size: 0.75rem;

    margin-top: 2px;
}}


/* =====================================================
   AI SECTION CARDS
===================================================== */

.ai-section {{
    border-radius: 15px;

    padding: 16px 18px;

    margin-bottom: 12px;

    background:
        {CARD_BACKGROUND};

    border:
        1px solid
        {CARD_BORDER};

    backdrop-filter:
        blur(17px);

    -webkit-backdrop-filter:
        blur(17px);

    box-shadow:
        0 7px 22px
        rgba(0,25,80,0.045);
}}

.ai-section-title {{
    color: {TEXT};

    font-size: 0.78rem;

    font-weight: 900;

    letter-spacing: 0.055em;

    text-transform: uppercase;

    margin-bottom: 9px;
}}

.ai-section-body {{
    color: {TEXT};

    font-size: 0.82rem;

    line-height: 1.58;
}}

.ai-section-body strong {{
    color: {TEXT};

    font-weight: 850;
}}

.ai-section-body ul {{
    margin-top: 5px;

    margin-bottom: 3px;

    padding-left: 19px;
}}

.ai-section-body li {{
    margin-bottom: 5px;
}}


/* =====================================================
   SPECIAL AI CARDS
===================================================== */

.ai-section.executive {{
    border-left:
        4px solid #1768DA;
}}

.ai-section.negative {{
    background:
        rgba(255,236,239,0.68);

    border:
        1px solid
        rgba(255,75,92,0.28);

    border-left:
        4px solid #FF2638;
}}

.ai-section.positive {{
    background:
        rgba(232,249,240,0.68);

    border:
        1px solid
        rgba(0,150,100,0.22);

    border-left:
        4px solid #00A06B;
}}

.ai-section.warning {{
    background:
        rgba(255,246,225,0.72);

    border:
        1px solid
        rgba(236,161,32,0.28);

    border-left:
        4px solid #E69A18;
}}

.ai-section.actions {{
    background:
        rgba(235,241,255,0.64);

    border:
        1px solid
        rgba(48,100,210,0.25);

    border-left:
        4px solid #316BDB;
}}


/* =====================================================
   AI INLINE METRICS
===================================================== */

.ai-number {{
    font-weight: 900;

    color: {TEXT};
}}

.ai-red {{
    color: #E31C2B;

    font-weight: 850;
}}

.ai-green {{
    color: #00865E;

    font-weight: 850;
}}


/* =====================================================
   DARK MODE AI
===================================================== */

[data-testid="stAppViewContainer"]
.metric-card {{
    background:
        rgba(35,61,120,0.48);

    border-color:
        rgba(165,191,237,0.28);
}}

[data-testid="stAppViewContainer"]
.metric-card.danger {{
    background:
        rgba(101,38,55,0.44);

    border-color:
        rgba(255,106,120,0.38);
}}

[data-testid="stAppViewContainer"]
.metric-card.success {{
    background:
        rgba(24,91,75,0.42);

    border-color:
        rgba(85,208,155,0.30);
}}

[data-testid="stAppViewContainer"]
.decline-panel {{
    background:
        rgba(101,34,49,0.55);

    border-color:
        rgba(255,100,115,0.38);
}}

[data-testid="stAppViewContainer"]
.decline-panel .panel-title,
[data-testid="stAppViewContainer"]
.decline-panel .panel-point {{
    color:
        #FF9AA5;
}}

[data-testid="stAppViewContainer"]
.positive-panel {{
    background:
        rgba(21,92,70,0.50);

    border-color:
        rgba(82,208,157,0.30);
}}

[data-testid="stAppViewContainer"]
.positive-panel .panel-title {{
    color:
        #7BE3BC;
}}

[data-testid="stAppViewContainer"]
.ai-section.negative {{
    background:
        rgba(101,34,49,0.43);
}}

[data-testid="stAppViewContainer"]
.ai-section.positive {{
    background:
        rgba(21,92,70,0.42);
}}

[data-testid="stAppViewContainer"]
.ai-section.warning {{
    background:
        rgba(105,77,29,0.40);
}}

[data-testid="stAppViewContainer"]
.ai-section.actions {{
    background:
        rgba(31,65,125,0.42);
}}


/* =====================================================
   CHAT SECTION
===================================================== */

.chat-shell {{
    margin-top: 32px;

    padding: 16px 19px 8px;

    border-radius: 16px;

    background:
        {CARD_BACKGROUND};

    border:
        1px solid
        {CARD_BORDER};

    backdrop-filter:
        blur(18px);

    -webkit-backdrop-filter:
        blur(18px);
}}

.chat-heading {{
    color: {TEXT} !important;

    font-size: 1rem;

    font-weight: 900;
}}

.chat-description {{
    color: {TEXT_SECONDARY} !important;

    font-size: 0.78rem;

    margin-top: 2px;
}}


/* =====================================================
   CHAT OUTER TRANSPARENCY
===================================================== */

[data-testid="stBottom"],
[data-testid="stBottomBlockContainer"],
.stChatFloatingInputContainer {{
    background:
        transparent !important;

    background-color:
        transparent !important;

    background-image:
        none !important;

    border:
        none !important;

    box-shadow:
        none !important;

    backdrop-filter:
        none !important;

    -webkit-backdrop-filter:
        none !important;
}}

[data-testid="stBottom"] > div,
[data-testid="stBottom"] > div > div,
[data-testid="stBottomBlockContainer"] > div,
.stChatFloatingInputContainer > div,
.stChatFloatingInputContainer > div > div {{
    background:
        transparent !important;

    background-color:
        transparent !important;

    border:
        none !important;

    box-shadow:
        none !important;
}}

[data-testid="stBottomBlockContainer"] {{
    padding-top: 8px !important;

    padding-bottom: 10px !important;
}}


/* =====================================================
   CHAT INPUT
===================================================== */

[data-testid="stChatInput"] {{
    background:
        transparent !important;

    background-color:
        transparent !important;

    border:
        none !important;

    box-shadow:
        none !important;

    padding:
        0 !important;

    width:
        calc(100% - 150px) !important;

    margin-left:
        auto !important;

    margin-right:
        auto !important;

    max-width:
        100% !important;
}}

[data-testid="stChatInput"] > div {{
    position:
        relative !important;

    width:
        100% !important;

    min-height:
        56px !important;

    border-radius:
        15px !important;

    background:
        {CHAT_BACKGROUND} !important;

    border:
        1px solid
        {CHAT_BORDER} !important;

    box-shadow:
        0 8px 28px
        rgba(0,20,70,0.08) !important;

    backdrop-filter:
        blur(20px)
        saturate(125%) !important;

    -webkit-backdrop-filter:
        blur(20px)
        saturate(125%) !important;

    overflow:
        hidden !important;
}}

[data-testid="stChatInput"] textarea {{
    background:
        transparent !important;

    color:
        {CHAT_TEXT} !important;

    -webkit-text-fill-color:
        {CHAT_TEXT} !important;

    border:
        none !important;

    outline:
        none !important;

    box-shadow:
        none !important;

    font-size:
        0.92rem !important;

    caret-color:
        {CHAT_TEXT} !important;
}}

[data-testid="stChatInput"]
textarea::placeholder {{
    color:
        transparent !important;

    opacity:
        0 !important;
}}

[data-testid="stChatInput"] > div::before {{
    content:
        "Why is CAC increasing for EPHM_05?";

    position:
        absolute;

    left:
        30px;

    top:
        50%;

    transform:
        translateY(-50%);

    pointer-events:
        none;

    z-index:
        5;

    color:
        {CHAT_PREVIEW};

    font-size:
        0.91rem;

    font-weight:
        500;

    white-space:
        nowrap;

    overflow:
        hidden;

    max-width:
        calc(100% - 105px);

    animation:
        chatQuestionCycle
        15s
        infinite;
}}

@keyframes chatQuestionCycle {{

    0% {{
        content:
            "Why is CAC increasing for EPHM_05?";
        opacity: 0;
    }}

    3% {{
        content:
            "Why is CAC increasing for EPHM_05?";
        opacity: 1;
    }}

    17% {{
        content:
            "Why is CAC increasing for EPHM_05?";
        opacity: 1;
    }}

    20% {{
        opacity: 0;
    }}

    20.01% {{
        content:
            "Which programs are driving the overall decline?";
        opacity: 0;
    }}

    23% {{
        opacity: 1;
    }}

    37% {{
        opacity: 1;
    }}

    40% {{
        opacity: 0;
    }}

    40.01% {{
        content:
            "Which acquisition sources are underperforming?";
        opacity: 0;
    }}

    43% {{
        opacity: 1;
    }}

    57% {{
        opacity: 1;
    }}

    60% {{
        opacity: 0;
    }}

    60.01% {{
        content:
            "Is the issue volume, conversion, CPL or CAC?";
        opacity: 0;
    }}

    63% {{
        opacity: 1;
    }}

    77% {{
        opacity: 1;
    }}

    80% {{
        opacity: 0;
    }}

    80.01% {{
        content:
            "Which programs are performing below plan?";
        opacity: 0;
    }}

    83% {{
        opacity: 1;
    }}

    97% {{
        opacity: 1;
    }}

    100% {{
        opacity: 0;
    }}
}}

[data-testid="stChatInput"]
> div:has(textarea:focus)::before {{
    opacity:
        0 !important;
}}

[data-testid="stChatInput"] button {{
    background:
        transparent !important;

    border:
        none !important;

    box-shadow:
        none !important;

    color:
        {CHAT_BUTTON} !important;
}}

[data-testid="stChatInput"] button svg {{
    color:
        {CHAT_BUTTON} !important;

    fill:
        {CHAT_BUTTON} !important;
}}


/* =====================================================
   MOBILE
===================================================== */

@media (max-width: 900px) {{

    .block-container {{
        padding-top:
            4px !important;
    }}

    .main-title {{
        font-size:
            2.1rem !important;
    }}

    .top-logo img {{
        width:
            170px !important;
    }}

    .objective-card {{
        gap:
            14px;

        padding:
            14px;
    }}

    [data-testid="stChatInput"] {{
        width:
            calc(100% - 40px) !important;
    }}

    [data-testid="stChatInput"] > div::before {{
        left:
            20px;

        font-size:
            0.82rem;

        max-width:
            calc(100% - 75px);
    }}
}}


/* =====================================================
   GLASS STATUS CARD OVERRIDE
   Glass surface + strong status border + inner color hue.
   Existing layout, spacing and structure unchanged.
===================================================== */

/* ---------------------------------------------------------
   SHARED GLASS SURFACE
--------------------------------------------------------- */

.metric-card,
.performance-panel,
.ai-section {{
    background:
        {CARD_BACKGROUND} !important;

    backdrop-filter:
        blur(20px)
        saturate(135%) !important;

    -webkit-backdrop-filter:
        blur(20px)
        saturate(135%) !important;

    box-shadow:
        inset 0 1px 0
            rgba(255,255,255,0.72),
        inset 0 -1px 0
            rgba(255,255,255,0.18),
        0 12px 34px
            rgba(0,25,80,0.08) !important;
}}


/* ---------------------------------------------------------
   KPI CARDS
--------------------------------------------------------- */

.metric-card {{
    border-width:
        2px !important;

    border-style:
        solid !important;

    border-radius:
        17px !important;

    position:
        relative;

    overflow:
        hidden;

    transition:
        transform 0.18s ease,
        box-shadow 0.18s ease,
        border-color 0.18s ease;
}}


/* subtle glass highlight */

.metric-card::after {{
    content:
        "";

    position:
        absolute;

    inset:
        0;

    pointer-events:
        none;

    border-radius:
        inherit;

    background:
        linear-gradient(
            135deg,
            rgba(255,255,255,0.22),
            rgba(255,255,255,0.00) 48%
        );

    opacity:
        0.65;
}}


/* Neutral / blue */

.metric-card.neutral {{
    border-color:
        #1768DA !important;

    box-shadow:
        inset 0 0 0 1px
            rgba(23,104,218,0.16),
        inset 0 0 28px
            rgba(23,104,218,0.075),
        inset 0 1px 0
            rgba(255,255,255,0.72),
        0 12px 34px
            rgba(0,25,80,0.08) !important;
}}

.metric-card.neutral .metric-label {{
    color:
        #1768DA !important;
}}


/* Negative / red */

.metric-card.danger {{
    border-color:
        #E31C2B !important;

    box-shadow:
        inset 0 0 0 1px
            rgba(227,28,43,0.16),
        inset 0 0 30px
            rgba(227,28,43,0.075),
        inset 0 1px 0
            rgba(255,255,255,0.72),
        0 12px 34px
            rgba(0,25,80,0.08) !important;
}}

.metric-card.danger .metric-label,
.metric-card.danger .metric-delta {{
    color:
        #E31C2B !important;
}}


/* Positive / green */

.metric-card.success {{
    border-color:
        #00A06B !important;

    box-shadow:
        inset 0 0 0 1px
            rgba(0,160,107,0.16),
        inset 0 0 30px
            rgba(0,160,107,0.075),
        inset 0 1px 0
            rgba(255,255,255,0.72),
        0 12px 34px
            rgba(0,25,80,0.08) !important;
}}

.metric-card.success .metric-label,
.metric-card.success .metric-delta {{
    color:
        #00865E !important;
}}


/* Warning / yellow */

.metric-card.warning {{
    border-color:
        #E69A18 !important;

    box-shadow:
        inset 0 0 0 1px
            rgba(230,154,24,0.16),
        inset 0 0 30px
            rgba(230,154,24,0.075),
        inset 0 1px 0
            rgba(255,255,255,0.72),
        0 12px 34px
            rgba(0,25,80,0.08) !important;
}}

.metric-card.warning .metric-label,
.metric-card.warning .metric-delta {{
    color:
        #C47B00 !important;
}}


/* KPI values stay readable */

.metric-current {{
    color:
        {TEXT} !important;
}}

.metric-vs {{
    color:
        {TEXT_SECONDARY} !important;
}}


/* KPI hover */

.metric-card:hover {{
    transform:
        translateY(-2px);

    box-shadow:
        inset 0 0 0 1px
            rgba(255,255,255,0.24),
        inset 0 0 32px
            rgba(255,255,255,0.045),
        0 16px 38px
            rgba(0,25,80,0.12) !important;
}}


/* ---------------------------------------------------------
   PERFORMANCE PANELS
--------------------------------------------------------- */

.performance-panel {{
    border-width:
        2px !important;

    border-style:
        solid !important;

    border-radius:
        18px !important;
}}


/* Red / deterioration */

.decline-panel {{
    background:
        linear-gradient(
            135deg,
            rgba(255,23,39,0.055),
            rgba(255,255,255,0.00) 62%
        ),
        {CARD_BACKGROUND} !important;

    border-color:
        #E31C2B !important;

    box-shadow:
        inset 0 0 0 1px
            rgba(227,28,43,0.15),
        inset 0 0 34px
            rgba(227,28,43,0.065),
        inset 0 1px 0
            rgba(255,255,255,0.72),
        0 12px 34px
            rgba(0,25,80,0.08) !important;
}}

.decline-panel .panel-label {{
    color:
        #E31C2B !important;
}}

.decline-panel .panel-title,
.decline-panel .panel-point {{
    color:
        #B51222 !important;
}}


/* Green / positive */

.positive-panel {{
    background:
        linear-gradient(
            135deg,
            rgba(0,160,107,0.055),
            rgba(255,255,255,0.00) 62%
        ),
        {CARD_BACKGROUND} !important;

    border-color:
        #00A06B !important;

    box-shadow:
        inset 0 0 0 1px
            rgba(0,160,107,0.15),
        inset 0 0 34px
            rgba(0,160,107,0.065),
        inset 0 1px 0
            rgba(255,255,255,0.72),
        0 12px 34px
            rgba(0,25,80,0.08) !important;
}}

.positive-panel .panel-label {{
    color:
        #00865E !important;
}}

.positive-panel .panel-title,
.positive-panel .panel-point {{
    color:
        #007654 !important;
}}


/* Blue / neutral */

.neutral-panel {{
    background:
        linear-gradient(
            135deg,
            rgba(23,104,218,0.045),
            rgba(255,255,255,0.00) 62%
        ),
        {CARD_BACKGROUND} !important;

    border-color:
        #1768DA !important;

    box-shadow:
        inset 0 0 0 1px
            rgba(23,104,218,0.13),
        inset 0 0 32px
            rgba(23,104,218,0.055),
        inset 0 1px 0
            rgba(255,255,255,0.72),
        0 12px 34px
            rgba(0,25,80,0.08) !important;
}}

.neutral-panel .panel-label {{
    color:
        #1768DA !important;
}}

.neutral-panel .panel-title,
.neutral-panel .panel-point {{
    color:
        {TEXT} !important;
}}


/* Yellow / warning */

.warning-panel {{
    background:
        linear-gradient(
            135deg,
            rgba(230,154,24,0.055),
            rgba(255,255,255,0.00) 62%
        ),
        {CARD_BACKGROUND} !important;

    border:
        2px solid #E69A18 !important;

    box-shadow:
        inset 0 0 0 1px
            rgba(230,154,24,0.14),
        inset 0 0 32px
            rgba(230,154,24,0.06),
        inset 0 1px 0
            rgba(255,255,255,0.72),
        0 12px 34px
            rgba(0,25,80,0.08) !important;
}}


/* ---------------------------------------------------------
   AI DIAGNOSIS CARDS
--------------------------------------------------------- */

.ai-section {{
    border-width:
        2px !important;

    border-style:
        solid !important;

    border-radius:
        17px !important;

    position:
        relative;

    overflow:
        hidden;
}}


/* Blue / executive */

.ai-section.executive {{
    background:
        linear-gradient(
            135deg,
            rgba(23,104,218,0.045),
            rgba(255,255,255,0.00) 65%
        ),
        {CARD_BACKGROUND} !important;

    border-color:
        #1768DA !important;

    border-left-width:
        5px !important;

    box-shadow:
        inset 0 0 0 1px
            rgba(23,104,218,0.13),
        inset 0 0 30px
            rgba(23,104,218,0.055),
        inset 0 1px 0
            rgba(255,255,255,0.72),
        0 10px 30px
            rgba(0,25,80,0.08) !important;
}}

.ai-section.executive .ai-section-title {{
    color:
        #1768DA !important;
}}


/* Red / negative */

.ai-section.negative {{
    background:
        linear-gradient(
            135deg,
            rgba(227,28,43,0.055),
            rgba(255,255,255,0.00) 65%
        ),
        {CARD_BACKGROUND} !important;

    border-color:
        #E31C2B !important;

    border-left-width:
        5px !important;

    box-shadow:
        inset 0 0 0 1px
            rgba(227,28,43,0.14),
        inset 0 0 30px
            rgba(227,28,43,0.065),
        inset 0 1px 0
            rgba(255,255,255,0.72),
        0 10px 30px
            rgba(0,25,80,0.08) !important;
}}

.ai-section.negative .ai-section-title {{
    color:
        #E31C2B !important;
}}

.ai-section.negative .ai-section-body strong {{
    color:
        #C51D2C !important;
}}


/* Green / positive */

.ai-section.positive {{
    background:
        linear-gradient(
            135deg,
            rgba(0,160,107,0.055),
            rgba(255,255,255,0.00) 65%
        ),
        {CARD_BACKGROUND} !important;

    border-color:
        #00A06B !important;

    border-left-width:
        5px !important;

    box-shadow:
        inset 0 0 0 1px
            rgba(0,160,107,0.14),
        inset 0 0 30px
            rgba(0,160,107,0.065),
        inset 0 1px 0
            rgba(255,255,255,0.72),
        0 10px 30px
            rgba(0,25,80,0.08) !important;
}}

.ai-section.positive .ai-section-title {{
    color:
        #00865E !important;
}}

.ai-section.positive .ai-section-body strong {{
    color:
        #00865E !important;
}}


/* Yellow / warning */

.ai-section.warning {{
    background:
        linear-gradient(
            135deg,
            rgba(230,154,24,0.055),
            rgba(255,255,255,0.00) 65%
        ),
        {CARD_BACKGROUND} !important;

    border-color:
        #E69A18 !important;

    border-left-width:
        5px !important;

    box-shadow:
        inset 0 0 0 1px
            rgba(230,154,24,0.14),
        inset 0 0 30px
            rgba(230,154,24,0.06),
        inset 0 1px 0
            rgba(255,255,255,0.72),
        0 10px 30px
            rgba(0,25,80,0.08) !important;
}}

.ai-section.warning .ai-section-title {{
    color:
        #C47B00 !important;
}}

.ai-section.warning .ai-section-body strong {{
    color:
        #B86F00 !important;
}}


/* Blue / actions */

.ai-section.actions {{
    background:
        linear-gradient(
            135deg,
            rgba(49,107,219,0.045),
            rgba(255,255,255,0.00) 65%
        ),
        {CARD_BACKGROUND} !important;

    border-color:
        #316BDB !important;

    border-left-width:
        5px !important;

    box-shadow:
        inset 0 0 0 1px
            rgba(49,107,219,0.13),
        inset 0 0 30px
            rgba(49,107,219,0.055),
        inset 0 1px 0
            rgba(255,255,255,0.72),
        0 10px 30px
            rgba(0,25,80,0.08) !important;
}}

.ai-section.actions .ai-section-title {{
    color:
        #1768DA !important;
}}

.ai-section.actions .ai-section-body strong {{
    color:
        #1768DA !important;
}}


/* AI body text */

.ai-section-body {{
    color:
        {TEXT} !important;
}}

.ai-section-body strong {{
    color:
        {TEXT} !important;
}}


/* ---------------------------------------------------------
   DARK MODE GLASS TUNING
   Keeps the same colored borders while using the dark
   background image underneath the translucent cards.
--------------------------------------------------------- */

[data-testid="stAppViewContainer"] .metric-card,
[data-testid="stAppViewContainer"] .performance-panel,
[data-testid="stAppViewContainer"] .ai-section {{
    box-shadow:
        inset 0 1px 0
            rgba(255,255,255,0.18),
        inset 0 -1px 0
            rgba(0,0,0,0.12),
        0 12px 34px
            rgba(0,0,0,0.18) !important;
}}

[data-testid="stAppViewContainer"] .metric-card::after {{
    background:
        linear-gradient(
            135deg,
            rgba(255,255,255,0.08),
            rgba(255,255,255,0.00) 48%
        );

    opacity:
        0.45;
}}


/* =========================================================
   PREMIUM KPI CARD TREATMENT
========================================================= */

.metric-card {{
    min-height: 128px !important;
    padding: 16px 16px 14px !important;
    border-radius: 17px !important;
    position: relative !important;
    overflow: hidden !important;
    border-width: 1.5px !important;

    background:
        linear-gradient(
            135deg,
            rgba(255,255,255,0.72) 0%,
            rgba(255,255,255,0.34) 48%,
            rgba(255,255,255,0.22) 100%
        ) !important;

    backdrop-filter: blur(22px) saturate(145%) !important;
    -webkit-backdrop-filter: blur(22px) saturate(145%) !important;

    box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.82),
        inset 0 -1px 0 rgba(255,255,255,0.20),
        0 10px 30px rgba(20,45,110,0.10) !important;

    transition: transform 0.18s ease, box-shadow 0.18s ease !important;
}}

.metric-card::before {{
    content: "";
    position: absolute;
    inset: -35% -12% -45% 28%;
    pointer-events: none;
    border-radius: 50%;
    filter: blur(22px);
    opacity: 0.80;
    z-index: 0;
}}

.metric-card::after {{
    content: "";
    position: absolute;
    inset: 0;
    pointer-events: none;
    border-radius: inherit;
    background: linear-gradient(
        125deg,
        rgba(255,255,255,0.34) 0%,
        rgba(255,255,255,0.00) 43%,
        rgba(255,255,255,0.10) 100%
    );
    z-index: 0;
}}

.metric-card > * {{
    position: relative;
    z-index: 2;
}}

.metric-card.danger {{
    border-color: rgba(255,57,76,0.72) !important;
    background: linear-gradient(
        135deg,
        rgba(255,255,255,0.76) 0%,
        rgba(255,242,246,0.46) 46%,
        rgba(255,210,221,0.30) 100%
    ) !important;
    box-shadow:
        inset 0 0 0 1px rgba(255,73,94,0.10),
        inset 0 0 38px rgba(255,73,94,0.085),
        inset 0 1px 0 rgba(255,255,255,0.90),
        0 10px 30px rgba(255,49,72,0.12) !important;
}}

.metric-card.danger::before {{
    background: radial-gradient(
        circle at 78% 76%,
        rgba(255,62,92,0.24) 0%,
        rgba(255,62,92,0.09) 36%,
        transparent 70%
    );
}}

.metric-card.success {{
    border-color: rgba(0,177,133,0.70) !important;
    background: linear-gradient(
        135deg,
        rgba(255,255,255,0.76) 0%,
        rgba(236,255,251,0.48) 46%,
        rgba(188,246,231,0.30) 100%
    ) !important;
    box-shadow:
        inset 0 0 0 1px rgba(0,171,130,0.10),
        inset 0 0 38px rgba(0,171,130,0.085),
        inset 0 1px 0 rgba(255,255,255,0.90),
        0 10px 30px rgba(0,164,126,0.11) !important;
}}

.metric-card.success::before {{
    background: radial-gradient(
        circle at 78% 76%,
        rgba(0,192,151,0.25) 0%,
        rgba(0,192,151,0.09) 36%,
        transparent 70%
    );
}}

.metric-card.neutral {{
    border-color: rgba(61,125,224,0.60) !important;
    background: linear-gradient(
        135deg,
        rgba(255,255,255,0.76) 0%,
        rgba(239,247,255,0.46) 48%,
        rgba(211,228,255,0.30) 100%
    ) !important;
    box-shadow:
        inset 0 0 0 1px rgba(61,125,224,0.08),
        inset 0 0 38px rgba(61,125,224,0.07),
        inset 0 1px 0 rgba(255,255,255,0.90),
        0 10px 30px rgba(30,80,160,0.10) !important;
}}

.metric-card.neutral::before {{
    background: radial-gradient(
        circle at 78% 76%,
        rgba(74,139,240,0.20) 0%,
        rgba(74,139,240,0.07) 36%,
        transparent 70%
    );
}}

.metric-topline {{
    display: flex;
    align-items: center;
    gap: 10px;
    min-height: 34px;
}}

.metric-icon {{
    width: 35px;
    height: 35px;
    min-width: 35px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    background: rgba(255,255,255,0.46);
    border: 1px solid rgba(255,255,255,0.72);
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.88),
        0 3px 10px rgba(30,50,100,0.06);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
}}

.metric-icon svg {{
    width: 18px;
    height: 18px;
    fill: none;
    stroke: currentColor;
    stroke-width: 1.9;
    stroke-linecap: round;
    stroke-linejoin: round;
}}

.metric-card.danger .metric-icon {{
    color: #EF1E32;
    background: rgba(255,238,242,0.62);
}}

.metric-card.success .metric-icon {{
    color: #008C69;
    background: rgba(227,252,246,0.62);
}}

.metric-card.neutral .metric-icon {{
    color: #1768DA;
    background: rgba(234,243,255,0.64);
}}

.metric-label {{
    margin: 0 !important;
    font-size: 0.72rem !important;
    font-weight: 900 !important;
    letter-spacing: 0.075em !important;
}}

.metric-values {{
    margin-top: 8px !important;
    gap: 8px !important;
}}

.metric-current {{
    font-size: 1.55rem !important;
    line-height: 1 !important;
    letter-spacing: -0.025em !important;
    font-weight: 900 !important;
}}

.metric-vs {{
    font-size: 0.72rem !important;
    font-weight: 650 !important;
}}

.metric-delta {{
    display: inline-flex !important;
    align-items: center !important;
    gap: 5px !important;
    width: fit-content !important;
    margin-top: 9px !important;
    padding: 5px 9px !important;
    border-radius: 999px !important;
    font-size: 0.73rem !important;
    font-weight: 850 !important;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
}}

.metric-card.danger .metric-delta {{
    background: rgba(255,70,91,0.11) !important;
}}

.metric-card.success .metric-delta {{
    background: rgba(0,173,132,0.11) !important;
}}

.metric-card.neutral .metric-delta {{
    background: rgba(23,104,218,0.09) !important;
}}

.metric-arrow {{
    font-size: 0.95rem;
    line-height: 1;
    font-weight: 900;
}}

.metric-trend {{
    position: absolute !important;
    right: 7px !important;
    bottom: -1px !important;
    width: 45% !important;
    height: 43px !important;
    opacity: 0.72;
    z-index: 1 !important;
    pointer-events: none;
}}

.metric-trend svg {{
    width: 100%;
    height: 100%;
    overflow: visible;
}}

.metric-trend .trend-line {{
    stroke: currentColor;
    stroke-width: 2.1;
    vector-effect: non-scaling-stroke;
}}

.metric-trend .trend-area {{
    opacity: 0.75;
}}

.metric-card.danger .metric-trend {{
    color: #F33B55;
}}

.metric-card.success .metric-trend {{
    color: #22B99B;
}}

.metric-card.neutral .metric-trend {{
    color: #4A91E8;
}}

.metric-card:hover {{
    transform: translateY(-2px) !important;
}}

.performance-panel {{
    min-height: 132px !important;
    border-radius: 18px !important;
    padding: 18px 20px 17px 20px !important;
    position: relative !important;
    overflow: hidden !important;
    backdrop-filter: blur(22px) saturate(140%) !important;
    -webkit-backdrop-filter: blur(22px) saturate(140%) !important;
}}

.performance-panel::after {{
    content: "";
    position: absolute;
    inset: 0;
    pointer-events: none;
    border-radius: inherit;
    background: linear-gradient(
        125deg,
        rgba(255,255,255,0.30),
        transparent 48%,
        rgba(255,255,255,0.08)
    );
}}

.decline-panel {{
    background: linear-gradient(
        135deg,
        rgba(255,255,255,0.70),
        rgba(255,224,231,0.32)
    ) !important;
}}

.positive-panel {{
    background: linear-gradient(
        135deg,
        rgba(255,255,255,0.70),
        rgba(207,250,241,0.32)
    ) !important;
}}

.neutral-panel {{
    background: linear-gradient(
        135deg,
        rgba(255,255,255,0.70),
        rgba(218,234,255,0.30)
    ) !important;
}}

.panel-label {{
    font-size: 0.72rem !important;
    font-weight: 900 !important;
    letter-spacing: 0.075em !important;
}}

.panel-title {{
    font-size: 0.94rem !important;
    font-weight: 800 !important;
    line-height: 1.45 !important;
}}

/* Dark-mode overrides are intentionally scoped below.
   Do NOT tint cards dark in light mode. */

@media (max-width: 900px) {{
    .metric-card {{
        min-height: 122px !important;
    }}

    .metric-current {{
        font-size: 1.35rem !important;
    }}

    .metric-trend {{
        width: 40% !important;
    }}
}}


.report-table-wrap {{ width:100%; max-width:100%; overflow-x:auto; margin:10px 0 14px; border-radius:10px; }}
.report-table {{ width:max-content; max-width:100%; table-layout:auto; border-collapse:collapse; font-size:0.88rem; background:rgba(8,25,83,0.35); }}
.report-table th, .report-table td {{ padding:6px 9px; text-align:left; vertical-align:top; border:1px solid rgba(160,190,255,0.28); }}
.report-table th {{ font-weight:800; background:rgba(70,110,210,0.28); white-space:normal; min-width:0; }}
.report-table tbody tr:nth-child(even) {{ background:rgba(255,255,255,0.035); }}
.report-table td {{ overflow-wrap:normal; white-space:nowrap; }}
.report-table th:first-child, .report-table td:first-child {{ white-space:normal; }}
</style>
""",
    unsafe_allow_html=True,
)



# =========================================================
# FINAL PREMIUM KPI CARD VISUAL OVERRIDE
# =========================================================

st.html(
    """
    <style>

    /* -----------------------------------------------------
       PREMIUM GLASS KPI CARDS
       White/translucent surface + strong status hue +
       colored mini area chart inside the lower-right.
    ----------------------------------------------------- */

    .metric-card {
        position: relative !important;
        min-height: 150px !important;
        padding: 18px 20px 17px 20px !important;
        overflow: hidden !important;
        isolation: isolate !important;
        border-radius: 18px !important;
        border: 1.5px solid rgba(255,255,255,0.72) !important;
        background: rgba(255,255,255,0.48) !important;
        backdrop-filter: blur(22px) saturate(145%) !important;
        -webkit-backdrop-filter: blur(22px) saturate(145%) !important;
        box-shadow:
            inset 0 1px 0 rgba(255,255,255,0.88),
            inset 0 0 26px rgba(255,255,255,0.22),
            0 12px 30px rgba(21,57,120,0.10) !important;
        transition: transform .18s ease, box-shadow .18s ease !important;
    }

    /* soft colored glass hue inside the card */
    .metric-card::before {
        content: "";
        position: absolute;
        inset: 0;
        border-radius: inherit;
        pointer-events: none;
        z-index: -2;
    }

    .metric-card.danger::before {
        background:
            radial-gradient(circle at 88% 84%, rgba(255,42,82,0.20), transparent 43%),
            linear-gradient(135deg, rgba(255,255,255,0.66), rgba(255,226,234,0.24));
    }

    .metric-card.success::before {
        background:
            radial-gradient(circle at 88% 84%, rgba(0,190,150,0.22), transparent 43%),
            linear-gradient(135deg, rgba(255,255,255,0.66), rgba(213,249,241,0.24));
    }

    .metric-card.neutral::before {
        background:
            radial-gradient(circle at 88% 84%, rgba(55,126,235,0.18), transparent 43%),
            linear-gradient(135deg, rgba(255,255,255,0.66), rgba(220,235,255,0.22));
    }

    /* colored status border */
    .metric-card.danger {
        border-color: #ff3650 !important;
        box-shadow:
            inset 0 0 0 1px rgba(255,54,80,0.16),
            inset 0 0 25px rgba(255,54,80,0.045),
            inset 0 1px 0 rgba(255,255,255,0.88),
            0 12px 30px rgba(255,54,80,0.10) !important;
    }

    .metric-card.success {
        border-color: #00a77a !important;
        box-shadow:
            inset 0 0 0 1px rgba(0,167,122,0.16),
            inset 0 0 25px rgba(0,167,122,0.045),
            inset 0 1px 0 rgba(255,255,255,0.88),
            0 12px 30px rgba(0,167,122,0.10) !important;
    }

    .metric-card.neutral {
        border-color: #2f79d8 !important;
    }

    .metric-card:hover {
        transform: translateY(-2px) !important;
        box-shadow:
            inset 0 1px 0 rgba(255,255,255,0.92),
            0 16px 36px rgba(21,57,120,0.14) !important;
    }

    /* keep actual content above the chart */
    .metric-topline,
    .metric-values,
    .metric-delta {
        position: relative !important;
        z-index: 4 !important;
    }

    .metric-topline {
        display: flex !important;
        align-items: center !important;
        gap: 10px !important;
        margin-bottom: 10px !important;
    }

    .metric-icon {
        width: 43px !important;
        height: 43px !important;
        min-width: 43px !important;
        border-radius: 50% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        background: rgba(255,255,255,0.42) !important;
        border: 1px solid rgba(255,255,255,0.78) !important;
        box-shadow:
            inset 0 1px 0 rgba(255,255,255,0.88),
            0 4px 12px rgba(20,50,100,0.08) !important;
    }

    .metric-icon svg {
        width: 21px !important;
        height: 21px !important;
        fill: none !important;
        stroke-width: 1.9 !important;
    }

    .metric-card.danger .metric-icon,
    .metric-card.danger .metric-icon svg {
        color: #e31c2b !important;
        stroke: #e31c2b !important;
    }

    .metric-card.success .metric-icon,
    .metric-card.success .metric-icon svg {
        color: #008d68 !important;
        stroke: #008d68 !important;
    }

    .metric-card.neutral .metric-icon,
    .metric-card.neutral .metric-icon svg {
        color: #1768da !important;
        stroke: #1768da !important;
    }

    .metric-label {
        font-size: .72rem !important;
        font-weight: 900 !important;
        letter-spacing: .075em !important;
    }

    .metric-card.danger .metric-label { color: #e31c2b !important; }
    .metric-card.success .metric-label { color: #008d68 !important; }
    .metric-card.neutral .metric-label { color: #1768da !important; }

    .metric-values {
        display: flex !important;
        align-items: baseline !important;
        gap: 7px !important;
        white-space: nowrap !important;
    }

    .metric-current {
        font-size: 1.48rem !important;
        font-weight: 900 !important;
        letter-spacing: -.025em !important;
        color: #092b68 !important;
    }

    .metric-vs {
        font-size: .69rem !important;
        font-weight: 700 !important;
        color: #46638f !important;
    }

    .metric-delta {
        display: inline-flex !important;
        align-items: center !important;
        gap: 6px !important;
        margin-top: 10px !important;
        padding: 7px 11px !important;
        border-radius: 999px !important;
        font-size: .72rem !important;
        font-weight: 850 !important;
        width: fit-content !important;
    }

    .metric-delta.danger {
        color: #e31c2b !important;
        background: rgba(255,40,65,0.10) !important;
    }

    .metric-delta.success {
        color: #008d68 !important;
        background: rgba(0,167,122,0.10) !important;
    }

    .metric-delta.neutral {
        color: #1768da !important;
        background: rgba(23,104,218,0.09) !important;
    }

    .metric-arrow {
        font-size: .9rem !important;
        font-weight: 950 !important;
    }

    /* -----------------------------------------------------
       MINI AREA CHART — the visual element from your sample
    ----------------------------------------------------- */

    .metric-chart {
        position: absolute !important;
        right: -1px !important;
        bottom: -1px !important;
        width: 55% !important;
        height: 68px !important;
        z-index: 2 !important;
        pointer-events: none !important;
        opacity: .90 !important;
        mask-image: linear-gradient(to right, transparent 0%, black 22%, black 100%);
        -webkit-mask-image: linear-gradient(to right, transparent 0%, black 22%, black 100%);
    }

    .metric-chart svg {
        width: 100% !important;
        height: 100% !important;
        overflow: visible !important;
    }

    .metric-chart .chart-line {
        fill: none !important;
        stroke-width: 2.4 !important;
        vector-effect: non-scaling-stroke !important;
        filter: drop-shadow(0 1px 3px rgba(0,0,0,.08));
    }

    .metric-chart .chart-area {
        opacity: .82 !important;
    }

    .metric-card.danger .metric-chart { color: #ff4262 !important; }
    .metric-card.success .metric-chart { color: #18b99a !important; }
    .metric-card.neutral .metric-chart { color: #3e8be7 !important; }

    .metric-card.danger .metric-chart .chart-line { stroke: #ff4262 !important; }
    .metric-card.success .metric-chart .chart-line { stroke: #18b99a !important; }
    .metric-card.neutral .metric-chart .chart-line { stroke: #3e8be7 !important; }

    /* dark mode: same structure, darker glass */
    body:has([data-testid="stAppViewContainer"] .dark-theme) .metric-card,
    .dark-theme .metric-card {
        background: rgba(8,22,55,0.52) !important;
        border-color: rgba(255,255,255,0.22) !important;
    }

    @media (max-width: 900px) {
        .metric-card { min-height: 145px !important; }
        .metric-current { font-size: 1.32rem !important; }
        .metric-chart { width: 50% !important; height: 58px !important; }
    }

    </style>
    """
)

# =========================================================
# HTML RENDER HELPER
# =========================================================

def render_html(content):
    st.html(
        textwrap.dedent(
            str(content)
        ).strip()
    )


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def clean_unique_values(values):

    result = []

    for value in values:

        if pd.isna(value):
            continue

        value = str(value).strip()

        if value:
            result.append(value)

    return sorted(set(result))


def _num(value, default=0.0):

    try:

        if pd.isna(value):
            return default

        return float(value)

    except Exception:

        return default


def _fmt_number(value):

    return f"{_num(value):,.0f}"


def _fmt_currency(value):

    return f"₹{_num(value):,.0f}"


def _fmt_pct(value):

    return f"{_num(value):+.1f}%"


def _movement(current, previous):

    current = _num(current)
    previous = _num(previous)

    difference = current - previous

    if previous == 0:

        pct = None

    else:

        pct = (
            difference
            / abs(previous)
            * 100
        )

    return difference, pct


def _performance_status(
    current,
    previous,
    higher_is_better=True,
):

    current = _num(current)
    previous = _num(previous)

    if current == previous:
        return "neutral"

    if higher_is_better:

        return (
            "improved"
            if current > previous
            else "deteriorated"
        )

    return (
        "improved"
        if current < previous
        else "deteriorated"
    )


# =========================================================
# CHECK TOOL PERFORMANCE
# =========================================================

def get_check_tool_performance(
    check_df,
    program_code,
):

    if check_df is None:
        return None

    if check_df.empty:
        return None

    if "program_code" not in check_df.columns:
        return None

    mask = (
        check_df["program_code"]
        .astype(str)
        .str.strip()
        .eq(
            str(program_code).strip()
        )
    )

    rows = check_df.loc[mask].copy()

    if rows.empty:
        return None


    def sum_column(column):

        if column not in rows.columns:
            return 0.0

        values = pd.to_numeric(
            rows[column],
            errors="coerce",
        )

        return values.sum(
            min_count=1
        )


    current_leads = sum_column("leads")
    current_applications = sum_column("applications")
    current_enrollments = sum_column("enrollments")
    current_spend = sum_column("spends")
    current_warm_leads = sum_column("warm_leads")

    previous_leads = sum_column(
        "previous_batch_leads"
    )

    previous_applications = sum_column(
        "previous_batch_applications"
    )

    previous_enrollments = sum_column(
        "previous_batch_enrollments"
    )

    previous_spend = sum_column(
        "previous_batch_spends"
    )

    previous_warm_leads = sum_column(
        "previous_batch_warm_leads"
    )


    current_cpl = (
        current_spend / current_leads
        if current_leads
        else 0
    )

    previous_cpl = (
        previous_spend / previous_leads
        if previous_leads
        else 0
    )


    current_cac = (
        current_spend / current_enrollments
        if current_enrollments
        else 0
    )

    previous_cac = (
        previous_spend / previous_enrollments
        if previous_enrollments
        else 0
    )


    current_lead_to_application = (
        current_applications
        / current_leads
        * 100
        if current_leads
        else 0
    )

    previous_lead_to_application = (
        previous_applications
        / previous_leads
        * 100
        if previous_leads
        else 0
    )


    current_application_to_enrollment = (
        current_enrollments
        / current_applications
        * 100
        if current_applications
        else 0
    )

    previous_application_to_enrollment = (
        previous_enrollments
        / previous_applications
        * 100
        if previous_applications
        else 0
    )


    current_lead_to_enrollment = (
        current_enrollments
        / current_leads
        * 100
        if current_leads
        else 0
    )

    previous_lead_to_enrollment = (
        previous_enrollments
        / previous_leads
        * 100
        if previous_leads
        else 0
    )


    return {

        "current": {

            "spend": current_spend,
            "leads": current_leads,
            "applications": current_applications,
            "enrollments": current_enrollments,
            "warm_leads": current_warm_leads,

            "cpl": current_cpl,
            "cac": current_cac,

            "lead_to_application":
                current_lead_to_application,

            "application_to_enrollment":
                current_application_to_enrollment,

            "lead_to_enrollment":
                current_lead_to_enrollment,
        },

        "previous": {

            "spend": previous_spend,
            "leads": previous_leads,
            "applications": previous_applications,
            "enrollments": previous_enrollments,
            "warm_leads": previous_warm_leads,

            "cpl": previous_cpl,
            "cac": previous_cac,

            "lead_to_application":
                previous_lead_to_application,

            "application_to_enrollment":
                previous_application_to_enrollment,

            "lead_to_enrollment":
                previous_lead_to_enrollment,
        },
    }


# =========================================================
# METRIC CARD
# =========================================================

def render_metric_card(
    title,
    current,
    previous,
    formatter,
    higher_is_better=True,
):

    difference, pct = _movement(
        current,
        previous,
    )

    status = _performance_status(
        current,
        previous,
        higher_is_better,
    )

    if status == "deteriorated":

        card_class = "metric-card danger"
        delta_class = "metric-delta danger"

    elif status == "improved":

        card_class = "metric-card success"
        delta_class = "metric-delta success"

    else:

        card_class = "metric-card neutral"
        delta_class = "metric-delta neutral"


    if pct is None:

        delta_text = f"{difference:+,.0f}"

    else:

        delta_text = (
            f"{difference:+,.0f}"
            f" ({pct:+.1f}%)"
        )


    icon_map = {
        "SPEND": """
            <svg viewBox="0 0 24 24" aria-hidden="true">
                <rect x="3.5" y="5.5" width="17" height="13" rx="2.5"></rect>
                <path d="M3.5 9.5h17"></path>
                <path d="M16 14h2.5"></path>
            </svg>
        """,
        "LEADS": """
            <svg viewBox="0 0 24 24" aria-hidden="true">
                <circle cx="12" cy="8" r="3.2"></circle>
                <path d="M5.5 19c.6-3.1 3-4.9 6.5-4.9s5.9 1.8 6.5 4.9"></path>
                <path d="M5.5 10.5a2.6 2.6 0 0 0-2 2.5"></path>
                <path d="M18.5 10.5a2.6 2.6 0 0 1 2 2.5"></path>
            </svg>
        """,
        "APPLICATIONS": """
            <svg viewBox="0 0 24 24" aria-hidden="true">
                <path d="M7 3.5h7l4 4V20.5H7z"></path>
                <path d="M14 3.5v4h4"></path>
                <path d="M10 11h5"></path>
                <path d="M10 14.5h5"></path>
                <path d="M10 18h3.5"></path>
            </svg>
        """,
        "ENROLLMENTS": """
            <svg viewBox="0 0 24 24" aria-hidden="true">
                <path d="M3.5 7.5 12 4l8.5 3.5L12 11z"></path>
                <path d="M6 9.5v5.5c2.5 2.1 9.5 2.1 12 0V9.5"></path>
                <path d="M20.5 8v6"></path>
            </svg>
        """,
        "CPL": """
            <svg viewBox="0 0 24 24" aria-hidden="true">
                <circle cx="10" cy="10" r="5.5"></circle>
                <path d="M14 14l6 6"></path>
                <path d="M8 10h4"></path>
                <path d="M10 8v4"></path>
            </svg>
        """,
        "CAC": """
            <svg viewBox="0 0 24 24" aria-hidden="true">
                <path d="M4 19.5V12"></path>
                <path d="M10 19.5V8"></path>
                <path d="M16 19.5V4.5"></path>
                <path d="M3 19.5h18"></path>
                <path d="m17 7 2-2 2 2"></path>
            </svg>
        """,
    }

    icon_svg = icon_map.get(
        str(title).upper(),
        """
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <circle cx="12" cy="12" r="8"></circle>
        </svg>
        """,
    )

    safe_id = re.sub(r"[^A-Za-z0-9_]", "_", str(title))

    render_html(
        f"""
        <div class="{card_class}">

            <div class="metric-topline">

                <div class="metric-icon">
                    {icon_svg}
                </div>

                <div class="metric-label">
                    {html.escape(str(title))}
                </div>

            </div>

            <div class="metric-values">

                <span class="metric-current">
                    {html.escape(str(formatter(current)))}
                </span>

                <span class="metric-vs">
                    vs {html.escape(str(formatter(previous)))}
                </span>

            </div>

            <div class="{delta_class}">
                <span class="metric-arrow">
                    {"↑" if difference >= 0 else "↓"}
                </span>
                {html.escape(delta_text)}
            </div>

            <div class="metric-chart" aria-hidden="true">
                <svg viewBox="0 0 150 46" preserveAspectRatio="none">
                    <defs>
                        <linearGradient id="trendFill_{safe_id}" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stop-color="currentColor" stop-opacity="0.22"></stop>
                            <stop offset="100%" stop-color="currentColor" stop-opacity="0"></stop>
                        </linearGradient>
                    </defs>
                    <path class="chart-area"
                        d="M0,42 C18,41 20,36 35,37 C49,38 54,27 68,29 C82,31 85,18 99,20 C111,22 118,9 130,13 C138,16 142,7 150,4 L150,46 L0,46 Z"
                        fill="url(#trendFill_{safe_id})">
                    </path>
                    <path class="chart-line"
                        d="M0,42 C18,41 20,36 35,37 C49,38 54,27 68,29 C82,31 85,18 99,20 C111,22 118,9 130,13 C138,16 142,7 150,4"
                        fill="none">
                    </path>
                </svg>
            </div>

        </div>
        """
    )


# =========================================================
# DETERMINISTIC PERFORMANCE REPORT
# =========================================================

def render_performance_report(
    performance,
    program_code,
    old_batch_code,
):

    if not performance:
        return


    current = performance["current"]
    previous = performance["previous"]


    render_html(
        f"""
        <div class="comparison-header">

            <div class="comparison-title">

                {html.escape(str(program_code))}

                <span>
                    • Current vs Old Batch
                </span>

            </div>

            <div class="comparison-subtitle">

                New Batch:
                <b>{html.escape(str(program_code))}</b>

                &nbsp;•&nbsp;

                Old Batch:
                <b>
                    {
                        html.escape(str(old_batch_code))
                        if old_batch_code
                        else "Unavailable"
                    }
                </b>

            </div>

        </div>
        """
    )


    row1 = st.columns(
        4,
        gap="small",
    )


    with row1[0]:

        render_metric_card(
            "SPEND",
            current["spend"],
            previous["spend"],
            _fmt_currency,
            True,
        )


    with row1[1]:

        render_metric_card(
            "LEADS",
            current["leads"],
            previous["leads"],
            _fmt_number,
            True,
        )


    with row1[2]:

        render_metric_card(
            "APPLICATIONS",
            current["applications"],
            previous["applications"],
            _fmt_number,
            True,
        )


    with row1[3]:

        render_metric_card(
            "ENROLLMENTS",
            current["enrollments"],
            previous["enrollments"],
            _fmt_number,
            True,
        )


    row2 = st.columns(
        [1, 1, 2],
        gap="small",
    )


    with row2[0]:

        render_metric_card(
            "CPL",
            current["cpl"],
            previous["cpl"],
            _fmt_currency,
            False,
        )


    with row2[1]:

        render_metric_card(
            "CAC",
            current["cac"],
            previous["cac"],
            _fmt_currency,
            False,
        )


    # -----------------------------------------------------
    # IDENTIFY MOVEMENTS
    # -----------------------------------------------------

    deteriorations = []
    improvements = []


    metrics = [

        (
            "CAC",
            current["cac"],
            previous["cac"],
            False,
        ),

        (
            "CPL",
            current["cpl"],
            previous["cpl"],
            False,
        ),

        (
            "Leads",
            current["leads"],
            previous["leads"],
            True,
        ),

        (
            "Applications",
            current["applications"],
            previous["applications"],
            True,
        ),

        (
            "Enrollments",
            current["enrollments"],
            previous["enrollments"],
            True,
        ),

        (
            "Lead → Application",
            current["lead_to_application"],
            previous["lead_to_application"],
            True,
        ),

        (
            "Application → Enrollment",
            current["application_to_enrollment"],
            previous["application_to_enrollment"],
            True,
        ),

        (
            "Lead → Enrollment",
            current["lead_to_enrollment"],
            previous["lead_to_enrollment"],
            True,
        ),
    ]


    for (
        name,
        current_value,
        previous_value,
        higher_is_better,
    ) in metrics:

        difference, pct = _movement(
            current_value,
            previous_value,
        )

        status = _performance_status(
            current_value,
            previous_value,
            higher_is_better,
        )

        item = {
            "name": name,
            "current": current_value,
            "previous": previous_value,
            "difference": difference,
            "pct": pct,
        }


        if status == "deteriorated":

            deteriorations.append(item)

        elif status == "improved":

            improvements.append(item)


    deteriorations.sort(
        key=lambda x:
        abs(x["pct"])
        if x["pct"] is not None
        else 0,
        reverse=True,
    )

    improvements.sort(
        key=lambda x:
        abs(x["pct"])
        if x["pct"] is not None
        else 0,
        reverse=True,
    )


    decline_col, positive_col = st.columns(
        [1.25, 0.85],
        gap="small",
    )


    # -----------------------------------------------------
    # PERFORMANCE DECLINED
    # -----------------------------------------------------

    with decline_col:

        if deteriorations:

            primary = deteriorations[0]

            pct = (
                primary["pct"]
                if primary["pct"] is not None
                else 0
            )


            cac_pct = (
                (
                    current["cac"]
                    - previous["cac"]
                )
                / previous["cac"]
                * 100
                if previous["cac"]
                else 0
            )

            cpl_pct = (
                (
                    current["cpl"]
                    - previous["cpl"]
                )
                / previous["cpl"]
                * 100
                if previous["cpl"]
                else 0
            )


            render_html(
                f"""
                <div class="performance-panel decline-panel">

                    <div class="panel-label decline-label">
                        PERFORMANCE DECLINED
                    </div>

                    <div class="panel-title">

                        {html.escape(str(primary["name"]))}
                        deteriorated by
                        {primary["difference"]:+,.0f}
                        ({pct:+.1f}%)
                        compared with
                        {html.escape(str(old_batch_code))}.

                    </div>

                    <div class="panel-point">
                        • CAC:
                        {current["cac"] - previous["cac"]:+,.0f}
                        ({cac_pct:+.1f}%)
                    </div>

                    <div class="panel-point">
                        • CPL:
                        {current["cpl"] - previous["cpl"]:+,.0f}
                        ({cpl_pct:+.1f}%)
                    </div>

                    <div class="panel-point">
                        • Lead → Enrollment:
                        {
                            current["lead_to_enrollment"]
                            - previous["lead_to_enrollment"]
                        :+.2f}
                        pp
                    </div>

                </div>
                """
            )

        else:

            render_html(
                """
                <div class="performance-panel neutral-panel">

                    <div class="panel-label">
                        PERFORMANCE
                    </div>

                    <div class="panel-title">
                        No material deterioration detected.
                    </div>

                </div>
                """
            )


    # -----------------------------------------------------
    # POSITIVE MOVEMENT
    # -----------------------------------------------------

    with positive_col:

        if improvements:

            primary = improvements[0]

            if "→" in primary["name"]:

                title = (
                    f"{primary['name']} improved by "
                    f"{primary['difference']:+.2f} pp"
                )

            else:

                pct = (
                    primary["pct"]
                    if primary["pct"] is not None
                    else 0
                )

                title = (
                    f"{primary['name']} improved by "
                    f"{primary['difference']:+,.0f}"
                    f" ({pct:+.1f}%)"
                )


            render_html(
                f"""
                <div class="performance-panel positive-panel">

                    <div class="panel-label positive-label">
                        POSITIVE MOVEMENT
                    </div>

                    <div class="panel-title">
                        {html.escape(title)}
                    </div>

                </div>
                """
            )

        else:

            render_html(
                """
                <div class="performance-panel neutral-panel">

                    <div class="panel-label">
                        POSITIVE MOVEMENT
                    </div>

                    <div class="panel-title">
                        No material positive movement detected.
                    </div>

                </div>
                """
            )


# =========================================================
# BUILD DETERMINISTIC CONTEXT FOR LLM
# =========================================================

def build_ai_analysis_context(
    performance,
    program_code,
    old_batch_code,
):

    if not performance:
        return ""

    c = performance["current"]
    p = performance["previous"]


    def comparison(
        name,
        current,
        previous,
        lower_is_better=False,
    ):

        difference, pct = _movement(
            current,
            previous,
        )

        if lower_is_better:

            status = (
                "IMPROVED"
                if difference < 0
                else
                "DETERIORATED"
                if difference > 0
                else
                "UNCHANGED"
            )

        else:

            status = (
                "IMPROVED"
                if difference > 0
                else
                "DETERIORATED"
                if difference < 0
                else
                "UNCHANGED"
            )

        return (
            f"{name}: "
            f"current={current:.2f}, "
            f"previous={previous:.2f}, "
            f"change={difference:+.2f}, "
            f"change_pct="
            f"{pct:+.2f}% "
            f"status={status}"
        )


    context = f"""
DETERMINISTIC BUSINESS PERFORMANCE CONTEXT

Program:
{program_code}

Current/New Batch:
{program_code}

Previous/Old Batch:
{old_batch_code or "Unavailable"}

IMPORTANT:
All numbers below are deterministic calculations.
Use these numbers directly.
Do not invent replacement values.
Do not average source-level CAC/CPL.
Do not claim causality unless the available evidence supports it.

VOLUME
{comparison("Leads", c["leads"], p["leads"])}

{comparison("Applications", c["applications"], p["applications"])}

{comparison("Enrollments", c["enrollments"], p["enrollments"])}

{comparison("Warm Leads", c["warm_leads"], p["warm_leads"])}

COST / EFFICIENCY
{comparison("Spend", c["spend"], p["spend"])}

{comparison("CPL", c["cpl"], p["cpl"], lower_is_better=True)}

{comparison("CAC", c["cac"], p["cac"], lower_is_better=True)}

FUNNEL CONVERSION
Current Lead → Application:
{c["lead_to_application"]:.2f}%

Previous Lead → Application:
{p["lead_to_application"]:.2f}%

Change:
{c["lead_to_application"] - p["lead_to_application"]:+.2f} pp

Current Application → Enrollment:
{c["application_to_enrollment"]:.2f}%

Previous Application → Enrollment:
{p["application_to_enrollment"]:.2f}%

Change:
{c["application_to_enrollment"] - p["application_to_enrollment"]:+.2f} pp

Current Lead → Enrollment:
{c["lead_to_enrollment"]:.2f}%

Previous Lead → Enrollment:
{p["lead_to_enrollment"]:.2f}%

Change:
{c["lead_to_enrollment"] - p["lead_to_enrollment"]:+.2f} pp
""".strip()

    return context


# =========================================================
# AI DIAGNOSIS PROMPT
# =========================================================

def build_ai_prompt(
    performance,
    program_code,
    old_batch_code,
):

    context = build_ai_analysis_context(
        performance,
        program_code,
        old_batch_code,
    )


    return f"""
You are a Senior Business Performance Analyst supporting Finance,
Marketing and management.

The objective is NOT to produce a generic summary.

Your job is to explain:
1. What changed?
2. How large is the change?
3. Where exactly did performance deteriorate?
4. Which funnel stage is responsible?
5. Is the issue primarily volume, conversion, CPL, CAC, or a combination?
6. What are the strongest positive signals?
7. What are the most important red flags?
8. What should Finance investigate?
9. What should Marketing investigate?
10. What actions could materially improve performance?

Use the deterministic data below.

{context}

STRICT ANALYSIS RULES:

- Use exact numbers from the deterministic context.
- Always compare Current/New Batch against Old Batch.
- Include absolute movement AND percentage movement wherever meaningful.
- For funnel conversion, use percentage-point movement.
- For CPL and CAC, lower is better.
- For leads, applications and enrollments, higher is better.
- Do not say performance improved simply because spend decreased.
- Distinguish lower spend from better efficiency.
- Do not invent source-level causality.
- Do not invent missing data.
- If source-level evidence is unavailable, explicitly say that source-level attribution needs investigation.
- Separate facts from hypotheses.
- Do not write a long narrative.
- Be analytical and management-oriented.
- Prioritize the 3-5 most important findings.
- Avoid repeating the KPI cards word-for-word.
- Explain the relationship between metrics.

IMPORTANT BUSINESS LOGIC:

If leads fall while spend also falls:
explain whether the fall in leads is proportionate to the spend reduction,
and use CPL to determine whether acquisition efficiency actually deteriorated.

If enrollments fall materially faster than leads:
investigate downstream conversion deterioration.

If CAC rises substantially:
identify whether the likely mathematical driver is lower enrollment volume,
higher spend, weaker funnel conversion, or a combination.

If CPL is stable but CAC increases:
highlight downstream conversion as the likely mathematical pressure point.

If CPL increases:
highlight acquisition efficiency deterioration.

If applications decline more than leads:
highlight Lead → Application conversion.

If enrollments decline more than applications:
highlight Application → Enrollment conversion.

Do NOT claim that a marketing channel, campaign, creative, audience,
sales team, or Finance process caused the change unless the supplied
data actually demonstrates it.

Return the analysis using EXACTLY these sections:

## Executive Read

Give 2-3 concise sentences explaining the overall business situation,
with numbers.

## Key Comparisons

Give 4-6 bullets containing the most decision-relevant comparisons.
Each bullet should contain numbers.

## Where Performance Deteriorated

Give 3-5 bullets identifying the largest negative movements.

Explain what each movement means for the business.

## Funnel Diagnosis

Explain:
- Lead → Application
- Application → Enrollment
- Lead → Enrollment

Identify the stage creating the biggest pressure.

Use percentage-point changes.

## Positive Signals

Give 2-4 bullets showing what is working or improving.

Do not force positives if there are none.

## Red Flags

Give 3-5 concise red flags.

Prioritize issues that could materially affect CAC, enrollment volume,
budget efficiency or future batch performance.

## Likely Drivers

Give 3-5 evidence-based drivers.

Clearly label anything that is a hypothesis rather than a proven fact.

Use wording such as:
"Evidence indicates..."
"Mathematically, this is driven by..."
"Hypothesis to investigate..."

## Finance Actions

Give 3-4 concrete actions Finance should investigate.

Examples:
- spend allocation
- budget pacing
- CAC efficiency
- cost concentration
- variance validation
- paid vs non-paid contribution

Only recommend things relevant to the available evidence.

## Marketing Actions

Give 3-4 concrete actions Marketing should investigate.

Examples:
- lead quality
- acquisition efficiency
- funnel conversion
- source mix
- campaign/channel performance

Do not invent channel problems without evidence.

## Priority Actions

End with the TOP 3 actions management should take first.

Format them as:

1. ACTION — WHY IT MATTERS
2. ACTION — WHY IT MATTERS
3. ACTION — WHY IT MATTERS

Keep the entire analysis concise enough for a management dashboard.
""".strip()


# =========================================================
# SAFE AI MARKDOWN → HTML
# =========================================================

def markdown_to_html(content):
    """Render report Markdown, including real HTML tables for Markdown tables."""
    if content is None:
        return ""

    lines = str(content).splitlines()
    output = []
    in_list = False
    in_table = False

    def inline_format(value):
        # Normalize any raw colored span emitted by the model BEFORE escaping.
        # Otherwise the browser displays literal <span ...> text in table cells.
        # Decode escaped HTML in cached/LLM reports before processing it.
        raw = html.unescape(str(value))

        def normalize_colored_span(match):
            attrs = match.group(1) or ""
            inner = re.sub(r"<[^>]+>", "", match.group(2) or "")
            # The requested convention is sign-based: + = green/up, - = red/down.
            sign_match = re.search(r"[+−-]\s*\d", inner)
            if sign_match:
                sign = sign_match.group(0).lstrip().replace("−", "-")[0]
                token = "[[UP]]" if sign == "+" else "[[DOWN]]"
                inner = re.sub(r"[↑↓↗↘]", "", inner, count=1).strip()
                return f"{token} {inner}"
            arrow_match = re.search(r"[↑↓↗↘]", inner)
            if arrow_match:
                arrow = arrow_match.group(0)
                token = "[[UP]]" if arrow in ("↑", "↗") else "[[DOWN]]"
                inner = inner.replace(arrow, "", 1).strip()
                return f"{token} {inner}"
            return inner

        raw = re.sub(
            r"<\s*span\b([^>]*)>(.*?)<\s*/\s*span\s*>",
            normalize_colored_span,
            raw,
            flags=re.IGNORECASE | re.DOTALL,
        )
        # Normalize existing plain arrows before handling signed percentages.
        # Their color follows the sign of the change, not the model's color hint.
        def normalize_plain_arrow(match):
            sign = match.group(2).replace("−", "-")
            token = "[[UP]]" if sign == "+" else "[[DOWN]]"
            return f"{token} {sign}"

        raw = re.sub(r"[↑↓↗↘]\s*([+−-])(?=\s*\d+(?:\.\d+)?\s*(?:%|pp)(?![A-Za-z]))", lambda m: ("[[UP]] " if m.group(1) == "+" else "[[DOWN]] ") + m.group(1), raw)

        # Also normalize plain signed percentage changes, including when the
        # model omits HTML entirely. Avoid adding a second arrow if one exists.
        def signed_change(match):
            # A normalized span may already have supplied the arrow token.
            prefix = raw[max(0, match.start() - 12):match.start()]
            if "[[UP]]" in prefix or "[[DOWN]]" in prefix or re.search(r"[↑↓↗↘]\s*$", prefix):
                return match.group(0)
            sign = match.group(1).replace("−", "-")
            token = "[[UP]]" if sign == "+" else "[[DOWN]]"
            return f"{token} {match.group(0)}"

        raw = re.sub(r"(?<![\w\]↑↓↗↘])([+−-])(?=\s*\d+(?:\.\d+)?\s*(?:%|pp)(?![A-Za-z]))", signed_change, raw)
        safe = html.escape(raw, quote=True)
        safe = safe.replace("[[UP]]", "<span style=\"color:#22c55e;font-weight:800;\">↑</span>")
        safe = safe.replace("[[DOWN]]", "<span style=\"color:#ef4444;font-weight:800;\">↓</span>")
        safe = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)
        safe = re.sub(r"`([^`]+)`", r"<code>\1</code>", safe)
        safe = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", safe)
        return safe

    def close_list():
        nonlocal in_list
        if in_list:
            output.append("</ul>")
            in_list = False

    def close_table():
        nonlocal in_table
        if in_table:
            output.append("</tbody></table></div>")
            in_table = False

    def is_table_row(value):
        value = value.strip()
        return value.startswith("|") and value.endswith("|") and value.count("|") >= 2

    def cells(value):
        return [cell.strip() for cell in value.strip().strip("|").split("|")]

    for raw in lines:
        line = raw.strip()
        if not line:
            close_list()
            close_table()
            continue

        # Markdown table row; separator row is consumed and not displayed.
        if is_table_row(line):
            close_list()
            row_cells = cells(line)
            if all(re.fullmatch(r":?-{3,}:?", c.replace(" ", "")) for c in row_cells):
                continue
            if not in_table:
                output.append("<div class='report-table-wrap'><table class='report-table'><thead><tr>")
                output.extend(f"<th>{inline_format(cell)}</th>" for cell in row_cells)
                output.append("</tr></thead><tbody>")
                in_table = True
            else:
                output.append("<tr>")
                output.extend(f"<td>{inline_format(cell)}</td>" for cell in row_cells)
                output.append("</tr>")
            continue
        close_table()

        if re.match(r"^[-•]\s+", line):
            if not in_list:
                output.append("<ul>")
                in_list = True
            item = re.sub(r"^[-•]\s+", "", line)
            output.append(f"<li>{inline_format(item)}</li>")
            continue

        close_list()
        if re.match(r"^\d+\.\s+", line):
            item = re.sub(r"^\d+\.\s+", "", line)
            output.append(f"<div style='margin-bottom:7px;'>{inline_format(item)}</div>")
        else:
            output.append(f"<div style='margin-bottom:7px;'>{inline_format(line)}</div>")

    close_list()
    close_table()
    return "\n".join(output)


# =========================================================
# RENDER AI DIAGNOSIS
# =========================================================

def render_ai_diagnosis(
    analysis_text,
):

    if not analysis_text:
        return


    text = str(
        analysis_text
    ).strip()


    text = re.sub(
        r"```(?:markdown|md|html)?",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = text.replace(
        "```",
        "",
    ).strip()

    # Keep percentage and percentage-point values consistent and readable.
    # Examples: 16.2918% -> 16.29%, -6.5910 pp -> -6.59 pp.
    def _round_percent_display(match):
        number = float(match.group(1).replace(",", ""))
        suffix = match.group(2)
        return f"{number:.2f}{suffix}"

    text = re.sub(
        r"(?<![\w.])(-?\d[\d,]*(?:\.\d+)?)\s*(%|pp\b|percentage points?\b)",
        _round_percent_display,
        text,
        flags=re.IGNORECASE,
    )

    render_html(
        """
        <div class="ai-diagnosis-wrap">

            <div class="ai-diagnosis-header">

                <div class="ai-diagnosis-icon">
                    ✦
                </div>

                <div>

                    <div class="ai-diagnosis-title">
                        AI Business Diagnosis
                    </div>

                    <div class="ai-diagnosis-subtitle">
                        Evidence-based interpretation of the
                        current vs old batch performance
                    </div>

                </div>

            </div>

        </div>
        """
    )


    sections = []

    current_title = None
    current_lines = []


    for line in text.splitlines():

        stripped = line.strip()

        if re.match(
            r"^#{1,6}\s+",
            stripped,
        ):

            title = re.sub(
                r"^#{1,6}\s+",
                "",
                stripped,
            ).strip()


            if current_title:

                sections.append(
                    (
                        current_title,
                        "\n".join(
                            current_lines
                        ).strip(),
                    )
                )


            current_title = title
            current_lines = []

        else:

            current_lines.append(line)


    if current_title:

        sections.append(
            (
                current_title,
                "\n".join(
                    current_lines
                ).strip(),
            )
        )

    else:

        sections = [
            (
                "Executive Read",
                text,
            )
        ]


    def section_class(title):

        normalized = (
            str(title)
            .lower()
            .strip()
        )


        if "executive" in normalized:

            return "executive"


        if (
            "deteriorat" in normalized
            or "negative" in normalized
            or "red flag" in normalized
        ):

            return "negative"


        if "positive" in normalized:

            return "positive"


        if (
            "action" in normalized
            or "priority" in normalized
        ):

            return "actions"


        if "driver" in normalized:

            return "warning"


        return ""


    for title, content in sections:

        if not content:
            continue


        css_class = section_class(
            title
        )


        body_html = markdown_to_html(
            content
        )


        render_html(
            f"""
            <div class="ai-section {css_class}">

                <div class="ai-section-title">
                    {html.escape(str(title))}
                </div>

                <div class="ai-section-body">
                    {body_html}
                </div>

            </div>
            """
        )

        # Add the historical-batch source tables button directly under this section.
        if "source-wise root-cause" in str(title).lower() or "source-wise analysis" in str(title).lower():
            if st.session_state.get("source_popup_batches", []):
                st.button(
                    "Click here to view other batches",
                    key="view_other_batch_source_tables",
                    type="secondary",
                    on_click=show_other_batch_source_tables,
                )


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_primary_data():
    """Load the unified dataset used by the overall analysis tool."""
    data_path = APP_DIR / "data" / "Program Batch Data.csv"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Unified data source not found: {data_path}. "
            "Place Batch_wise_program_data.csv inside the data/ folder."
        )
    return pd.read_csv(data_path, low_memory=False)


try:

    raw_df = load_primary_data()

except Exception as exc:

    st.error(
        f"Unable to load primary data: {exc}"
    )

    st.stop()


# The old Program Check Tool CSV has been retired. Keep this alias for
# backward-compatible calls in the existing UI helpers; it points to the
# same unified dataset, not a second data source.

check_df = raw_df


@st.cache_data
def load_advisor_mix():
    advisor_path = APP_DIR / "data" / "Advisor Mix.csv"
    if not advisor_path.exists():
        return pd.DataFrame()
    return pd.read_csv(advisor_path, low_memory=False)


advisor_df = load_advisor_mix()


# =========================================================
# HEADER
# =========================================================

header_left, header_right = st.columns(
    [7.5, 2.5],
    vertical_alignment="center",
)


with header_left:

    try:

        # Use the supplied TimesPro logo for the active theme.
        logo_filename = (
            "timespro_dark_mode_logo.png"
            if st.session_state.dark_mode
            else "timespro_light_mode_logo.png"
        )

        logo_candidates = [
            APP_DIR / logo_filename,
            APP_DIR / "assets" / logo_filename,
        ]

        logo_path = next(
            (p for p in logo_candidates if p.exists()),
            None,
        )

        if logo_path is None:
            raise FileNotFoundError(
                f"Logo not found: {logo_filename}"
            )

        st.image(
            str(logo_path),
            width=205,
        )

    except Exception:

        render_html(
            """
            <div class="logo-fallback">
                TIMES<span>PRO</span>
            </div>
            """
        )


with header_right:

    toggle_label = (
        "☾   Dark"
        if st.session_state.dark_mode
        else "☀   Light"
    )


    theme_switch = st.toggle(
        toggle_label,
        value=st.session_state.dark_mode,
        key="theme_switch",
    )


    if (
        theme_switch
        != st.session_state.dark_mode
    ):

        st.session_state.dark_mode = (
            theme_switch
        )

        st.rerun()


# =========================================================
# TITLE
# =========================================================

render_html(
    """
    <div class="main-title">
        Program Performance Analysis Agent
    </div>
    """
)


render_html(
    """
    <div class="main-subtitle">
        Business intelligence for Finance &amp; Marketing
    </div>
    """
)


render_html(
    """
    <div class="brand-accent"></div>
    """
)


# =========================================================
# BUSINESS OBJECTIVE
# =========================================================

render_html(
    """
    <div class="objective-card">

        <div class="objective-icon">
            🎯
        </div>

        <div class="objective-content">

            <div class="objective-title">
                Business Objective
            </div>

            <div class="objective-text">
                Assess business performance, pinpoint areas of deterioration,
                identify key drivers, and prioritize Finance and Marketing actions.
            </div>

        </div>

    </div>
    """
)


# =========================================================
# SOURCE FILTER
# =========================================================

filtered_result = apply_source_filter(
    raw_df,
    complete_rows_only=True,
)


if isinstance(
    filtered_result,
    tuple,
):

    df = filtered_result[0]

else:

    df = filtered_result


# =========================================================
# CURRENT PROGRAMS
# =========================================================

primary_programs = []

if "program_code" in df.columns:

    primary_programs = clean_unique_values(
        df["program_code"].tolist()
    )


check_programs = []

if (
    check_df is not None
    and "program_code" in check_df.columns
):

    check_programs = clean_unique_values(
        check_df["program_code"].tolist()
    )


try:

    current_batches = _current_batch_codes(
        df
    )

except Exception:

    current_batches = []


current_batches = clean_unique_values(
    list(current_batches)
)


current_batches = sorted(
    set(current_batches)
    |
    set(check_programs)
)


# =========================================================
# OLD BATCH RELATIONSHIP
# =========================================================

def get_old_batches(new_batch):
    """Return selectable historical batches for the selected program.

    Include (1) available batches in the explicit previous_batch_code lineage,
    and (2) earlier numbered batches with the same program prefix. The second
    rule allows valid historical comparisons when the data's previous-batch
    pointer skips a batch (for example, CQF_09 points to CQF_07, while CQF_08
    exists as another earlier CQF batch).
    """
    if not new_batch or str(new_batch).strip().lower() in {"", "select a program", "nan", "none", "<na>"}:
        return []

    relationship_frames = []
    all_codes = set()
    for frame in (check_df, df):
        if frame is None or frame.empty or "program_code" not in frame.columns:
            continue
        all_codes.update(
            str(value).strip()
            for value in frame["program_code"].dropna().tolist()
            if str(value).strip() and str(value).strip().lower() not in {"nan", "none", "<na>"}
        )
        if "previous_batch_code" in frame.columns:
            relationship_frames.append(frame[["program_code", "previous_batch_code"]].copy())

    new_batch = str(new_batch).strip()
    previous_by_program = {}
    if relationship_frames:
        relationships = pd.concat(relationship_frames, ignore_index=True)
        for program, previous in relationships.itertuples(index=False, name=None):
            if pd.isna(program) or pd.isna(previous):
                continue
            program, previous = str(program).strip(), str(previous).strip()
            if program.lower() in {"", "nan", "none", "<na>"} or previous.lower() in {"", "nan", "none", "<na>"}:
                continue
            previous_by_program.setdefault(program, [])
            if previous not in previous_by_program[program]:
                previous_by_program[program].append(previous)

    # Follow explicit historical links first.
    historical = []
    visited = {new_batch}
    frontier = [new_batch]
    while frontier:
        current = frontier.pop(0)
        for previous in previous_by_program.get(current, []):
            if previous in visited:
                continue
            visited.add(previous)
            if previous in all_codes:
                historical.append(previous)
                frontier.append(previous)

    # Also include earlier numbered batches sharing the same prefix, even if
    # the explicit pointer skips them. Example: CQF_09 -> CQF_07, but CQF_08
    # is still offered as a historical comparison because it exists in data.
    import re
    match = re.match(r"^(.*?)(\d+)$", new_batch)
    if match:
        prefix, current_number = match.group(1), int(match.group(2))
        same_family = []
        for code in all_codes:
            candidate = re.match(r"^(.*?)(\d+)$", code)
            if not candidate or candidate.group(1) != prefix:
                continue
            number = int(candidate.group(2))
            if number < current_number:
                same_family.append((number, code))
        # Newest earlier batches first; avoid duplicating lineage entries.
        for _, code in sorted(same_family, reverse=True):
            if code not in visited:
                historical.append(code)
                visited.add(code)

    return historical


# =========================================================
# PROGRAM SELECTION
# =========================================================

render_html(
    """
    <div class="selection-header">

        <div class="selection-bar"></div>

        <div class="selection-title">
            Program Selection
        </div>

    </div>
    """
)


render_html(
    """
    <div class="selection-description">
        Choose the program and batch to analyse performance.
    </div>
    """
)


new_col, old_col = st.columns(
    [1, 1],
    gap="large",
)


with new_col:

    render_html(
        """
        <div class="batch-heading">

            <div class="batch-icon">
                🎓
            </div>

            <div class="batch-title">
                New Batch
            </div>

        </div>
        """
    )


    new_options = (
        ["Select a program"]
        +
        current_batches
    )


    default_new = (
        st.session_state.selected_program
    )


    if default_new not in new_options:

        default_new = "Select a program"


    selected_new = st.selectbox(
        "New Batch",
        new_options,
        index=new_options.index(
            default_new
        ),
        label_visibility="collapsed",
        key="new_batch_selector",
    )

    # Clear the Old Batch selection whenever New Batch changes. Otherwise,
    # Streamlit session state can carry old selections into the next program
    # and they may be unintentionally included in the next analysis.
    previous_new_batch = st.session_state.get("_last_new_batch_for_old_selector")
    if previous_new_batch != selected_new:
        st.session_state["selected_old_batch"] = []
        if "old_batch_selector" in st.session_state:
            st.session_state["old_batch_selector"] = []
        st.session_state["_last_new_batch_for_old_selector"] = selected_new

    if selected_new != "Select a program":

        st.session_state.selected_program = (
            selected_new
        )


with old_col:

    render_html(
        """
        <div class="batch-heading">

            <div class="batch-icon">
                👥
            </div>

            <div class="batch-title">
                Old Batch
            </div>

        </div>
        """
    )


    if selected_new != "Select a program":

        old_options = get_old_batches(
            selected_new
        )

    else:

        old_options = []


    # Allow selecting more than one historical batch.
    if not old_options:
        old_options = []

    previous_selected = st.session_state.selected_old_batch
    if isinstance(previous_selected, str):
        previous_selected = [previous_selected] if previous_selected in old_options else []
    elif not isinstance(previous_selected, (list, tuple, set)):
        previous_selected = []
    else:
        previous_selected = [item for item in previous_selected if item in old_options]

    selected_old = st.multiselect(
        "Old Batch",
        options=old_options,
        default=previous_selected[:3],
        placeholder="Select up to 3 old batches",
        label_visibility="collapsed",
        key="old_batch_selector",
        max_selections=3,
    )

    # Keep session state aligned with the maximum allowed historical selection.
    selected_old = list(selected_old)[:3]
    st.session_state.selected_old_batch = selected_old



program_code = (
    None
    if selected_new == "Select a program"
    else selected_new
)


# Existing source-analysis functions compare against one previous batch at a time.
# Keep the full selection for the UI/session state and use the first selected
# batch for that single-batch comparison path. Batch Analysis independently
# traces and compares the available historical lineage.
old_batch_codes = list(dict.fromkeys(selected_old))[:3] if selected_old else []
old_batch_code = old_batch_codes[0] if old_batch_codes else None


# =========================================================
# SOURCE PREVIEW
# =========================================================

@st.dialog(
    "Source Data Preview",
    width="large",
)
def show_data_preview():

    st.caption("Unified source: data/Program Batch Data.csv")
    st.dataframe(
        df,
        width="stretch",
        height=500,
        hide_index=True,
    )


# =========================================================
# ANALYSIS TYPE
# =========================================================
# Analysis type selection is intentionally hidden. This tool always runs
# Overall program analysis.
analysis_type = "Overall program analysis"


# =========================================================
# ACTION BUTTONS
# =========================================================

render_html(
    """
    <div style="height:12px;"></div>
    """
)


button_left, button_middle, button_space = st.columns(
    [2.0, 2.0, 3.1],
    gap="small",
)


with button_left:

    preview_clicked = st.button(
        "▣   Source Data Preview",
        type="secondary",
        width="stretch",
        key="source_preview",
    )


with button_middle:

    run_analysis = st.button(
        "✦   Generate Analysis",
        type="primary",
        width="stretch",
        key="generate_analysis",
    )


if preview_clicked:

    show_data_preview()


# =========================================================
# GENERATE ANALYSIS
# =========================================================

if run_analysis:

    if analysis_type not in {"Source analysis", "Batch Analysis", "Overall program analysis"}:

        st.info(
            f"{analysis_type} is planned but not connected yet. "
            "Source analysis, Batch Analysis, and Overall program analysis are available now."
        )

    elif not program_code:

        st.warning("Please select a New Batch first.")

    elif analysis_type == "Overall program analysis":

        primary_data_path = APP_DIR / "data" / "Program Batch Data.csv"
        advisor_data_path = APP_DIR / "data" / "Advisor Mix.csv"
        selected_old_for_analysis = list(dict.fromkeys(
            code for code in old_batch_codes if code in old_options
        ))[:3]
        st.session_state.source_popup_batches = selected_old_for_analysis
        st.session_state.source_popup_new_batch = program_code
        cache_key = make_cache_key(
            program_code=program_code,
            primary_data_path=primary_data_path,
            advisor_data_path=advisor_data_path,
            prompt_version="overall-program-prompt-v4-selected-batches-up-to-four",
            selected_old_batches=selected_old_for_analysis,
        )

        with st.container():
            try:
                # The process-local lock avoids duplicate same-key OpenAI calls in this worker.
                # GitHub stores reports across users and app restarts.
                with get_process_lock(cache_key):
                    cached = get_cached_report(cache_key)
                    cache_diagnostic = get_cache_diagnostic()
                    if cached:
                        answer = cached["report"]
                        metadata = cached.get("metadata", {})
                        st.session_state.analysis_report = {
                            "found": True,
                            "analysis_type": "Overall program analysis",
                            "program_code": program_code,
                            "compared_batches": metadata.get("compared_batches", [program_code] + selected_old_for_analysis),
                            "selected_old_batches": metadata.get("selected_old_batches", selected_old_for_analysis),
                            "cache_status": "hit",
                        }
                        st.session_state.analysis_text = answer
                    else:
                        overall_result = build_overall_program_analysis(
                            raw_df=raw_df,
                            program_code=program_code,
                            advisor_df=advisor_df,
                            selected_old_batches=selected_old_for_analysis,
                        )
                        if overall_result.get("error"):
                            st.error(overall_result["error"])
                            st.stop()

                        # Compact JSON reduces prompt size and latency without changing evidence.
                        overall_payload = json.dumps(
                            overall_result,
                            ensure_ascii=False,
                            separators=(",", ":"),
                            default=str,
                        )
                        overall_prompt = f"""Write a concise, executive-ready Overall Program Analysis for {program_code}. Use ONLY the JSON evidence below; never invent data or causes. Missing values are N/A.

REQUIRED OUTPUT — preserve this approved section order and naming exactly; do not invent extra top-level sections:
1. “Executive Diagnosis + Summary of Reason”: start with one concise paragraph (maximum 2 sentences) stating whether the program is doing well, poorly, or mixed against plan and history. Follow with 4–6 short, evidence-backed bullets explaining the largest drivers.
Immediately after section 1, add a new section titled “Where it went wrong”. Keep this section focused on why the program is underperforming or mixed, using 4–5 numbered, concise evidence-backed points. Explicitly compare plan vs actual and the selected current batch vs the previous and previous-to-previous batches where available. Quantify the gaps with actual values and percentages/percentage points. Do not assume a metric is bad merely because it changed: interpret conversion rates as higher-is-better and CPL/CAC as lower-is-better. Distinguish observed facts from hypotheses, and say when evidence is insufficient. Include two transparent ratings when the required data exists: (a) “Plan attainment rating” based on the average attainment of available planned leads, applications, enrollments, and GSV metrics (cap each attainment at 100% for scoring; exclude unavailable plan metrics), with 90–100 = Good, 75–89 = Watch, below 75 = Poor; show the underlying attainment percentage and metrics included; and (b) “Historical comparison rating” based on direction-aware comparison of available leads, applications, enrollments, L2W, L2A, L2E, W2A, A2E, CPL, and CAC against each available prior batch (higher is better for volumes/conversion rates; lower is better for CPL/CAC). Label this Good if more than half of comparable metrics improved, Mixed if improvements and declines are tied or neither side has a majority, and Poor if more than half deteriorated. Show the counts improved/declined/unchanged and never calculate a rating when evidence is insufficient. Do not treat missing values as zero and do not fabricate an overall score if the needed fields are unavailable.
2. “Plan vs Actual”: a compact table with leads, applications, enrollments, observed spend vs planned budget, Actual GSV, Planned GSV, GSV variance, and plan attainment where available. Then explain the single most material gap in 1–2 sentences.
3. “Current vs previous batch”, containing exactly these subsections:
   - “3.1 Volume and commercial outcomes”: leads, warm leads, applications, enrollments, program fee, Actual GSV.
   - “3.2 Conversion and acquisition efficiency”: L2W, L2A, L2E, W2A, A2E, CPL, CAC.
   Compare the current batch with every selected Old Batch in the evidence (up to three old batches, four total). Include each selected batch as its own value column and comparison. Show absolute values in batch-value columns. Show percentage change for counts/costs and percentage-point (pp) change for rates. Arrow markers only: [[UP]] = green ↑ for positive mathematical change; [[DOWN]] = red ↓ for negative mathematical change. No circles/dots. Keep program fee neutral. Do not reverse arrows based on whether a metric is desirable; arrows indicate the sign of the change.
4. “Source-wise root-cause analysis”: provide a concise 3–5 line interpretation followed by a source table. Identify strongest/weakest sources only when supported by adequate data. If lead volume is not the issue but L2A declines, inspect L2W and W2A to locate the funnel leakage. Separate observations from hypotheses and disclose incomplete spend coverage.
5. “Advisor-tenure mix”: include only if data is available. Rank tenure groups using equal-weight score (L2W + L2E) / 2; discuss top/bottom groups with lead volumes and clarify that associations do not establish causation. Add a brief 1–2 sentence experience interpretation directly under this section: lower tenure-month values mean more experienced advisors, while higher tenure-month values mean less experienced advisors. Relate the observed performance of the tenure groups to this experience scale only where the table supports it; do not infer causation.
6. “Prioritized actions”: give 3–5 actionable steps tied to findings, including what Finance/Marketing should validate when evidence cannot establish a cause.

RULES: Actual GSV = enrollments × program-level fee (fee repeats across source rows; use once). Planned_GSV is in crore units and must be converted to rupees × 10,000,000; label this assumption. Spend may be incomplete; describe CPL/CAC as observed-spend metrics. Rates are ratios; show percentage-point changes. Do not add a separate Metric Scorecard; the specifically requested “Where it went wrong” section is required and should not be duplicated elsewhere. Do not invent missing metrics or causal explanations. Use valid Markdown tables. Aim for moderate detail: useful interpretation, not a terse summary or a long essay.

EVIDENCE JSON:
{overall_payload}
""".strip()

                        try:
                            client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
                            model = st.secrets.get("OPENAI_MODEL", "gpt-5.6")
                            response = client.responses.create(
                                model=model,
                                instructions=(
                                    "You are a senior marketing analytics consultant. Follow the requested report "
                                    "structure exactly. Return complete Markdown with numbered headings and valid Markdown "
                                    "tables. Never invent metrics, history, or causes. Use N/A when evidence is unavailable."
                                ),
                                input=overall_prompt,
                                max_output_tokens=5200,
                                reasoning={"effort": "low"},
                            )
                            answer = (response.output_text or "").strip()

                            # Validate the required new section before caching the report.
                            # If the model omits it, ask for only that missing section and insert it
                            # immediately after the Executive Diagnosis section.
                            if answer and "where it went wrong" not in answer.lower():
                                repair = client.responses.create(
                                    model=model,
                                    instructions=(
                                        "Write only the missing Markdown section requested. "
                                        "Use only the supplied evidence, do not invent values, "
                                        "and return a section heading plus 4-5 concise points."
                                    ),
                                    input=(
                                        f"Add a section titled 'Where it went wrong' for program {program_code}. "
                                        "Include 4-5 evidence-backed points comparing plan vs actual and current "
                                        "batch vs up to two prior batches where available. Include plan-attainment "
                                        "and historical-comparison ratings only when evidence is sufficient. "
                                        "Place this section after 'Executive Diagnosis + Summary of Reason'. "
                                        "Do not repeat other sections.\n\nEVIDENCE JSON:\n"
                                        + overall_payload
                                    ),
                                    max_output_tokens=1600,
                                    reasoning={"effort": "low"},
                                )
                                missing_section = (repair.output_text or "").strip()
                                if missing_section:
                                    # Insert directly after the first executive-diagnosis section block.
                                    next_heading = re.search(r"(?m)^##?\\s+(?:2[.)]?\\s*)?(?:Plan vs Actual|Planned versus actual)", answer, re.I)
                                    if next_heading:
                                        answer = answer[:next_heading.start()].rstrip() + "\\n\\n" + missing_section + "\\n\\n" + answer[next_heading.start():]
                                    else:
                                        answer = answer.rstrip() + "\\n\\n" + missing_section
                        except Exception:
                            # Deterministic fallback remains available if AI is unavailable.
                            answer = format_overall_program_fallback(overall_result)

                        answer = answer or format_overall_program_fallback(overall_result)
                        compared_batches = [
                            x.get("program_code") for x in overall_result.get("batches", [])
                        ]
                        # Cache only a complete non-empty report. A GitHub outage does not
                        # block the user; it simply means this result may not be shared.
                        cache_saved = put_cached_report(
                            cache_key,
                            answer,
                            metadata={
                                "program_code": program_code,
                                "compared_batches": [program_code] + selected_old_for_analysis,
                                "selected_old_batches": selected_old_for_analysis,
                                "analysis_type": "Overall program analysis",
                                "prompt_version": "overall-program-prompt-v4-selected-batches-up-to-four",
                            },
                        )
                        st.session_state.analysis_report = {
                            "found": True,
                            "analysis_type": "Overall program analysis",
                            "program_code": program_code,
                            "compared_batches": compared_batches,
                            "cache_status": "saved" if cache_saved else "not_saved",
                        }
                        st.session_state.analysis_text = answer

            except Exception as exc:
                st.session_state.analysis_report = {
                    "found": False,
                    "analysis_type": "Overall program analysis",
                    "message": str(exc),
                }
                st.session_state.analysis_text = None
                st.error(f"Overall program analysis failed: {exc}")

    elif analysis_type == "Batch Analysis":

        if not old_batch_codes:
            st.warning("Please select one or more Old Batches for comparison.")
            st.stop()

        with st.spinner("Analyzing the New Batch against your selected Old Batches..."):
            try:
                # Include exactly the user's selected old batches, up to three.
                # Combined with the selected new batch, the report covers at most four batches.
                selected_old_for_analysis = list(dict.fromkeys(
                    code for code in old_batch_codes if code in old_options
                ))[:3]
                batch_result = analyze_batch_history(
                    raw_df,
                    program_code=program_code,
                    max_batches=4,
                    selected_old_batches=selected_old_for_analysis,
                )
                batch_markdown = format_batch_analysis_markdown(batch_result)
                st.session_state.analysis_report = {
                    "found": True,
                    "analysis_type": "Batch Analysis",
                    "program_code": program_code,
                    "batch_lineage": batch_result.get("lineage_newest_first", []),
                    "selected_old_batches": selected_old_for_analysis,
                }
                batch_payload = json.dumps(batch_result, ensure_ascii=False, indent=2, default=str)
                # Keep the LLM context compact: the structured package is authoritative,
                # while the deterministic Markdown remains available as a complete fallback.
                prompt = f"""Create a concise, complete Batch Analysis for {program_code} using ONLY the JSON evidence below.

Rules:
- Python-calculated values are authoritative. Do not recalculate, invent numbers, or infer unsupported causes.
- Compare only program_code values that actually exist in the supplied dataset and are present in the provided batches/lineage.
- A previous_batch_code value that does not exist as a program_code is a dangling reference, NOT an available historical batch. Mention it only as "referenced but not found in dataset"; never add it as a batch row, metric column, or comparison.
- Treat missing values as unavailable, not zero. Flag incomplete spend coverage and selling-window differences where relevant.
- Separate observations from hypotheses; use pp for differences between percentages.
- Use compact, valid Markdown tables only where they add value. Do not reproduce every metric multiple times.
- Keep narrative concise: prioritize the largest quantified changes, persistent trends, reversals, plan gaps, and practical next actions.
- Analyze exactly the batches selected by the user: the selected New Batch plus each selected Old Batch, capped at the first 3 selected Old Batches. The total must never exceed 4 batches. If the user selects 1, 2, or 3 Old Batches, analyze only those selected batches; do not add extra historical batches. If more than 3 Old Batches are submitted by stale state or another caller, use only the first 3 selected Old Batches. Planned-versus-actual comparisons must cover every included batch where plan data exists.
- Explicitly analyze total_program_fee (or legacy total_program_fees, if present) as the program price point across batches: show fee changes in amount and percentage, and explain whether fee movement may affect GSV.
- Explicitly include the deterministic GSV impact decomposition when available: enrollment-volume effect versus fee/price effect. State whether it reconciles to observed GSV change; if not, explain that the attribution is incomplete and do not overstate it.
- Treat total_program_fee (or legacy total_program_fees) as a program-level value repeated across source rows, never sum it across rows.

Include all nine sections, with these exact headings:
1. Executive diagnosis
2. Batch lineage and comparability
3. Multi-batch performance scorecard
4. Metric-by-metric historical trends
5. Planned vs actual across batches
6. Price-point, GSV impact, hidden patterns and likely drivers
7. Root-cause hypotheses ranked by impact and confidence
8. Recommended actions
9. Final business conclusion

Deterministic evidence (JSON):
{batch_payload}
"""
                try:
                    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
                    model = st.secrets.get("OPENAI_MODEL", "gpt-5.6")
                    response = client.responses.create(
                        model=model,
                        instructions=(
                            "Return a complete executive-ready Batch Analysis in Markdown. "
                            "Include all nine required sections and properly rendered Markdown tables. Include program-fee trends and GSV impact decomposition in section 6 when data supports it. "
                            "Never omit sections; never invent numbers or causal claims."
                        ),
                        input=prompt,
                        max_output_tokens=4500,
                    )
                    generated = (response.output_text or "").strip()
                    required = [
                        "Executive diagnosis", "Batch lineage and comparability", "Multi-batch performance scorecard",
                        "Metric-by-metric historical trends", "Planned vs actual across batches",
                        "Price-point, GSV impact, hidden patterns and likely drivers", "Root-cause hypotheses ranked by impact and confidence",
                        "Recommended actions", "Final business conclusion",
                    ]
                    if generated and all(section.lower() in generated.lower() for section in required):
                        st.session_state.analysis_text = generated
                    else:
                        st.session_state.analysis_text = batch_markdown
                        if generated:
                            st.warning("The generated narrative omitted required sections, so the complete deterministic report is shown instead.")
                except Exception:
                    st.session_state.analysis_text = batch_markdown
            except Exception as exc:
                st.session_state.analysis_report = {
                    "found": False, "analysis_type": "Batch Analysis", "message": str(exc)
                }
                st.session_state.analysis_text = None
                st.error(f"Batch analysis failed: {exc}")

    elif not old_batch_code:

        st.warning(
            "Please select an Old Batch for comparison."
        )

    else:

        with st.spinner(
            "Validating source data and generating source-wise diagnosis..."
        ):

            try:

                # Use the unified raw dataset so missing source values and
                # spend gaps are preserved for the data-quality checks.
                source_result = analyze_sources(
                    raw_df,
                    program_code=program_code,
                    previous_batch_code=old_batch_code,
                )

                st.session_state.source_analysis_result = source_result
                st.session_state.analysis_report = {
                    "found": True,
                    "analysis_type": "Source analysis",
                    "program_code": program_code,
                    "previous_batch_code": old_batch_code,
                }

                # Send only the deterministic structured results to the LLM;
                # omit the DataFrame object used for the display table.
                source_payload = {
                    key: value
                    for key, value in source_result.items()
                    if key != "table"
                }

                source_context = json.dumps(
                    source_payload,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )

                source_report_prompt = f"""
ROLE: Senior Marketing Performance and Growth Analytics Consultant.

TASK:
Write an executive-ready source-wise diagnostic report for selected program
{program_code}, comparing it with previous batch {old_batch_code}.
Do NOT include a standalone section titled "Data quality and comparability".
Mention material data limitations briefly in the relevant section or as a short
final note only when they affect interpretation.

AUTHORITATIVE INPUT:
The deterministic Python source-analysis engine has already validated and
calculated the attached JSON. Treat its metrics, comparisons, data-quality
notes, and assumptions as the authoritative numerical evidence. Do not
recalculate or invent missing metrics. Do not treat unavailable values as zero.
Do not attribute repeated program-level paid metrics to individual sources.
Clearly distinguish source-level results from program-level paid metrics.

ANALYTICAL RULES:
- Validate data quality and batch comparability internally before interpreting results, but do not create a standalone Data quality and comparability section.
- Identify strongest and weakest sources using scale, conversion, economics,
  plan attainment, historical momentum, business impact, and data reliability.
- Diagnose lead qualification, warm-lead-to-application, application-to-
  enrollment, and lead-to-enrollment conversion.
- Quantify source contributions to changes only where the provided results
  support the calculation. Separate observations from counterfactual estimates.
- Assess plans only where valid source-level plans are supplied.
- Do not claim causal attribution from observational data.
- Every major finding should state evidence, interpretation, impact if
  calculable, confidence (high/medium/low), and missing evidence/assumptions.
- Recommendations should state issue, evidence, plausible explanation,
  action, success metric, and evaluation method. Do not invent targets,
  budget amounts, expected gains, owners, or deadlines.
- If the JSON does not contain a required calculation, state the limitation.

REPORT STRUCTURE — use exactly these eight sections in this order:
1. Executive diagnosis
2. Source performance scorecard
3. Funnel diagnostics
4. Previous-batch performance bridge
5. Planned vs actual performance
6. Paid vs non-paid performance
7. Prioritized root causes and recommended actions
8. Final business conclusion

Planned vs actual performance must compare each source's actual leads,
applications, enrollments, GSV and spend with its valid plan, showing actual,
plan, absolute variance and variance percentage in a Markdown table. Also
include the program-level paid plan vs paid actual comparison when available.
Do not classify missing plans as underperformance.

Paid vs non-paid performance must use the deterministic paid_vs_non_paid
object. Compare current and previous paid vs non-paid leads, enrollments, GSV,
spend, CAC/CPL and GSV-to-spend efficiency where valid. Include actual vs plan
for paid and non-paid segments where the supplied plan data supports it. State
clearly that matching paid fields are not supplied for applications or warm
leads, so do not invent that split. Do not attribute repeated program-level
paid metrics to individual source rows. If a split is unavailable due to missing
spend or inconsistent totals, show N/A and explain briefly.

Write precisely and commercially. Lead with business implications, use tables
for comparisons, avoid repetition, and keep facts separate from hypotheses.

DETERMINISTIC SOURCE-ANALYSIS PACKAGE:
{source_context}
""".strip()

                # Generate this report directly, without the legacy tool-calling
                # agent wrapper. That wrapper can divert the request into old
                # program diagnostics and return an incomplete/non-source report.
                client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
                model = st.secrets.get("OPENAI_MODEL", "gpt-5.6")
                response = client.responses.create(
                    model=model,
                    instructions=(
                        "You are a senior marketing performance and growth analytics consultant. "
                        "Follow the user's report structure exactly. Return complete Markdown, "
                        "with all eight sections and real Markdown tables. Do not output pipe-delimited "
                        "rows as plain text. Never omit a section. If evidence is unavailable, "
                        "include the section and explicitly state the limitation."
                    ),
                    input=source_report_prompt + "\n\nOUTPUT REQUIREMENTS:\n"
                    "- Include all nine numbered sections, each with a Markdown heading.\n"
                    "- Use Markdown tables for scorecards, bridges, plan comparisons, and priorities.\n"
                    "- Keep each table row and column properly separated with pipes and include a separator row.\n"
                    "- Complete every section; do not stop after the first few sections.\n"
                    "- Use specific calculated values from the JSON, with unavailable values shown as N/A.\n"
                    "- Avoid duplicating the entire input JSON in the report.",
                    max_output_tokens=9000,
                )
                answer = (response.output_text or "").strip()

                # Verify all required sections. If the first response was cut
                # short, ask the model to finish the missing sections.
                required_sections = [
                    "Executive diagnosis",
                    "Source performance scorecard",
                    "Funnel diagnostics",
                    "Previous-batch performance bridge",
                    "Planned vs actual performance",
                    "Paid vs non-paid performance",
                    "Prioritized root causes and recommended actions",
                    "Final business conclusion",
                ]
                normalized_answer = answer.lower()
                missing_sections = [
                    title for title in required_sections
                    if title.lower() not in normalized_answer
                ]
                if missing_sections:
                    completion = client.responses.create(
                        model=model,
                        instructions=(
                            "Complete an executive-ready source-wise diagnostic report. "
                            "Return only the missing sections as valid Markdown. Use the supplied "
                            "JSON as authoritative evidence; do not invent values or causes. Use Markdown tables."
                        ),
                        input=(
                            "The report started below but omitted these required sections: "
                            + ", ".join(missing_sections)
                            + "\n\nExisting report (do not repeat it):\n" + answer
                            + "\n\nOriginal deterministic source analysis and report requirements:\n"
                            + source_report_prompt
                        ),
                        max_output_tokens=6000,
                    )
                    completion_text = (completion.output_text or "").strip()
                    if completion_text:
                        answer = answer.rstrip() + "\n\n" + completion_text

                if not answer:
                    answer = format_source_analysis_markdown(source_result)

                st.session_state.analysis_text = answer

            except Exception as exc:

                st.session_state.analysis_report = {
                    "found": False,
                    "analysis_type": "Source analysis",
                    "message": str(exc),
                }
                st.session_state.analysis_text = None
                st.session_state.source_analysis_result = None
                st.error(f"Source analysis failed: {exc}")


# =========================================================
# PERFORMANCE CARDS
# =========================================================

if program_code and analysis_type not in {"Source analysis", "Batch Analysis", "Overall program analysis"}:

    performance = get_check_tool_performance(
        check_df,
        program_code,
    )


    if performance:

        render_performance_report(
            performance=performance,
            program_code=program_code,
            old_batch_code=old_batch_code,
        )


# =========================================================
# SOURCE-WISE HISTORICAL BATCH POPUP
# =========================================================

def show_other_batch_source_tables():
    """Show one source-wise table for each selected historical batch."""
    @st.dialog("Source-wise analysis — other selected batches", width="large")
    def _dialog():
        batch_codes = st.session_state.get("source_popup_batches", [])[:3]
        if not batch_codes:
            st.info("No historical batches were selected.")
            return

        st.caption(
            "Each table uses source-level data for that selected batch. "
            "The New Batch source table remains in the main report."
        )

        def num_sum(series):
            return pd.to_numeric(series, errors="coerce").sum(min_count=1)

        def ratio(numerator, denominator):
            if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
                return None
            return numerator / denominator * 100

        for batch_code in batch_codes:
            batch_rows = raw_df[
                raw_df["program_code"].astype(str).str.strip().eq(str(batch_code).strip())
            ].copy()
            st.markdown(f"### {batch_code}")

            if batch_rows.empty:
                st.info(f"No source-level rows were found for {batch_code}.")
                continue
            if "source_created" not in batch_rows.columns:
                st.warning("Source column `source_created` is not available in this dataset.")
                continue

            records = []
            for source, group in batch_rows.groupby("source_created", dropna=False):
                leads = num_sum(group["leads"]) if "leads" in group else None
                warm = num_sum(group["warm_leads"]) if "warm_leads" in group else None
                apps = num_sum(group["applications"]) if "applications" in group else None
                enrollments = num_sum(group["enrollments"]) if "enrollments" in group else None
                spend = num_sum(group["spends"]) if "spends" in group else None
                records.append({
                    "Source": str(source) if pd.notna(source) else "Unknown",
                    "Leads": leads,
                    "Warm leads": warm,
                    "Applications": apps,
                    "Enrollments": enrollments,
                    "L2W": ratio(warm, leads),
                    "L2A": ratio(apps, leads),
                    "W2A": ratio(apps, warm),
                    "L2E": ratio(enrollments, leads),
                    "A2E": ratio(enrollments, apps),
                    "Observed spend": spend,
                    "CPL": spend / leads if pd.notna(spend) and pd.notna(leads) and leads else None,
                    "CAC": spend / enrollments if pd.notna(spend) and pd.notna(enrollments) and enrollments else None,
                })

            table = pd.DataFrame(records)
            if table.empty:
                st.info(f"No source rows are available for {batch_code}.")
                continue

            table = table.sort_values("Enrollments", ascending=False, na_position="last")
            for column in ["L2W", "L2A", "W2A", "L2E", "A2E"]:
                table[column] = table[column].map(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
            for column in ["Observed spend", "CPL", "CAC"]:
                table[column] = table[column].map(lambda x: f"₹{x:,.2f}" if pd.notna(x) else "N/A")
            for column in ["Leads", "Warm leads", "Applications", "Enrollments"]:
                table[column] = table[column].map(lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A")
            st.dataframe(table, width="stretch", hide_index=True)

    _dialog()


# =========================================================
# AI BUSINESS DIAGNOSIS
# =========================================================

if st.session_state.analysis_text:

    render_ai_diagnosis(
        st.session_state.analysis_text
    )


# =========================================================
# CHAT SECTION
# =========================================================

render_html(
    """
    <div class="chat-shell">

        <div class="chat-heading">
            Chat with Agent
        </div>

        <div class="chat-description">
            Ask follow-up questions about programs, batches,
            sources, CAC, CPL, funnel performance,
            plan variance, or business drivers.
        </div>

    </div>
    """
)


# =========================================================
# OPTIMAL LAUNCH METRICS (SEPARATE MODULE; EXISTING UI OTHERWISE UNCHANGED)
# =========================================================

from optimal_launch_metrics import render_optimal_launch_metrics

render_optimal_launch_metrics(
    st=st,
    raw_df=raw_df,
    selected_code=program_code,
)




# =========================================================
# CHAT DATA SOURCES: GITHUB CACHED REPORT + EXCEL WORKBOOKS
# =========================================================

def _chat_dataframe_context(frame: pd.DataFrame, label: str, selected_code: str | None = None,
                            max_rows: int = 35) -> str:
    """Build a compact, grounded context block from a dataframe."""
    if frame is None or frame.empty:
        return f"Source: {label}\nNo rows available."
    view = frame.copy()
    if selected_code and "program_code" in view.columns:
        match = view["program_code"].astype(str).str.strip().eq(str(selected_code).strip())
        if match.any():
            view = view.loc[match]
    view = view.head(max_rows)
    try:
        table = view.to_csv(index=False)
    except Exception:
        table = view.astype(str).to_string(index=False)
    if len(table) > 14000:
        table = table[:14000] + "\n[truncated for context length]"
    return (
        f"Source: {label}\n"
        f"Rows in loaded source: {len(frame)}; rows provided below: {len(view)}\n"
        f"Columns: {', '.join(map(str, frame.columns))}\n"
        f"Data (CSV format):\n{table}"
    )


def _load_excel_chat_context(data_dir: Path, selected_code: str | None = None) -> str:
    """Read any .xlsx/.xls workbooks placed in data/ and expose their sheets to chat."""
    blocks = []
    if not data_dir.exists():
        return "No data directory found for Excel workbooks."
    paths = sorted(list(data_dir.glob("*.xlsx")) + list(data_dir.glob("*.xls")))
    if not paths:
        return "No .xlsx or .xls workbooks found in the app data/ directory."
    for path in paths:
        try:
            workbook = pd.ExcelFile(path)
            for sheet_name in workbook.sheet_names:
                frame = pd.read_excel(path, sheet_name=sheet_name)
                blocks.append(_chat_dataframe_context(
                    frame, f"Excel workbook {path.name} | sheet {sheet_name}",
                    selected_code=selected_code, max_rows=30
                ))
        except Exception as exc:
            blocks.append(f"Source: Excel workbook {path.name}\nCould not read workbook: {exc}")
    return "\n\n".join(blocks)


def _get_chat_cached_report(selected_code: str | None) -> str:
    """Fetch the current program's cached Overall Program Analysis report from GitHub."""
    if not selected_code:
        return "No program is selected; no program-specific cached report was requested."
    try:
        primary_data_path = APP_DIR / "data" / "Program Batch Data.csv"
        advisor_data_path = APP_DIR / "data" / "Advisor Mix.csv"
        key = make_cache_key(
            program_code=selected_code,
            primary_data_path=primary_data_path,
            advisor_data_path=advisor_data_path,
            prompt_version="overall-program-prompt-approved-v3-where-it-went-wrong",
        )
        cached = get_cached_report(key)
        if cached and cached.get("report"):
            return (
                f"GitHub cached Overall Program Analysis for program {selected_code}:\n"
                + str(cached["report"])[:24000]
            )
    except Exception:
        pass
    # When this session has already generated or retrieved a report, use it even if
    # the GitHub lookup is temporarily unavailable.
    session_report = st.session_state.get("analysis_text")
    report_meta = st.session_state.get("analysis_report") or {}
    if session_report and (
        not report_meta.get("program_code") or str(report_meta.get("program_code")) == str(selected_code)
    ):
        return f"Current session Overall Program Analysis for {selected_code}:\n{str(session_report)[:24000]}"
    return (
        f"No cached Overall Program Analysis was found for {selected_code}. "
        "Answer from the available spreadsheet/data context and do not pretend a report was retrieved."
    )


def _build_chat_extra_context(selected_code: str | None, data_frame: pd.DataFrame) -> str:
    parts = []
    parts.append(_get_chat_cached_report(selected_code))
    parts.append(_chat_dataframe_context(
        data_frame, "Program Batch Data.csv", selected_code=selected_code, max_rows=35
    ))
    try:
        parts.append(_chat_dataframe_context(
            advisor_df, "Advisor Mix.csv", selected_code=selected_code, max_rows=30
        ))
    except Exception:
        pass
    parts.append(_load_excel_chat_context(APP_DIR / "data", selected_code=selected_code))
    return "\n\n--- SOURCE SEPARATOR ---\n\n".join(parts)


# =========================================================
# CHAT HISTORY
# =========================================================

# Keep only the latest five messages (user + assistant messages combined).
st.session_state.chat_history = st.session_state.chat_history[-5:]

for message in st.session_state.chat_history:

    role = message.get(
        "role",
        "assistant",
    )

    content = message.get(
        "content",
        "",
    )


    with st.chat_message(role):

        st.markdown(
            content
        )


# =========================================================
# CHAT INPUT
# =========================================================

question = st.chat_input(
    " ",
    key="agent_chat_input",
)


# =========================================================
# CHAT PROCESSING
# =========================================================

if question:

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question,
        }
    )


    with st.chat_message("user"):

        st.markdown(
            question
        )


    with st.chat_message("assistant"):

        with st.spinner(
            "Analyzing..."
        ):

            try:

                answer = ask_agent(
                    question=question,
                    df=df,
                    program_code=program_code,
                    check_df=check_df,
                    chat_history=(
                        st.session_state.chat_history[:-1][-5:]
                    ),
                    extra_context=_build_chat_extra_context(
                        selected_code=program_code,
                        data_frame=raw_df,
                    ),
                )

            except TypeError:

                try:

                    answer = ask_agent(
                        question=question,
                        df=df,
                        program_code=program_code,
                    )

                except Exception as exc:

                    answer = (
                        f"Analysis failed: `{exc}`"
                    )

            except Exception as exc:

                answer = (
                    f"Analysis failed: `{exc}`"
                )


        st.markdown(
            answer
        )


    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )
    # Persist at most five messages in Streamlit session memory.
    st.session_state.chat_history = st.session_state.chat_history[-5:]