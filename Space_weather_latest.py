# Space Weather Dashboard — BOM-style Overview + Severity + Impact + PDF colors
# ---------------------------------------------------------------------------
# pip install: streamlit requests plotly numpy scipy fpdf pyspaceweather kaleido (optional)
# ---------------------------------------------------------------------------

import os
import base64
import re
from datetime import datetime
import tempfile

import plotly.graph_objects as go
import requests
import streamlit as st
import numpy as np
from scipy.ndimage import uniform_filter1d
from fpdf import FPDF
from streamlit.components.v1 import html

# ========== Setup ==========
st.set_page_config(page_title="Space Weather Dashboard", layout="wide")
UA = {"User-Agent": "SpaceWeatherDashboard/4.0 (+streamlit)"}

# ---------- BOM Aurora API ----------
try:
    from pyspaceweather import SpaceWeather
    HAVE_BOM = True
except Exception as e:
    HAVE_BOM = False
    BOM_ERR = f"pyspaceweather unavailable: {e}"

BOM_API_KEY = os.getenv("BOM_API_KEY", "").strip()
# --- DEVELOPMENT ONLY: Hardcoded BOM key (replace in prod) ---
if not BOM_API_KEY:
    BOM_API_KEY = "enert bom api"  # TODO: replace for your environment
# -------------------------------------------------------------

if HAVE_BOM and BOM_API_KEY:
    try:
        bom = SpaceWeather(BOM_API_KEY)
    except Exception as e:
        bom = None
        BOM_ERR = f"pyspaceweather init failed: {e}"
else:
    bom = None
    if HAVE_BOM and not BOM_API_KEY:
        BOM_ERR = "BOM_API_KEY env var not set"

# ========== Sidebar: Settings ==========
with st.sidebar:
    st.markdown("## Dashboard Settings")
    refresh_min = st.slider("Auto-refresh (minutes)", 0, 30, 10)
    if refresh_min > 0:
        interval_ms = int(refresh_min * 60 * 1000)
        html(f"<script>setTimeout(function(){{ window.location.reload(); }}, {interval_ms});</script>", height=0)
    high_contrast = st.toggle("High-contrast mode", True)
    font_scale = st.slider("Font scale", 1.0, 1.6, 1.2, 0.05)
    label_style = st.selectbox("Badge label style", ["Text + Color (default)", "Text-only"])
    st.markdown("### BOM API Status")
    if bom:
        st.success("BOM aurora enabled.")
    else:
        st.info("BOM aurora disabled.")
        if 'BOM_ERR' in globals() and BOM_ERR:
            st.caption(BOM_ERR)
        st.caption("To enable: export BOM_API_KEY in your environment.")

# ========== Helper functions ==========
@st.cache_data(ttl=600, show_spinner=True)
def fetch_json(url, timeout=20):
    try:
        r = requests.get(url, timeout=timeout, headers=UA)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.warning(f"Failed to load {url}: {e}")
        return None

@st.cache_data(ttl=600, show_spinner=True)
def fetch_text(url, timeout=20):
    try:
        r = requests.get(url, timeout=timeout, headers=UA)
        r.raise_for_status()
        return r.text
    except Exception as e:
        st.warning(f"Failed to load {url}: {e}")
        return ""

def clamp_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        try:
            return float(str(x).strip())
        except Exception:
            return default

def r_scale(xray_flux_wm2):
    if xray_flux_wm2 >= 1e-4: return ("R2", "high")
    if xray_flux_wm2 >= 1e-5: return ("R1", "med")
    return ("R0", "low")

def s_scale(proton_pfu_10mev):
    if proton_pfu_10mev >= 10: return ("S2", "high")
    if proton_pfu_10mev >= 1: return ("S1", "med")
    return ("S0", "low")

def g_scale_from_kp(kp):
    k = clamp_float(kp, 0)
    if k >= 7: return ("G3", "veryhigh")
    if k >= 5: return ("G2", "high")
    if k >= 4: return ("G1", "med")
    return ("G0", "low")

def last_updated():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

# ---------- NZ-flavoured, plain-English rewrite helpers ----------
_NZ_REGIONAL_HINT = (
    " (NZ/South Pacific focus)"
)

def _any(txt: str, *phrases) -> bool:
    low = (txt or "").lower()
    return any(p in low for p in phrases)

def _nz_risk_phrase(kind: str, level: str) -> str:
    if kind == "R":
        return {
            "ok": "HF comms across NZ should be fine.",
            "caution": "Short HF dropouts are possible, mainly sunlit side; most NZ circuits OK.",
            "watch": "Heightened risk of HF and GNSS disruption across NZ, esp. midday paths.",
            "severe": "Significant HF and GNSS disruption likely across NZ and the Pacific."
        }[level]
    if kind == "S":
        return {
            "ok": "Radiation environment normal over NZ.",
            "caution": "Elevated radiation — minor impacts; commercial flights OK, polar routes more affected.",
            "watch": "High radiation risk for polar operations; monitor aviation/space assets in our region.",
            "severe": "Severe radiation storm — restrict high-latitude ops; protect space assets."
        }[level]
    return {
        "ok": "Geomagnetic field quiet; GNSS is stable across NZ.",
        "caution": "Field unsettled — small GNSS accuracy dips possible; slim aurora chance in Southland.",
        "watch": "Storm conditions — GNSS accuracy can degrade at times; good aurora odds in the deep south.",
        "severe": "Severe storm — GNSS, HF, and power systems may be impacted; widespread aurora possible."
    }[level]

def _class_to_level(cls_key: str) -> str:
    m = {"ok":"ok","caution":"caution","watch":"watch","severe":"severe"}
    return m.get((cls_key or "").lower(), "ok")

# --- FIND THIS FUNCTION ---
def rewrite_to_nz(section: str, text: str, *,
                  r_now="R0", s_now="S0", g_now="G0",
                  day1=None) -> str:
    tx = (text or "").strip()
    if not tx:
        base = "No significant activity reported."
    else:
        low = tx.lower()
        if section == "solar_activity":
            if _any(low, "x-class", "major flare", "significant flare"):
                base = "Major solar flares noted — higher chance of radio/GNSS issues across New Zealand."
            elif _any(low, "m-class", "moderate"):
                base = "Moderate solar flares observed — brief HF/GNSS hiccups possible over NZ."
            elif _any(low, "c-class", "low", "quiet"):
                base = "The Sun is fairly quiet — only small flares, negligible impact for NZ."
            else:
                base = "Solar activity is mixed but not unusual for the cycle; NZ impacts limited."
        elif section == "solar_wind":
            if _any(low, "cme", "shock", "sheath"):
                base = "A CME is influencing the solar wind — conditions can stir up NZ geomagnetic activity."
            elif _any(low, "high speed", "coronal hole", "600 km/s", "elevated"):
                base = "Solar wind is running fast — may unsettle Earth’s field; aurora possible in the far south."
            else:
                base = "Solar wind conditions are near normal — minimal impact expected over NZ."
        elif section == "geospace":
            if _any(low, "g2", "g3", "storm"):
                base = "Geomagnetic storming occurred — GNSS accuracy could dip; aurora chances improve in Southland."
            elif _any(low, "active", "unsettled"):
                base = "Field was unsettled — small GNSS wobbles possible; low aurora chance."
            else:
                base = "Geomagnetic field is quiet for NZ — comms and GNSS are stable."
        else:
            if _any(low, "elevated", "enhanced", "storm"):
                base = "Energetic particles elevated — low operational impact for NZ; monitor polar routes."
            else:
                base = "Radiation environment looks normal for NZ operations."

    # Choose ONLY the relevant risk line for this section
    if section == "solar_activity":
        r_cls = _r_class(r_now)
        main_risk = _nz_risk_phrase("R", _class_to_level(r_cls))
        return f"{base}{_NZ_REGIONAL_HINT}\n• {main_risk}"
    elif section == "solar_wind":
        s_cls = _s_class(s_now)
        main_risk = _nz_risk_phrase("S", _class_to_level(s_cls))
        return f"{base}{_NZ_REGIONAL_HINT}\n• {main_risk}"
    elif section == "geospace":
        g_cls = _g_class(g_now)
        main_risk = _nz_risk_phrase("G", _class_to_level(g_cls))
        return f"{base}{_NZ_REGIONAL_HINT}\n• {main_risk}"
    else:
        # fallback: just base (for energetic particle etc)
        return f"{base}{_NZ_REGIONAL_HINT}"

# --- END PATCH ---

# ========== Narrative fallback detector ==========
def detect_r_s_watch_flags(structured_disc: dict) -> dict:
    """
    Look through NOAA text for explicit mentions of R4/R5 or S4/S5 and set watch flags.
    This is a coarse, global detector (3-day narrative isn't day-specific).
    """
    parts = []
    for key in ("solar_activity", "energetic_particle", "solar_wind", "geospace"):
        sec = (structured_disc.get(key, {}) or {})
        parts.extend([sec.get("summary", ""), sec.get("forecast", "")])
    parts.append(structured_disc.get("_reflowed", ""))
    blob = " ".join([p for p in parts if p]).upper()

    # Allow forms like "R4", "R5", "R4+", "R4 or greater", "R4-R5"
    r4plus = bool(re.search(r"\bR(?:4|5)(?:\+|\b)", blob)) or bool(re.search(r"R4\s*(?:OR GREATER|\+)", blob))
    s4plus = bool(re.search(r"\bS(?:4|5)(?:\+|\b)", blob)) or bool(re.search(r"S4\s*(?:OR GREATER|\+)", blob))

    return {"r4plus": r4plus, "s4plus": s4plus}

# ========== Data Fetchers ==========

def get_bom_aurora():
    if bom is None:
        return f"Aurora info unavailable. ({'BOM_ERR' in globals() and BOM_ERR or 'pyspaceweather not installed or API key missing'})"
    try:
        outlooks = bom.get_aurora_outlook()
        watches  = bom.get_aurora_watch()
        alerts   = bom.get_aurora_alert()
        lines = []
        if outlooks: lines.append(f"Aurora Outlook: {getattr(outlooks[0],'comments','(no text)')}")
        if watches:  lines.append(f"Aurora Watch: {getattr(watches[0],'comments','(no text)')}")
        if alerts:   lines.append(f"Aurora Alert: {getattr(alerts[0],'description','(no text)')}")
        return "\n".join(lines) if lines else "No aurora alerts/outlooks."
    except Exception as e:
        return f"Aurora info unavailable. ({e})"

def get_noaa_rsg_now_and_past():
    try:
        kp = fetch_json("https://services.swpc.noaa.gov/json/planetary_k_index_1m.json")
        k_now = clamp_float(kp[-1].get("kp_index", 0)) if kp else 0.0
        last = kp[-24:] if kp and len(kp) >= 24 else kp
        k_past = max(clamp_float(v.get("kp_index", 0)) for v in last) if last else k_now
    except Exception:
        k_now = k_past = 0.0

    try:
        xr = fetch_json("https://services.swpc.noaa.gov/json/goes/primary/xray-flares-latest.json")
        x_now = clamp_float(xr[-1].get("flux", 0)) if xr else 0.0
        last = xr[-24:] if xr and len(xr) >= 24 else xr
        x_past = max(clamp_float(v.get("flux", 0)) for v in last) if last else x_now
    except Exception:
        x_now = x_past = 0.0

    try:
        pr = fetch_json("https://services.swpc.noaa.gov/json/goes/primary/integral-protons-1-day.json")
        p_now = clamp_float(pr[-1].get("flux", 0)) if pr else 0.0
        last = pr[-24:] if pr and len(pr) >= 24 else pr
        p_past = max(clamp_float(v.get("flux", 0)) for v in last) if last else p_now
    except Exception:
        p_now = p_past = 0.0

    r_now, r_now_lvl = r_scale(x_now)
    r_past, r_past_lvl = r_scale(x_past)
    s_now, s_now_lvl = s_scale(p_now)
    s_past, s_past_lvl = s_scale(p_past)
    g_now, g_now_lvl = g_scale_from_kp(k_now)
    g_past, g_past_lvl = g_scale_from_kp(k_past)

    current = {"r": r_now, "r_txt": "Radio blackouts", "r_status": "No" if r_now == "R0" else "Active", "lvl": r_now_lvl,
               "s": s_now, "s_txt": "Radiation storms", "s_status": "No" if s_now == "S0" else "Active", "lvl_s": s_now_lvl,
               "g": g_now, "g_txt": "Geomagnetic storms", "g_status": "No" if g_now == "G0" else "Active", "lvl_g": g_now_lvl}
    past = {"r": r_past, "r_txt": "Radio blackouts", "r_status": "No" if r_past == "R0" else "Active", "lvl": r_past_lvl,
            "s": s_past, "s_txt": "Radiation storms", "s_status": "No" if s_past == "S0" else "Active", "lvl_s": s_past_lvl,
            "g": g_past, "g_txt": "Geomagnetic storms", "g_status": "No" if g_past == "G0" else "Active", "lvl_g": g_past_lvl}
    return past, current

# -------- Three-day parsing (Day1/Day2/Day3) --------
def parse_three_day_full(txt: str):
    clean = re.sub(r"[–—]", "-", " ".join(txt.split()))

    def _triplet(pattern):
        m = re.search(pattern, clean, re.I)
        if not m: return [None, None, None]
        a,b,c = m.groups()
        return [int(a), int(b), int(c)]

    r12 = _triplet(r"R1-?R2\s+(\d+)%\s+(\d+)%\s+(\d+)%")
    r3p = _triplet(r"R3\s*(?:\+|or greater)\s+(\d+)%\s+(\d+)%\s+(\d+)%")
    s1p = _triplet(r"S1\s*(?:\+|or greater)\s+(\d+)%\s+(\d+)%\s+(\d+)%")

    kp_trip = [None, None, None]
    table = re.findall(r"\d{2}-\d{2}UT\s+(\d(?:\.\d+)?)\s+(\d(?:\.\d+)?)\s+(\d(?:\.\d+)?)", clean)
    if table:
        colmax = [0.0, 0.0, 0.0]
        for a,b,c in table:
            colmax[0] = max(colmax[0], clamp_float(a))
            colmax[1] = max(colmax[1], clamp_float(b))
            colmax[2] = max(colmax[2], clamp_float(c))
        kp_trip = colmax
    else:
        fb = re.search(r"greatest expected 3 hr Kp .*? is\s+(\d+(?:\.\d+)?)", clean, re.I)
        if fb:
            k = clamp_float(fb.group(1))
            kp_trip = [k, k, k]

    days = []
    for i in range(3):
        kp = kp_trip[i]
        g_bucket = g_scale_from_kp(kp)[0] if kp is not None else "G0"
        days.append({
            "r12": int(r12[i] or 0), "r3": int(r3p[i] or 0), "s1": int(s1p[i] or 0),
            "kp": kp, "g": g_bucket
        })
    return {"days": days}

@st.cache_data(ttl=600, show_spinner=False)
def get_3day_summary():
    try:
        txt = fetch_text("https://services.swpc.noaa.gov/text/3-day-forecast.txt") or ""
        return parse_three_day_full(txt)
    except Exception:
        return {"days": [
            {"r12":0,"r3":0,"s1":0,"kp":None,"g":"G0"},
            {"r12":0,"r3":0,"s1":0,"kp":None,"g":"G0"},
            {"r12":0,"r3":0,"s1":0,"kp":None,"g":"G0"},
        ]}

# ---------- tiny "impact matrix" ----------
def _impact_level(current, day1):
    levels = {}
    g = day1["g"]; gnum = int(g[1]) if len(g) > 1 and g[1].isdigit() else 0
    levels["HF Comms"] = "Watch" if current["r"] != "R0" or day1["r12"] >= 10 else "Nominal"
    levels["Radiation / Polar"] = "Watch" if current["s"] != "S0" or day1["s1"] >= 10 else "Nominal"
    levels["GNSS"] = "Watch" if gnum >= 2 else ("Caution" if gnum >= 1 else "Nominal")
    levels["Power (GIC)"] = "Watch" if gnum >= 2 else ("Caution" if gnum >= 1 else "Nominal")
    levels["LEO Ops/Drag"] = "Caution" if gnum >= 2 else "Nominal"
    return levels

def _impact_tag(label):
    color = {"Nominal":"#1faa72", "Caution":"#e2b200", "Watch":"#cc4c39"}[label]
    return f"<span style='padding:.15rem .45rem;border:1px solid rgba(255,255,255,.15);border-radius:.5rem;background:rgba(255,255,255,.06);color:{color};font-weight:600'>{label}</span>"

def first_sentence(text: str, max_chars: int = 220):
    s = text.strip().split("\n")[0]
    m = re.search(r"(.+?\.)", s)
    s = m.group(1) if m else s
    if len(s) > max_chars:
        s = s[:max_chars-1].rstrip() + "…"
    return s

# ---------- severity class helpers ----------
def _r_class(r: str) -> str:
    """
    New mapping:
      R0 = ok
      R1 = caution
      R2 = watch
      R3 = caution   (changed)
      R4/R5 = watch  (changed)
    Others -> severe (fallback)
    """
    r = (r or "").upper()
    if r.startswith("R0"): return "ok"
    if r.startswith("R1"): return "caution"
    if r.startswith("R2"): return "watch"
    if r.startswith("R3"): return "caution"        # changed per request
    if r.startswith(("R4", "R5")): return "watch"  # 4+ are watch
    return "severe"

def _s_class(s: str) -> str:
    """
    New mapping:
      S0 = ok
      S1 = caution
      S2 = severe (unchanged from your old mapping)
      S3 = caution   (changed)
      S4/S5 = watch  (changed)
    Others -> severe (fallback)
    """
    s = (s or "").upper()
    if s.startswith("S0"): return "ok"
    if s.startswith("S1"): return "caution"
    if s.startswith("S2"): return "severe"
    if s.startswith("S3"): return "caution"        # changed per request
    if s.startswith(("S4", "S5")): return "watch"  # 4+ are watch
    return "severe"

def _g_class(g: str) -> str:
    g = (g or "").upper()
    if g.startswith("G0"): return "ok"
    if g.startswith("G1"): return "caution"
    if g.startswith("G2"): return "watch"
    return "severe"

def r_label_and_class_for_day(day: dict, flags: dict | None = None) -> tuple[str, str]:
    """
    Day-card chip logic:
      - If narrative mentions R4/R5 anywhere -> show R4+ as watch (override).
      - Else if R3+ prob >=1% -> show R3+ as caution.
      - Else if R1–R2 >=10% -> show R1–R2 as caution.
      - Else R0.
    """
    if flags and flags.get("r4plus"):
        return "R4+", "watch"
    if (day.get("r3", 0) or 0) >= 1:
        return "R3+", "caution"
    if (day.get("r12", 0) or 0) >= 10:
        return "R1–R2", "caution"
    return "R0", "ok"

def s_label_and_class_for_day(day: dict, flags: dict | None = None) -> tuple[str, str]:
    """
    Day-card chip logic:
      - If narrative mentions S4/S5 anywhere -> show S4+ as watch (override).
      - Else if S1+ >=10% -> show S1+ as caution.
      - Else S0.
    """
    if flags and flags.get("s4plus"):
        return "S4+", "watch"
    if (day.get("s1", 0) or 0) >= 10:
        return "S1+", "caution"
    return "S0", "ok"

def g_label_and_class_for_day(day: dict) -> tuple[str, str]:
    g = (day.get("g") or "G0").upper()
    return g, _g_class(g)

# -------- Next-24 summary (kept for PDF text) --------
def parse_three_day_for_next24(txt: str):
    clean = re.sub(r"[–—]", "-", " ".join(txt.split()))
    r12 = r3p = s1p = None
    kpmax_day1 = kpmax_day2 = None

    m_r = re.search(r"R1-?R2\s+(\d+)%\s+(\d+)%\s+(\d+)%.*?R3\s*(?:\+|or greater)\s+(\d+)%\s+(\d+)%\s+(\d+)%", clean, re.I)
    if m_r:
        r12, _r12d2, _r12d3, r3p, _r3d2, _r3d3 = map(int, m_r.groups())
    else:
        m_r12 = re.search(r"R1-?R2\s+(\d+)%", clean, re.I)
        m_r3  = re.search(r"R3\s*(?:\+|or greater)\s+(\d+)%", clean, re.I)
        r12 = int(m_r12.group(1)) if m_r12 else 0
        r3p = int(m_r3.group(1)) if m_r3 else 0

    m_s = re.search(r"S1\s*(?:\+|or greater)\s+(\d+)%\s+(\d+)%\s+(\d+)%", clean, re.I)
    if m_s:
        s1p, _s1d2, _s1d3 = map(int, m_s.groups())
    else:
        m_s1 = re.search(r"S1\s*(?:\+|or greater)\s+(\d+)%", clean, re.I)
        s1p = int(m_s1.group(1)) if m_s1 else 0

    triplets = re.findall(r"\d{2}-\d{2}UT\s+(\d(?:\.\d+)?)\s+(\d(?:\.\d+)?)\s+(\d(?:\.\d+)?)", clean)
    if triplets:
        colmax = [0.0, 0.0, 0.0]
        for a, b, c in triplets:
            colmax[0] = max(colmax[0], clamp_float(a))
            colmax[1] = max(colmax[1], clamp_float(b))
            colmax[2] = max(colmax[2], clamp_float(c))
        kpmax_day1, kpmax_day2 = colmax[0], colmax[1]
    else:
        fb = re.search(r"greatest expected 3 hr Kp .*? is\s+(\d+(?:\.\d+)?)", clean, re.I)
        if fb:
            k = clamp_float(fb.group(1))
            kpmax_day1 = kpmax_day2 = k

    r_bucket = "R0"
    if (r12 or 0) >= 10: r_bucket = "R1"
    if (r3p or 0) >= 1:  r_bucket = "R2"

    s_bucket = "S0"
    if (s1p or 0) >= 10: s_bucket = "S1"

    if kpmax_day1 is not None:
        g_bucket, _ = g_scale_from_kp(kpmax_day1)
        kp_str = f"{kpmax_day1:.2f}"
    elif kpmax_day2 is not None:
        g_bucket, _ = g_scale_from_kp(kpmax_day2)
        kp_str = f"{kpmax_day2:.2f}"
    else:
        g_bucket = "G0"; kp_str = "~"

    return {
        "r_bucket": r_bucket, "r12_prob": int(r12 or 0), "r3_prob": int(r3p or 0),
        "s_bucket": s_bucket, "s1_prob": int(s1p or 0),
        "g_bucket": g_bucket, "kp_max": kp_str
    }

@st.cache_data(ttl=600, show_spinner=False)
def get_next24_summary():
    try:
        txt = fetch_text("https://services.swpc.noaa.gov/text/3-day-forecast.txt")
        return parse_three_day_for_next24(txt)
    except Exception:
        return {"r_bucket":"R0","r12_prob":0,"r3_prob":0,"s_bucket":"S0","s1_prob":0,"g_bucket":"G0","kp_max":"~"}

@st.cache_data(ttl=600, show_spinner=False)
def get_noaa_forecast_text():
    """
    Return a structured dict plus the source URL and the FULL raw text (no truncation).
    We try discussion.txt first, then 3-day-forecast.txt as a fallback.
    """
    urls = [
        "https://services.swpc.noaa.gov/text/discussion.txt",
        "https://services.swpc.noaa.gov/text/3-day-forecast.txt",
    ]

    def _try(url):
        raw = fetch_text(url) or ""
        if not raw.strip():
            return None
        # Use the full text; do NOT stop at 'III.' or 'synopsis'
        full = raw.strip()
        try:
            structured = parse_discussion_structured(full)
        except Exception:
            structured = {
                "solar_activity": {"summary": "", "forecast": ""},
                "energetic_particle": {"summary": "", "forecast": ""},
                "solar_wind": {"summary": "", "forecast": ""},
                "geospace": {"summary": "", "forecast": ""},
                "_reflowed": full
            }
        return structured, url, full

    for u in urls:
        got = _try(u)
        if got:
            return got

    # Hard fallback
    return ({
        "solar_activity": {"summary": "", "forecast": ""},
        "energetic_particle": {"summary": "", "forecast": ""},
        "solar_wind": {"summary": "", "forecast": ""},
        "geospace": {"summary": "", "forecast": ""},
        "_reflowed": "NOAA forecast discussion unavailable."
    }, None, "")



def make_summary(current, next24):
    g = next24["g_bucket"]
    kp = next24["kp_max"]
    r12 = next24["r12_prob"]; r3 = next24["r3_prob"]; s1 = next24["s1_prob"]
    return (
        f"Now (R/S/G): {current['r']}/{current['s']}/{current['g']}. "
        f"Next 24 h: {g} (Kp≃{kp}); R1–R2 {r12}% | R3+ {r3}% | S1+ {s1}%. "
        f"Implications: {'HF comms at risk; watch D-layer absorption' if current['r']!='R0' or r12>=10 else 'Nominal HF'}, "
        f"{'SEPs possible; elevate EVA/aviation polar routes' if current['s']!='S0' or s1>=10 else 'Nominal radiation'}, "
        f"{'Geomagnetic impacts possible; GIC risk on high-lat power & GNSS scintillation' if g!='G0' else 'Nominal geomagnetic.'}"
    )

# Small chart helpers for PDF
def create_xray_chart():
    data = fetch_json("https://services.swpc.noaa.gov/json/goes/primary/xrays-6-hour.json")
    if not data: return None
    times = [row.get("time_tag") for row in data if "time_tag" in row]
    fluxes = [row.get("flux", 0) for row in data]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=fluxes, mode="lines", name="X-ray Flux"))
    fig.update_layout(title="X-rays (6-hour)", height=220,
                      margin=dict(l=10, r=10, t=30, b=10),
                      xaxis=dict(title="Time", color="#9fc8ff"),
                      yaxis=dict(title="Flux", color="#9fc8ff"))
    return fig

def create_proton_chart():
    data = fetch_json("https://services.swpc.noaa.gov/json/goes/primary/integral-protons-1-day.json")
    if not data: return None
    times = [row.get("time_tag") for row in data if "time_tag" in row]
    fluxes = [row.get("flux", 0) for row in data]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=fluxes, mode="lines", name="Proton Flux"))
    fig.update_layout(title="Integral Protons (1-day)", height=220,
                      margin=dict(l=10, r=10, t=30, b=10),
                      xaxis=dict(title="Time", color="#9fc8ff"),
                      yaxis=dict(title="Flux", color="#9fc8ff"))
    return fig

def create_kp_chart():
    kp = fetch_json("https://services.swpc.noaa.gov/json/planetary_k_index_1m.json")
    if not kp: return None
    times = [row.get("time_tag") for row in kp if "time_tag" in row]
    vals = [clamp_float(row.get("kp_index", 0)) for row in kp]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=vals, mode="lines", name="Kp Index"))
    fig.update_layout(title="Kp Index (1-minute)", height=220,
                      margin=dict(l=10, r=10, t=30, b=10),
                      xaxis=dict(title="Time", color="#9fc8ff"),
                      yaxis=dict(title="Kp", color="#9fc8ff"))
    return fig
# ========== UI Styles ==========
st.markdown(f"""<style>
:root {{
  --scale: {font_scale};
  --neon: {"#00ffff" if high_contrast else "#8be9fd"};
  --fg: {"#ffffff" if high_contrast else "#dbe7ff"};
  --bg: {"#0a0a0a" if high_contrast else "#0d1419"};
  --card: {"#0f0f0f" if high_contrast else "#111a21"};
  --border: {"#ffffff90" if high_contrast else "rgba(139,233,253,.25)"};
}}
html, body, .main, .block-container {{ font-size: calc(16px * var(--scale)); }}
.badge-col {{ display:flex; gap:.5rem; flex-wrap:wrap; margin:.5rem 0 1rem 0; }}
.neon-badge {{ padding:.35rem .6rem; border-radius:.5rem; border:1px solid var(--border); color:var(--fg); }}
.grid-bom {{ display:grid; gap:.75rem; grid-template-columns: repeat(5, 1fr); }}
.box {{ background:var(--card); border:1px solid var(--border); border-radius:.6rem; padding:.75rem; }}
.box h5 {{ margin:.2rem 0 .6rem 0; color:#cfe3ff; font-size:1rem; }}
.rs-inline {{ display:flex; gap:.35rem; flex-wrap:wrap; }}
.rs-pill {{ padding:.18rem .45rem; border-radius:.45rem; color:#0a0a0a; font-weight:700; }}
.subline {{ margin-top:.35rem; color:#cfe3ff; opacity:.85; font-size:.9rem; }}
.impact {{ margin-top:.6rem; display:grid; grid-template-columns: 1.2fr 1fr 1fr 1fr 1fr; gap:.3rem .6rem; }}
.impact div {{ padding:.35rem .5rem; border-bottom:1px dashed rgba(139,233,253,.18); }}
.impact .hdr {{ color:#9fc8ff; font-weight:700; border-bottom:1px solid rgba(139,233,253,.25); }}
.tri-grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:.75rem; margin-top:.75rem; }}
.tri-card {{ background:var(--card); border:1px solid var(--border); border-radius:.6rem; padding:.8rem; }}
.tri-card h5 {{ margin:.1rem 0 .4rem 0; color:#cfe3ff; font-size:1rem; }}
.tri-card .body {{ color:#cfe3ff; opacity:.95; line-height:1.35; }}
.ok     {{ background:#2ecc71; color:#0b0b0b; }}
.caution{{ background:#f1c40f; color:#0b0b0b; }}
.watch  {{ background:#e67e22; color:#0b0b0b; }}
.severe {{ background:#e74c3c; color:#0b0b0b; }}
</style>""", unsafe_allow_html=True)

# ========== Tabs ==========
tab_overview, tab_charts, tab_forecast, tab_aurora, tab_expert, tab_pdf, tab_help = st.tabs([
    "Overview", "Charts", "Forecasts", "Aurora", "Expert Data", "PDF Export", "Help & Info"
])

# ========== Overview Tab ==========
# ========== Overview Tab ==========
with tab_overview:
    st.markdown("""
<style>
  /* Match Forecast tab pill style exactly */
  .rs-inline { display:flex; gap:1.8rem; align-items:center; flex-wrap:wrap; }

  .rs-pill{
    display:inline-flex;
    align-items:center; justify-content:center;
    padding:.18rem .48rem;                 /* compact like forecast */
    border-radius:8px;                     /* small rounded corners */
    font-weight:800; font-size:.95rem;     /* same weight/size as forecast look */
    line-height:1.1;
    border:2px solid transparent;
    background:transparent;
    letter-spacing:.2px;
  }

  /* Tone colours (same mapping used in Forecast tab visuals) */
  .rs-pill.ok{
    color:#22c55e;                         /* green */
    border-color:#22c55e;
    background:rgba(34,197,94,.12);        /* faint fill */
  }
  .rs-pill.caution{
    color:#f59e0b;                         /* amber */
    border-color:#f59e0b;
    background:rgba(245,158,11,.12);
  }
  .rs-pill.watch{
    color:#ef4444;                         /* red */
    border-color:#ef4444;
    background:rgba(239,68,68,.12);
  }
</style>
""", unsafe_allow_html=True)




    # --- data pulls ---
    past, current = get_noaa_rsg_now_and_past()
    three = get_3day_summary()
    day1, day2, day3 = three["days"]
    structured_disc, noaa_discussion_src, _raw = get_noaa_forecast_text()
    src_note = noaa_discussion_src.split('/')[-1] if noaa_discussion_src else 'unavailable'

    # ----------------- 1) IMPACT (Next 24 h) — unchanged layout -----------------
    levels = _impact_level(current, day1)
    st.markdown("### Impact (Next 24 h)")
    im = ["<div class='impact'>",
          "<div class='hdr'>Domain</div>",
          "<div class='hdr'>HF Comms</div>",
          "<div class='hdr'>GNSS</div>",
          "<div class='hdr'>Power (GIC)</div>",
          "<div class='hdr'>Radiation / Polar</div>",
          "<div><strong>Status</strong></div>",
          f"<div>{_impact_tag(levels['HF Comms'])}</div>",
          f"<div>{_impact_tag(levels['GNSS'])}</div>",
          f"<div>{_impact_tag(levels['Power (GIC)'])}</div>",
          f"<div>{_impact_tag(levels['Radiation / Polar'])}</div>",
          "</div>"]
    st.markdown("".join(im), unsafe_allow_html=True)
    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # ----------------- Tone mapping to match NOAA/Forecast -----------------
    def _tone_from_label(label_text: str) -> str:
        if not label_text:
            return "ok"
        m = re.search(r"(\d+)", label_text)
        n = int(m.group(1)) if m else 0
        if n >= 4: return "watch"
        if n == 3: return "caution"
        return "ok"

    def _pill(label_text: str) -> str:
        return f"<span class='rs-pill {_tone_from_label(label_text)}'>{label_text}</span>"


    # ----------------- 2) Past/Current + Day 1–3 cards -----------------
    def col_block(title, r_tuple, s_tuple, g_tuple, subline_html=""):
        # Only use labels; tones come from labels to ensure consistency
        r_lbl, _ = r_tuple
        s_lbl, _ = s_tuple
        g_lbl, _ = g_tuple
        return f"""
        <div class="box">
          <h5>{title}</h5>
          <div class="rs-inline">
            {_pill(r_lbl)}
            {_pill(s_lbl)}
            {_pill(g_lbl)}
          </div>
          <div class="subline">{subline_html}</div>
        </div>"""

    past_html = col_block(
        "Past 24 hours",
        (past['r'], _tone_from_label(past['r'])),
        (past['s'], _tone_from_label(past['s'])),
        (past['g'], _tone_from_label(past['g'])),
        "Radio blackouts · Radiation storms · Geomagnetic storms"
    )
    curr_html = col_block(
        "Current",
        (current['r'], _tone_from_label(current['r'])),
        (current['s'], _tone_from_label(current['s'])),
        (current['g'], _tone_from_label(current['g'])),
        "Radio blackouts · Radiation storms · Geomagnetic storms"
    )

    r1, r1_cls = r_label_and_class_for_day(day1)
    s1, s1_cls = s_label_and_class_for_day(day1)
    g1, g1_cls = g_label_and_class_for_day(day1)
    d1_html = col_block(
        "Day 1 forecast",
        (r1, r1_cls), (s1, s1_cls), (g1, g1_cls),
        f"R1–R2: {day1['r12']}% · R3+: {day1['r3']}% · S1+: {day1['s1']}% · Kp≈{day1['kp'] if day1['kp'] is not None else '~'}"
    )

    r2, r2_cls = r_label_and_class_for_day(day2)
    s2, s2_cls = s_label_and_class_for_day(day2)
    g2, g2_cls = g_label_and_class_for_day(day2)
    d2_html = col_block(
        "Day 2 forecast",
        (r2, r2_cls), (s2, s2_cls), (g2, g2_cls),
        f"R1–R2: {day2['r12']}% · R3+: {day2['r3']}% · S1+: {day2['s1']}% · Kp≈{day2['kp'] if day2['kp'] is not None else '~'}"
    )

    # Fix: ensure Day 3 uses the correct helpers for S/G (not r_label again)
    r3, r3_cls = r_label_and_class_for_day(day3)
    s3, s3_cls = s_label_and_class_for_day(day3)
    g3, g3_cls = g_label_and_class_for_day(day3)
    d3_html = col_block(
        "Day 3 forecast",
        (r3, r3_cls), (s3, s3_cls), (g3, g3_cls),
        f"R1–R2: {day3['r12']}% · R3+: {day3['r3']}% · S1+: {day3['s1']}% · Kp≈{day3['kp'] if day3['kp'] is not None else '~'}"
    )

    st.markdown(
        f"<div class='grid-bom'>{past_html}{curr_html}{d1_html}{d2_html}{d3_html}</div>",
        unsafe_allow_html=True
    )
    st.caption(f"Last updated: {last_updated()} · Source: {src_note}")

    # ----------------- helpers to pull NOAA 24h text robustly -----------------
    def _clean_noaa_lines(txt: str) -> str:
        if not txt:
            return ""
        lines = [ln.strip() for ln in txt.replace("\r", "").split("\n")]
        keep = [ln for ln in lines if ln and not ln.startswith(':') and "Prepared by" not in ln]
        out = " ".join(keep).strip()
        return re.sub(r"\s{2,}", " ", out)

    def _pick_source_text() -> str:
        a = fetch_text("https://services.swpc.noaa.gov/text/discussion.txt") or ""
        if len(a.strip()) > 800:
            return a
        b = fetch_text("https://services.swpc.noaa.gov/text/3-day-forecast.txt") or ""
        return b or a

    def _grab_between(text: str, start_pat: str, end_pat_list: list[str]) -> str:
        m = re.search(start_pat, text, re.I | re.S)
        if not m:
            return ""
        start = m.end()
        end = len(text)
        for p in end_pat_list:
            m2 = re.search(p, text[start:], re.I | re.S)
            if m2:
                end = start + m2.start()
                break
        return text[start:end].strip()

    @st.cache_data(ttl=600, show_spinner=False)
    def get_noaa_24h_summaries_direct() -> dict:
        blob = _pick_source_text().replace("\r", "")
        sa_start = r"(?:^|\n)\s*(?:I\.|\d\.)?\s*Solar\s*Activity.*?(?:24\s*hr|24hr|24\s*hour)?\s*summary\b.*?"
        sw_start = r"(?:^|\n)\s*(?:II\.|\d\.)?\s*Solar\s*Wind.*?(?:24\s*hr|24hr|24\s*hour)?\s*summary\b.*?"
        gs_start = r"(?:^|\n)\s*(?:III\.|\d\.)?\s*(?:Geospace|Geo[\s-]*Space).*?(?:24\s*hr|24hr|24\s*hour)?\s*summary\b.*?"

        ends = [
            r"(?:^|\n)\s*\.?\s*(?:forecast|forcast)\b",
            r"(?:^|\n)\s*(?:II\.|\d\.)\s*Solar\s*Wind\b",
            r"(?:^|\n)\s*(?:III\.|\d\.)\s*(?:Geospace|Geo[\s-]*Space)\b",
            r"(?:^|\n)\s*(?:I\.|\d\.)\s*Solar\s*Activity\b",
        ]

        sa = _clean_noaa_lines(_grab_between(blob, sa_start, ends))
        sw = _clean_noaa_lines(_grab_between(blob, sw_start, ends))
        gs = _clean_noaa_lines(_grab_between(blob, gs_start, ends))

        if not sa:
            sa = _clean_noaa_lines(_grab_between(blob, r"(?:^|\n)\s*(?:I\.|\d\.)?\s*Solar\s*Activity\b.*?", ends))
        if not sw:
            sw = _clean_noaa_lines(_grab_between(blob, r"(?:^|\n)\s*(?:II\.|\d\.)?\s*Solar\s*Wind\b.*?", ends))
        if not gs:
            gs = _clean_noaa_lines(_grab_between(blob, r"(?:^|\n)\s*(?:III\.|\d\.)?\s*(?:Geospace|Geo[\s-]*Space)\b.*?", ends))

        return {"solar_activity": sa or "—", "solar_wind": sw or "—", "geospace": gs or "—"}

    # ----------------- 3) NZ plain-English summaries -----------------
    direct_24h = get_noaa_24h_summaries_direct()
    sa_full = direct_24h["solar_activity"]
    sw_full = direct_24h["solar_wind"]
    gs_full = direct_24h["geospace"]

    st.markdown("### New Zealand Plain-English Summaries")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### Solar Activity (NZ)")
        st.info(rewrite_to_nz("solar_activity", sa_full,
                              r_now=current['r'], s_now=current['s'], g_now=current['g'], day1=day1))
    with c2:
        st.markdown("#### Solar Wind (NZ)")
        st.info(rewrite_to_nz("solar_wind", sw_full,
                              r_now=current['r'], s_now=current['s'], g_now=current['g'], day1=day1))
    with c3:
        st.markdown("#### Geospace (NZ)")
        st.info(rewrite_to_nz("geospace", gs_full,
                              r_now=current['r'], s_now=current['s'], g_now=current['g'], day1=day1))

    # ----------------- 4) NOAA 24-hour Summaries (Raw text) -----------------
    st.markdown("### NOAA 24-hour Summaries (Raw text)")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Solar Activity — 24 hr Summary**")
        st.markdown(sa_full.replace("\n", "<br>"), unsafe_allow_html=True)
    with c2:
        st.markdown("**Solar Wind — 24 hr Summary**")
        st.markdown(sw_full.replace("\n", "<br>"), unsafe_allow_html=True)
    with c3:
        st.markdown("**Geospace — 24 hr Summary**")
        st.markdown(gs_full.replace("\n", "<br>"), unsafe_allow_html=True)


# ========== Charts Tab ==========
with tab_charts:
    st.markdown("## Space Weather Analytics (NOAA Style)")

    time_ranges = {
        "6 Hour": 72,
        "1 Day": 288,
        "3 Day": 864,
        "7 Day": 2016
    }
    st.markdown("### Select Time Range for All Charts")
    selected_range = st.radio("Time Range", list(time_ranges.keys()), index=3, horizontal=True)  # Default: "7 Day"
    count = time_ranges[selected_range]

    def plot_kp_index():
        url = "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json"
        data = fetch_json(url)
        if not data or not isinstance(data, list):
            st.caption("No Kp index data available.")
            return
        times = [row.get("time_tag") for row in data]
        kp_values = [row.get("kp_index", 0) for row in data]
        times = times[-count:]
        kp_values = kp_values[-count:]
        colors = []
        for kp in kp_values:
            if kp < 5: colors.append("limegreen")
            elif kp == 5: colors.append("gold")
            elif kp == 6: colors.append("orange")
            elif kp >= 7: colors.append("red")
            else: colors.append("gray")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=times, y=kp_values, marker_color=colors, name="Kp Index"))
        fig.update_layout(title="Estimated Planetary K Index", height=320,
                          xaxis_title="Time", yaxis_title="Kp index",
                          showlegend=False)
        fig.add_annotation(
            xref="paper", yref="paper", x=0, y=1.12, text="Kp < 5: G0 (green) • Kp = 5: G1 (yellow) • Kp = 6: G2 (orange) • Kp ≥ 7: G3+ (red)",
            showarrow=False, font=dict(color="white", size=13), bgcolor="rgba(0,0,0,0.2)"
        )
        st.plotly_chart(fig, use_container_width=True, config={
            "displayModeBar": True,
            "toImageButtonOptions": {"format":"png","filename":"Kp_Index","height":400,"width":1200,"scale":2}
        })

    def plot_xray_flux():
        url = "https://services.swpc.noaa.gov/json/goes/primary/xrays-3-day.json"
        data = fetch_json(url)
        if not data or not isinstance(data, list):
            st.caption("No X-ray data available.")
            return
        times = [row.get("time_tag") for row in data]
        b_flux = [row.get("flux_b", row.get("flux", 0)) for row in data]
        m_flux = [row.get("flux_m", 0) for row in data]
        x_flux = [row.get("flux_x", 0) for row in data]
        times = times[-count:]
        b_flux = b_flux[-count:]
        m_flux = m_flux[-count:]
        x_flux = x_flux[-count:]
        smooth = st.checkbox("Smooth (1-hour MA)", value=False, key="xray_smooth")
        if smooth and len(b_flux) > 12:
            b_flux = uniform_filter1d(b_flux, size=12)
            m_flux = uniform_filter1d(m_flux, size=12)
            x_flux = uniform_filter1d(x_flux, size=12)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=times, y=b_flux, mode="lines", name="B-class", line=dict(color="blue")))
        fig.add_trace(go.Scatter(x=times, y=m_flux, mode="lines", name="M-class", line=dict(color="orange")))
        fig.add_trace(go.Scatter(x=times, y=x_flux, mode="lines", name="X-class", line=dict(color="red")))
        fig.update_layout(title="GOES X-ray Flux", height=300, xaxis_title="Time", yaxis_title="Flux (W/m²)")
        st.plotly_chart(fig, use_container_width=True, config={
            "displayModeBar": True,
            "toImageButtonOptions": {"format":"png","filename":"GOES_Xray_Flux","height":400,"width":1200,"scale":2}
        })

    def plot_proton_flux():
        url = "https://services.swpc.noaa.gov/json/goes/primary/integral-protons-3-day.json"
        data = fetch_json(url)
        if not data or not isinstance(data, list):
            st.caption("No Proton data available.")
            return
        times = [row.get("time_tag") for row in data]
        flux_1mev = [row.get("flux_1mev", row.get("flux", 0)) for row in data]
        flux_10mev = [row.get("flux_10mev", 0) for row in data]
        times = times[-count:]
        flux_1mev = flux_1mev[-count:]
        flux_10mev = flux_10mev[-count:]
        smooth = st.checkbox("Smooth (1-hour MA)", value=False, key="proton_smooth")
        if smooth and len(flux_1mev) > 12:
            flux_1mev = uniform_filter1d(flux_1mev, size=12)
            flux_10mev = uniform_filter1d(flux_10mev, size=12)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=times, y=flux_1mev, mode="lines", name=">1 MeV", line=dict(color="orange")))
        fig.add_trace(go.Scatter(x=times, y=flux_10mev, mode="lines", name=">10 MeV", line=dict(color="red")))
        fig.update_layout(title="GOES Proton Flux", height=300, xaxis_title="Time", yaxis_title="Flux (pfu)")
        st.plotly_chart(fig, use_container_width=True, config={
            "displayModeBar": True,
            "toImageButtonOptions": {"format":"png","filename":"GOES_Proton_Flux","height":400,"width":1200,"scale":2}
        })

    def plot_magnetometer():
        url = "https://services.swpc.noaa.gov/json/goes/primary/magnetometers-3-day.json"
        data = fetch_json(url)
        if not data or not isinstance(data, list):
            st.caption("No Magnetometer data available.")
            return
        times = [row.get("time_tag") for row in data]
        bx = [row.get("bx_gsm", 0) for row in data]
        by = [row.get("by_gsm", 0) for row in data]
        bz = [row.get("bz_gsm", 0) for row in data]
        times = times[-count:]
        bx = bx[-count:]
        by = by[-count:]
        bz = bz[-count:]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=times, y=bx, mode="lines", name="Bx GSM", line=dict(color="cornflowerblue")))
        fig.add_trace(go.Scatter(x=times, y=by, mode="lines", name="By GSM", line=dict(color="tomato")))
        fig.add_trace(go.Scatter(x=times, y=bz, mode="lines", name="Bz GSM", line=dict(color="limegreen")))
        fig.update_layout(title="GOES Magnetometers (GSM)", height=300, xaxis_title="Time", yaxis_title="nT")
        st.plotly_chart(fig, use_container_width=True, config={
            "displayModeBar": True,
            "toImageButtonOptions": {"format":"png","filename":"GOES_Magnetometer","height":400,"width":1200,"scale":2}
        })

    def plot_suvi_flares():
        url = "https://services.swpc.noaa.gov/json/goes/primary/suvi-flares-7-day.json"
        data = fetch_json(url)
        if not data or not isinstance(data, list):
            st.caption("No SUVI flares data available.")
            return
        times = [row.get("begin_time") for row in data if "begin_time" in row]
        intensities = [row.get("peak_intensity", 0) for row in data]
        times = times[-count:]
        intensities = intensities[-count:]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=times, y=intensities, name="SUVI Flare Intensity", marker_color="gold"))
        fig.update_layout(title="GOES SUVI Flares", height=300, xaxis_title="Time", yaxis_title="Peak Intensity")
        st.plotly_chart(fig, use_container_width=True, config={
            "displayModeBar": True,
            "toImageButtonOptions": {"format":"png","filename":"GOES_SUVI_Flares","height":400,"width":1200,"scale":2}
        })

    st.markdown("### Estimated Planetary K Index")
    plot_kp_index()
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### GOES X-ray Flux")
        plot_xray_flux()
        st.markdown("### GOES Magnetometers")
        plot_magnetometer()
    with col2:
        st.markdown("### GOES Proton Flux")
        plot_proton_flux()
        st.markdown("### GOES SUVI Flares")
        plot_suvi_flares()
    st.caption(f"Last updated: {last_updated()}")


# ========== Forecasts Tab ==========
with tab_forecast:
    # ---- Scoped styles to match your standard boxes/pills ----
    st.markdown("""
    <style>
      .stdgrid3 { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:1rem; }
      .stdgrid2 { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:1rem; }
      @media (max-width: 1000px) {
        .stdgrid3 { grid-template-columns:1fr; }
        .stdgrid2 { grid-template-columns:1fr; }
      }

      /* Standard box style from your screenshot */
      .stdbox {
        border:1px solid rgba(255,255,255,.18);
        border-radius:12px;
        padding:12px 14px;
        background:rgba(255,255,255,.03);
      }
      .stdhdr {
        font-weight:700; font-size:.95rem; color:#93c5fd; /* blue header */
        margin-bottom:.35rem;
      }
      .stdsub { opacity:.8; font-size:.9rem; margin-top:-.15rem; }
      .subtle { opacity:.75; font-size:.9rem; }

      /* Chips row + chips */
      .chips-row { display:flex; gap:.5rem; flex-wrap:wrap; margin-top:.25rem; }
      .chip {
        display:inline-block;
        padding:.12rem .55rem;
        border-radius:999px;
        font-weight:700;
        font-size:.85rem;
        line-height:1.2;
        border:2px solid;     /* outlined look like your screenshot */
        background:transparent;
      }

      /* Tones (visual mapping) */
      .chip.ok      { border-color:#22c55e33; color:#a7f3d0; }  /* green-ish */
      .chip.caution { border-color:#f59e0b33; color:#fde68a; }  /* amber */
      .chip.watch   { border-color:#ef444433; color:#fecaca; }  /* red */

      /* Little dot for Kp (kept if you want to show it later) */
      .kp-dot {
        display:inline-block; width:.6rem; height:.6rem; border-radius:50%;
        margin-right:.35rem; vertical-align:middle; background:rgba(255,255,255,.35);
      }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("## Three-Day Forecast")

    # --- Data pulls ---
    structured_disc, noaa_discussion_src, _raw = get_noaa_forecast_text()
    narrative_flags = detect_r_s_watch_flags(structured_disc)
    src_note = noaa_discussion_src.split('/')[-1] if noaa_discussion_src else 'unavailable'
    three = get_3day_summary()
    day1, day2, day3 = three["days"]
    past, current = get_noaa_rsg_now_and_past()

    # ---------- NOAA forecast text helpers ----------
    def _clean_noaa_text(txt: str) -> str:
        if not txt:
            return ""
        lines = [ln.strip() for ln in txt.replace("\r", "").split("\n")]
        keep = [ln for ln in lines if ln and not ln.startswith(":") and "Prepared by" not in ln]
        return re.sub(r"\s{2,}", " ", " ".join(keep)).strip()

    def _fallback_forecast_from_reflow(structured: dict, sec_label_regex: str) -> str:
        """
        If the 'forecast' field is empty, carve a forecast paragraph out of
        the raw '_reflowed' blob after a 'Forecast' marker.
        """
        blob = (structured.get("_reflowed") or "").replace("\r", "")
        if not blob:
            return ""
        start_pat = rf"(?:^|\n).*?(?:{sec_label_regex}).*?(?:Forecast|Forcast)\b.*?"
        end_pats = [
            r"(?:^|\n)\s*(?:Summary|24\s*hr|24hr)\b",
            r"(?:^|\n)\s*Solar\s*Activity\b",
            r"(?:^|\n)\s*Energetic\s*Particle\b",
            r"(?:^|\n)\s*Solar\s*Wind\b",
            r"(?:^|\n)\s*(?:Geo[\s-]*Space|Geospace)\b",
        ]
        m = re.search(start_pat, blob, re.I | re.S)
        if not m:
            return ""
        start = m.end()
        end = len(blob)
        for pat in end_pats:
            m2 = re.search(pat, blob[start:], re.I | re.S)
            if m2:
                end = start + m2.start()
                break
        carved = blob[start:end].strip()
        if carved:
            sent = re.findall(r"(.+?\.)", carved.replace("\n", " "))
            carved = " ".join(sent[:5]).strip() if sent else carved
        return _clean_noaa_text(carved)

    def _forecast_for(structured: dict, key: str, sec_label_regex: str) -> str:
        fc = _clean_noaa_text((structured.get(key, {}) or {}).get("forecast", ""))
        if fc:
            return fc
        return _fallback_forecast_from_reflow(structured, sec_label_regex) or "—"

    # ---------- Forecast text pulls ----------
    fc_sa = _forecast_for(structured_disc, "solar_activity", r"Solar\s*Activity")
    fc_ep = _forecast_for(structured_disc, "energetic_particle", r"Eng?ergetic\s*Particle")
    fc_sw = _forecast_for(structured_disc, "solar_wind", r"Solar\s*Wind")
    fc_gs = _forecast_for(structured_disc, "geospace", r"(?:Geo[\s-]*Space|Geospace)")

    # ---------- NZ narrative at the top (standard boxes) ----------
    st.markdown("### New Zealand Plain-English Summaries")

    nz_fc_sa = rewrite_to_nz("solar_activity", fc_sa, r_now=current['r'], s_now=current['s'], g_now=current['g'], day1=day1)
    nz_fc_sw = rewrite_to_nz("solar_wind",     fc_sw, r_now=current['r'], s_now=current['s'], g_now=current['g'], day1=day1)
    nz_fc_gs = rewrite_to_nz("geospace",       fc_gs, r_now=current['r'], s_now=current['s'], g_now=current['g'], day1=day1)

    st.markdown(
        f"""
        <div class="stdgrid3">
          <div class="stdbox">
            <div class="stdhdr">Solar Activity (NZ)</div>
            <div class="stdsub">Sunspots & flare risk for operations</div>
            <div style="margin-top:.45rem;">{nz_fc_sa}</div>
          </div>
          <div class="stdbox">
            <div class="stdhdr">Solar Wind (NZ)</div>
            <div class="stdsub">IMF, speed & CME impacts for comms</div>
            <div style="margin-top:.45rem;">{nz_fc_sw}</div>
          </div>
          <div class="stdbox">
            <div class="stdhdr">Geospace (NZ)</div>
            <div class="stdsub">Kp/aurora & GNSS stability across NZ</div>
            <div style="margin-top:.45rem;">{nz_fc_gs}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.caption(f"Last updated: {last_updated()} · Source: {src_note}")

    # ---------- Severity pill tone mapping ----------
    def _tone_from_label(label_text: str) -> str:
        """
        Returns: 'ok' | 'caution' | 'watch'
        Rule: level 3 => caution; level 4+ => watch; else ok
        """
        if not label_text:
            return "ok"
        m = re.search(r"(\d+)", label_text)
        n = int(m.group(1)) if m else 0
        if n >= 4:
            return "watch"
        if n == 3:
            return "caution"
        return "ok"

    # ---------- Quick matrix: Day 1–3 (standard boxes + pills) ----------
    def _kp_str(v): return "~" if v is None else f"{v:.1f}"

    def _badge_row(d):
        r_lbl, _ = r_label_and_class_for_day(d, narrative_flags)
        s_lbl, _ = s_label_and_class_for_day(d, narrative_flags)
        g_lbl, _ = g_label_and_class_for_day(d)
        r_tone = _tone_from_label(r_lbl)
        s_tone = _tone_from_label(s_lbl)
        g_tone = _tone_from_label(g_lbl)
        return (
            f"<div class='chips-row'>"
            f"<span class='chip {g_tone}'>G: {g_lbl} · Kp≈{_kp_str(d['kp'])}</span>"
            f"<span class='chip {s_tone}'>S: {s_lbl} · S1+: {d['s1']}%</span>"
            f"<span class='chip {r_tone}'>R: {r_lbl} · R1–R2: {d['r12']}% · R3+: {d['r3']}%</span>"
            f"</div>"
        )

    st.markdown("<div style='height:.75rem;'></div>", unsafe_allow_html=True)
    st.markdown("### Next 72 Hours — At a Glance")
    st.markdown(
        f"""
        <div class="stdgrid3">
          <div class="stdbox">
            <div class="stdhdr">Day 1 forecast</div>
            {_badge_row(day1)}
          </div>
          <div class="stdbox">
            <div class="stdhdr">Day 2 forecast</div>
            {_badge_row(day2)}
          </div>
          <div class="stdbox">
            <div class="stdhdr">Day 3 forecast</div>
            {_badge_row(day3)}
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ---------- Status: Past 24 hours & Current (slot above NOAA narrative) ----------
    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    st.markdown("### Status — Past 24 hours & Now")

    def _now_chips(d: dict) -> str:
        r_lbl = d.get("r", "")
        s_lbl = d.get("s", "")
        g_lbl = d.get("g", "")
        kp    = d.get("kp", None)
        r_tone = _tone_from_label(r_lbl)
        s_tone = _tone_from_label(s_lbl)
        g_tone = _tone_from_label(g_lbl)
        kp_str = f" · Kp≈{kp:.1f}" if isinstance(kp, (int, float)) else ""
        return (
            "<div class='chips-row'>"
            f"<span class='chip {g_tone}'>G: {g_lbl}{kp_str}</span>"
            f"<span class='chip {s_tone}'>S: {s_lbl}</span>"
            f"<span class='chip {r_tone}'>R: {r_lbl}</span>"
            "</div>"
        )

    st.markdown(
        f"""
        <div class="stdgrid2">
          <div class="stdbox">
            <div class="stdhdr">Past 24 hours</div>
            {_now_chips(past)}
            <div class="subtle" style="margin-top:.4rem;">Radio blackouts · Radiation storms · Geomagnetic storms</div>
          </div>
          <div class="stdbox">
            <div class="stdhdr">Current</div>
            {_now_chips(current)}
            <div class="subtle" style="margin-top:.4rem;">Radio blackouts · Radiation storms · Geomagnetic storms</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.caption(f"Last updated: {last_updated()} · Source: {src_note}")

    # ---------- NOAA detailed narrative (boxed) ----------
    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    st.markdown("### NOAA Detailed Narrative")

    for title, fc_txt in [
        ("Solar Activity", fc_sa),
        ("Energetic Particle", fc_ep),
        ("Solar Wind", fc_sw),
        ("Geospace", fc_gs),
    ]:
        if fc_txt and fc_txt != "—":
            with st.expander(title, expanded=False):
                st.markdown(
                    f"<div class='stdbox'><div class='stdhdr'>{title}</div>{fc_txt.replace('\n','<br>')}</div>",
                    unsafe_allow_html=True
                )

    st.caption(f"Last updated: {last_updated()} · Source: {src_note}")








# ========== Aurora Tab ==========
with tab_aurora:
    # Only show BOM Aurora content (no NOAA forecast)
    bom_aurora_text = get_bom_aurora()

    st.markdown("## Aurora (BOM)")

    st.markdown("""
    <div class='section' role='region' aria-label='BOM Aurora'>
      <div class='subttl'>BOM Aurora</div>
      <div style='white-space:pre-line;color:#cfe3ff;margin-top:.5rem;'>%s</div>
    </div>
    """ % (bom_aurora_text or "No aurora alerts/outlooks."),
    unsafe_allow_html=True)
    
    st.caption(f"Last updated: {last_updated()} · Source: BOM Aurora / SWS")



# ========== Expert Data Tab (Charts) ==========
with tab_expert:
    st.markdown("## GOES / L1 Expert Data (Expanded, Charts)")
    col1, col2 = st.columns(2)

    # ---------- Column 1 ----------
    with col1:
        st.markdown("### Differential Electrons (3-day)")
        elec_data = fetch_json("https://services.swpc.noaa.gov/json/goes/primary/differential-electrons-3-day.json")
        if elec_data and isinstance(elec_data, list):
            times = [row.get("time_tag") for row in elec_data if "time_tag" in row]
            fluxes = [row.get("flux", 0) for row in elec_data]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=times, y=fluxes, mode="lines", name="Electron Flux"))
            fig.update_layout(title="Differential Electron Flux", height=220,
                             margin=dict(l=10, r=10, t=30, b=10),
                             xaxis=dict(title="Time", color="#9fc8ff"),
                             yaxis=dict(title="Flux", color="#9fc8ff"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No electron chart data available.")

        st.markdown("### Integral Protons (1-day)")
        ip_data = fetch_json("https://services.swpc.noaa.gov/json/goes/primary/integral-protons-1-day.json")
        if ip_data and isinstance(ip_data, list):
            times = [row.get("time_tag") for row in ip_data if "time_tag" in row]
            fluxes = [row.get("flux", 0) for row in ip_data]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=times, y=fluxes, mode="lines", name="Proton Flux"))
            fig.update_layout(title="Integral Proton Flux (1-day)", height=220,
                             margin=dict(l=10, r=10, t=30, b=10),
                             xaxis=dict(title="Time", color="#9fc8ff"),
                             yaxis=dict(title="Flux", color="#9fc8ff"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No integral protons chart data available.")

        st.markdown("### Integral Protons Plot (3-day)")
        ipp_data = fetch_json("https://services.swpc.noaa.gov/json/goes/primary/integral-protons-plot-3-day.json")
        if ipp_data and isinstance(ipp_data, list):
            times = [row.get("time_tag") for row in ipp_data if "time_tag" in row]
            fluxes = [row.get("flux", 0) for row in ipp_data]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=times, y=fluxes, mode="lines", name="Proton Plot"))
            fig.update_layout(title="Integral Proton Plot (3-day)", height=220,
                             margin=dict(l=10, r=10, t=30, b=10),
                             xaxis=dict(title="Time", color="#9fc8ff"),
                             yaxis=dict(title="Flux", color="#9fc8ff"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No integral protons plot chart data available.")

        st.markdown("### Magnetometers (3-day)")
        mag_data = fetch_json("https://services.swpc.noaa.gov/json/goes/primary/magnetometers-3-day.json")
        if mag_data and isinstance(mag_data, list):
            times = [row.get("time_tag") for row in mag_data if "time_tag" in row]
            bx = [row.get("bx_gsm", 0) for row in mag_data]
            by = [row.get("by_gsm", 0) for row in mag_data]
            bz = [row.get("bz_gsm", 0) for row in mag_data]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=times, y=bx, mode="lines", name="Bx GSM"))
            fig.add_trace(go.Scatter(x=times, y=by, mode="lines", name="By GSM"))
            fig.add_trace(go.Scatter(x=times, y=bz, mode="lines", name="Bz GSM"))
            fig.update_layout(title="Magnetometers (3-day)", height=220,
                             margin=dict(l=10, r=10, t=30, b=10),
                             xaxis=dict(title="Time", color="#9fc8ff"),
                             yaxis=dict(title="nT", color="#9fc8ff"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No magnetometer chart data available.")

    # ---------- Column 2 ----------
    with col2:
        st.markdown("### SUVI Flares (Latest)")
        suvi_data = fetch_json("https://services.swpc.noaa.gov/json/goes/primary/suvi-flares-latest.json")
        if suvi_data and isinstance(suvi_data, list):
            times = [row.get("begin_time") for row in suvi_data if "begin_time" in row]
            intensities = [row.get("peak_intensity", 0) for row in suvi_data]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=times, y=intensities, name="SUVI Flare Intensity"))
            fig.update_layout(title="SUVI Flares (Latest)", height=220,
                             margin=dict(l=10, r=10, t=30, b=10),
                             xaxis=dict(title="Time", color="#9fc8ff"),
                             yaxis=dict(title="Peak Intensity", color="#9fc8ff"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No SUVI flare chart data available.")

        st.markdown("### X-ray Background (7-day)")
        xrb_data = fetch_json("https://services.swpc.noaa.gov/json/goes/primary/xray-background-7-day.json")
        if xrb_data and isinstance(xrb_data, list):
            times = [row.get("time_tag") for row in xrb_data if "time_tag" in row]
            fluxes = [row.get("flux", 0) for row in xrb_data]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=times, y=fluxes, mode="lines", name="X-ray Background Flux"))
            fig.update_layout(title="X-ray Background (7-day)", height=220,
                             margin=dict(l=10, r=10, t=30, b=10),
                             xaxis=dict(title="Time", color="#9fc8ff"),
                             yaxis=dict(title="Flux", color="#9fc8ff"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No X-ray background chart data available.")

        st.markdown("### X-ray Flares (7-day)")
        xrf_data = fetch_json("https://services.swpc.noaa.gov/json/goes/primary/xray-flares-7-day.json")
        if xrf_data and isinstance(xrf_data, list):
            times = [row.get("begin_time") for row in xrf_data if "begin_time" in row]
            fluxes = [row.get("peak_flux", 0) for row in xrf_data]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=times, y=fluxes, name="X-ray Flare Peak Flux"))
            fig.update_layout(title="X-ray Flares (7-day)", height=220,
                             margin=dict(l=10, r=10, t=30, b=10),
                             xaxis=dict(title="Time", color="#9fc8ff"),
                             yaxis=dict(title="Peak Flux", color="#9fc8ff"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No X-ray flares chart data available.")

        st.markdown("### X-rays (3-day)")
        xrays_data = fetch_json("https://services.swpc.noaa.gov/json/goes/primary/xrays-3-day.json")
        if xrays_data and isinstance(xrays_data, list):
            times = [row.get("time_tag") for row in xrays_data if "time_tag" in row]
            fluxes = [row.get("flux", 0) for row in xrays_data]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=times, y=fluxes, mode="lines", name="X-ray Flux"))
            fig.update_layout(title="X-rays (3-day)", height=220,
                             margin=dict(l=10, r=10, t=30, b=10),
                             xaxis=dict(title="Time", color="#9fc8ff"),
                             yaxis=dict(title="Flux", color="#9fc8ff"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No X-rays chart data available.")

    st.caption(f"Last updated: {last_updated()}")

# ========== PDF Export Tab ==========
with tab_pdf:
    st.markdown("## Export Management PDF")
    past, current = get_noaa_rsg_now_and_past()
    three = get_3day_summary()
    day1 = three["days"][0]
    next24 = get_next24_summary()
    bom_aurora_text = get_bom_aurora()
    structured_disc, noaa_discussion_src, noaa_discussion_raw = get_noaa_forecast_text()
    narrative_flags = detect_r_s_watch_flags(structured_disc)  # NEW
    summary_text = make_summary(current, next24)

    st.info("You can customize which sections and charts to include in your PDF below.")
    include_xray = st.checkbox("Include Solar X-ray Chart", True)
    include_proton = st.checkbox("Include Proton Chart", True)
    include_kp = st.checkbox("Include Kp Chart", True)

    fig_xray = create_xray_chart() if include_xray else None
    fig_proton = create_proton_chart() if include_proton else None
    fig_kp = create_kp_chart() if include_kp else None

    # ---- PDF generator with severity-colored cells ----
    def export_management_pdf(
        noaa_discussion_raw,
        past, current, next24, day1,
        bom_aurora, summary_text,
        fig_xray=None, fig_proton=None, fig_kp=None,
        fname="space_weather_management.pdf"
    ):
        chart_paths = []
        try:
            import plotly.io as pio
            pio.kaleido.scope.mathjax = None
            figs = [("xray.png", fig_xray), ("proton.png", fig_proton), ("kp.png", fig_kp)]
            for name, fig in figs:
                if fig is not None:
                    out = f"/tmp/{name}"
                    fig.write_image(out, width=1200, height=500, scale=2)
                    chart_paths.append(out)
        except Exception:
            chart_paths = []

        logo_path = None
        for candidate in ("logo.png", "logo.jpg", "logo.jpeg"):
            if os.path.exists(candidate):
                logo_path = candidate
                break

        # colors
        C_TEAL = (30, 115, 190) if high_contrast else (79, 158, 255)
        C_SLATE = (20, 20, 20) if high_contrast else (40, 47, 56)
        C_GRAY  = (60, 60, 60) if high_contrast else (105, 115, 125)

        COLOR_MAP = {
            "ok": (82, 204, 113),       # #2ecc71
            "caution": (241, 196, 15),  # #f1c40f
            "watch": (230, 126, 34),    # #e67e22
            "severe": (231, 76, 60),    # #e74c3c
        }

        def cls_for_triplet(label, kind):
            # kind: 'r','s','g'
            if kind == 'r':
                return _r_class(label)
            if kind == 's':
                return _s_class(label)
            return _g_class(label)

        # derive day1 labels/classes for "Next 24 h" row (use narrative fallback)
        r_lbl, r_cls = r_label_and_class_for_day(day1, narrative_flags)
        s_lbl, s_cls = s_label_and_class_for_day(day1, narrative_flags)
        g_lbl, g_cls = g_label_and_class_for_day(day1)

        class PDF(FPDF):
            def header(self):
                self.set_fill_color(*C_SLATE); self.rect(0, 0, 210, 20, "F")
                if logo_path:
                    try: self.image(logo_path, x=10, y=4, h=12)
                    except Exception: pass
                self.set_xy(30 if logo_path else 10, 6)
                self.set_text_color(255, 255, 255)
                self.set_font("Helvetica", "B", 16)
                self.cell(0, 8, "Space Weather - Executive Brief", align="L")

            def footer(self):
                self.set_y(-15)
                self.set_draw_color(200, 200, 200); self.line(10, self.get_y(), 200, self.get_y())
                self.set_y(-12); self.set_text_color(*C_GRAY); self.set_font("Helvetica", "", 10)
                ts = datetime.utcnow().strftime("Generated %Y-%m-%d %H:%M UTC")
                self.cell(0, 8, f"{ts}  ·  Page {self.page_no()}", align="R")

        pdf = PDF()
        pdf.set_auto_page_break(auto=True, margin=18)

        def H(title, size=18):
            pdf.set_text_color(*C_TEAL); pdf.set_font("Helvetica", "B", size); pdf.cell(0, 10, title, ln=1)
            pdf.set_text_color(0,0,0)
        
        # Executive Summary
        pdf.add_page(); pdf.ln(6)
        H("Executive Summary", size=18)
        pdf.set_font("Helvetica", "", 12)
        for b in [
            f"Current status (NOAA): {current['r']}/{current['s']}/{current['g']}.",
            f"Next 24 h (UTC): {g_lbl} (Kp~{day1['kp'] if day1['kp'] is not None else '~'}), R1- R2 {day1['r12']}% / R3+ {day1['r3']}%, S1+ {day1['s1']}%.",
            "BOM Aurora: " + (bom_aurora.splitlines()[0] if bom_aurora else "-"),
        ]:
            pdf.set_x(10); pdf.cell(5, 7, "•")
            pdf.multi_cell(0, 7, b)
        pdf.ln(2)

        # Key Metrics table (severity colored)
        def cell_colored(w, h, text, cls_key, border=1, align="C"):
            rgb = COLOR_MAP.get(cls_key, (245, 245, 245))
            pdf.set_fill_color(*rgb)
            pdf.cell(w, h, text, border=border, align=align, fill=True)

        def metrics_row(label, r, s, g, r_cls, s_cls, g_cls, bold=False):
            pdf.set_font("Helvetica", "B" if bold else "", 11)
            # label cell
            fill = 250 if high_contrast else 245
            grid = 120 if high_contrast else 220
            pdf.set_fill_color(fill, fill, fill); pdf.set_draw_color(grid, grid, grid)
            pdf.cell(60, 9, label, border=1, align="L", fill=True)
            # colored cells
            cell_colored(43, 9, r, r_cls)
            cell_colored(43, 9, s, s_cls)
            cell_colored(44, 9, g, g_cls)
            pdf.ln(9)

        H("Key Metrics (NOAA R/S/G)", size=16)
        metrics_row("Scope", "R - Radio Blackouts", "S - Radiation Storms", "G - Geomagnetic Storms",
                    "ok","ok","ok", bold=True)
        metrics_row("Current",
                    current['r'], current['s'], current['g'],
                    cls_for_triplet(current['r'],'r'),
                    cls_for_triplet(current['s'],'s'),
                    cls_for_triplet(current['g'],'g'))
        metrics_row("Past 24 h",
                    past['r'], past['s'], past['g'],
                    cls_for_triplet(past['r'],'r'),
                    cls_for_triplet(past['s'],'s'),
                    cls_for_triplet(past['g'],'g'))
        metrics_row("Next 24 h",
                    r_lbl, s_lbl, g_lbl,
                    r_cls, s_cls, g_cls)
        pdf.ln(3)

        pdf.set_font("Helvetica", "", 12)
        pdf.multi_cell(0, 7, summary_text)

        # Charts page (optional)
        if chart_paths:
            pdf.add_page(); H("Recent Trends", size=16)
            img_w = 180; img_h = 70
            for pth in chart_paths:
                try:
                    pdf.image(pth, x=(210 - img_w) / 2, y=None, w=img_w, h=img_h)
                    pdf.ln(6)
                except Exception:
                    pass
            pdf.ln(1)
            pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(0, 5, "Note: Charts embedded using Plotly + Kaleido. If missing, install kaleido to enable chart export.")

        # NOAA Discussion full text
        if noaa_discussion_raw:
            pdf.add_page(); H("NOAA Discussion (Full Text Excerpt)", size=16)
            pdf.set_font("Helvetica", "", 12)
            for para in (noaa_discussion_raw or "").split("\n\n"):
                pdf.multi_cell(0, 7, para.strip()); pdf.ln(1)

        # Aurora summary
        if bom_aurora:
            pdf.add_page(); H("BOM Aurora (Summary)", size=16)
            pdf.set_font("Helvetica", "", 12)
            pdf.multi_cell(0, 7, bom_aurora)

        try:
            pdf_bytes = pdf.output(dest="S").encode("latin-1")
        except Exception:
            pdf_bytes = pdf.output(dest="S").encode("latin-1", errors="replace")
        b64 = base64.b64encode(pdf_bytes).decode()
        return b64

    if st.button("Generate PDF"):
        pdf_base64 = export_management_pdf(
            noaa_discussion_raw=noaa_discussion_raw or structured_disc["_reflowed"],
            past=past,
            current=current,
            next24=next24,
            day1=day1,
            bom_aurora=bom_aurora_text,
            summary_text=summary_text,
            fig_xray=fig_xray,
            fig_proton=fig_proton,
            fig_kp=fig_kp,
            fname="space_weather_management.pdf"
        )
        st.markdown(
            f'<a href="data:application/pdf;base64,{pdf_base64}" download="space_weather_management.pdf">📄 Download PDF</a>',
            unsafe_allow_html=True
        )

# ========== Help & Info Tab ==========
with tab_help:
    st.markdown("## Help & Information")
    st.info(
        "This dashboard displays real-time space weather data from NOAA and BOM, including solar activity, "
        "geomagnetic storms, and aurora outlooks. Use the tabs above to view detailed charts, expert data, "
        "and export management-grade PDFs. See the sidebar for settings and accessibility options."
    )

    st.markdown("""
### Key Metrics Explained
- **R scale**: Radio blackout risk due to solar flares  
- **S scale**: Solar radiation storms  
- **G scale**: Geomagnetic storm risk (Kp index)  
- **Aurora**: Visibility/disruption risk from BOM
""")

    st.markdown("### Data Sources")
    st.markdown(
        "- NOAA Space Weather Prediction Center (SWPC) feeds  \n"
        "  <https://www.swpc.noaa.gov/>  \n"
        "- Australian Bureau of Meteorology (BOM) Space Weather Service (SWS) / Aurora API  \n"
        "  <https://www.sws.bom.gov.au/>"
    )

    st.markdown("### Credits & Attribution")
    st.markdown(
        "This application acknowledges and thanks the following organisations for their public data and services:\n\n"
        "- **National Oceanic and Atmospheric Administration (NOAA)** — "
        "[Space Weather Prediction Center (SWPC)](https://www.swpc.noaa.gov/)\n"
        "- **Australian Bureau of Meteorology (BOM)** — "
        "[Space Weather Service (SWS)](https://www.sws.bom.gov.au/)\n\n"
        "Data are used under each provider’s respective terms of use and availability."
    )

    st.markdown("### Accessibility")
    st.markdown("High-contrast and text-only modes available. All metrics and charts include ARIA labels.")

    st.markdown("### Feedback")
    st.markdown(
        "For feature requests or bug reports, please submit an issue on "
        "[GitHub](https://github.com/novellgeek/space-weather/issues)."
    )
    
    st.caption(f"Last updated: {last_updated()} (NOAA & BOM data refresh every 10 minutes)")



# ========== Footer ==========
st.caption(f"Server time: {last_updated()}  •  Refresh page to update feeds.")

#51585962-2fdd-4cf5-9d9e-74cdd09e3bab
