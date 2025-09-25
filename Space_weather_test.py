# Space Weather Dashboard — BOM-style Overview + Severity + Impact + PDF colors
# ---------------------------------------------------------------------------
# pip install: streamlit requests plotly numpy scipy fpdf pyspaceweather kaleido (optional)
# ---------------------------------------------------------------------------

import os
import base64
import re
from datetime import datetime

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
    BOM_API_KEY = "51585962-2fdd-4cf5-9d9e-74cdd09e3bab"  # TODO: replace for your environment
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

# -------- NOAA text helpers & parser --------
def _strip_meta_lines(raw_txt: str) -> list[str]:
    lines = [ln.rstrip() for ln in raw_txt.splitlines()]
    keep = []
    for ln in lines:
        t = ln.strip()
        if not t:
            keep.append("")
            continue
        if any(t.upper().startswith(x) for x in [
            "FORECASTER:", "ISSUED:", "VALID:", "SOURCE:", "PRODUCT:",
            "WWW.", "HTTP", "HTTPS"
        ]):
            continue
        if set(t) <= set("-_=*#") and len(t) > 10:
            continue
        keep.append(ln)
    # collapse >2 blank lines
    out = []
    blank_run = 0
    for ln in keep:
        if ln.strip() == "":
            blank_run += 1
            if blank_run <= 2:
                out.append("")
        else:
            blank_run = 0
            out.append(ln)
    return out

def parse_discussion_structured(raw_txt: str) -> dict:
    lines = _strip_meta_lines(raw_txt)

    name_map = {
        "solar activity": "solar_activity",
        "energetic particle": "energetic_particle",
        "solar wind": "solar_wind",
        "geospace": "geospace",
        "geo space": "geospace",
        "geo-space": "geospace",
    }

    main_header_re = re.compile(r"^(Solar\s*Activity|Energetic\s*Particle|Solar\s*Wind|Geo[\s-]*Space|Geospace)\b", re.I)
    sub_header_re  = re.compile(r"^\.(?:24\s*hr\s*summary|summary)\b|^\.(?:forecast|forcast)\b", re.I)

    data = {
        "solar_activity": {"summary": "", "forecast": ""},
        "energetic_particle": {"summary": "", "forecast": ""},
        "solar_wind": {"summary": "", "forecast": ""},
        "geospace": {"summary": "", "forecast": ""},
        "_reflowed": ""
    }

    current_sec_key = None
    current_sub = None
    buf = []

    def flush_buf():
        nonlocal buf, current_sec_key, current_sub
        if current_sec_key:
            text = "\n".join([b.strip() for b in buf]).strip()
            if text:
                if current_sub is None:
                    data[current_sec_key]["summary"] = (
                        (data[current_sec_key]["summary"] + "\n\n" if data[current_sec_key]["summary"] else "")
                        + text
                    )
                else:
                    data[current_sec_key][current_sub] = (
                        (data[current_sec_key][current_sub] + "\n\n" if data[current_sec_key][current_sub] else "")
                        + text
                    )
        buf = []

    for ln in lines:
        mh = main_header_re.search(ln)
        if mh:
            flush_buf()
            header = mh.group(1).lower().replace("  ", " ").strip()
            for alias, key in name_map.items():
                if alias in header:
                    current_sec_key = key
                    break
            else:
                current_sec_key = None
            current_sub = None
            continue

        sh = sub_header_re.search(ln)
        if sh and current_sec_key:
            flush_buf()
            low = ln.lower()
            if "forecast" in low or "forcast" in low:
                current_sub = "forecast"
            else:
                current_sub = "summary"
            continue

        if current_sec_key:
            buf.append(ln)

    flush_buf()

    reflow = []
    for ln in lines:
        if ln.strip():
            reflow.append(ln.strip())
        else:
            reflow.append("")
    data["_reflowed"] = "\n".join(reflow).strip() or raw_txt.strip()
    return data

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
    r = (r or "").upper()
    if r.startswith("R0"): return "ok"
    if r.startswith("R1"): return "caution"
    if r.startswith("R2"): return "watch"
    return "severe"

def _s_class(s: str) -> str:
    s = (s or "").upper()
    if s.startswith("S0"): return "ok"
    if s.startswith("S1"): return "caution"
    return "severe"

def _g_class(g: str) -> str:
    g = (g or "").upper()
    if g.startswith("G0"): return "ok"
    if g.startswith("G1"): return "caution"
    if g.startswith("G2"): return "watch"
    return "severe"

def r_label_and_class_for_day(day: dict) -> tuple[str, str]:
    if (day.get("r3", 0) or 0) >= 1:
        return "R3+", "severe"
    if (day.get("r12", 0) or 0) >= 10:
        return "R1–R2", "caution"
    return "R0", "ok"

def s_label_and_class_for_day(day: dict) -> tuple[str, str]:
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
    urls = [
        "https://services.swpc.noaa.gov/text/discussion.txt",
        "https://services.swpc.noaa.gov/text/3-day-forecast.txt",
    ]
    for url in urls:
        txt = fetch_text(url) or ""
        if not txt:
            continue
        lines = [ln.rstrip() for ln in txt.splitlines()]
        block = []
        for ln in lines:
            block.append(ln)
            if ln.startswith("III.") or ln.lower().startswith("synopsis"):
                break
        top = "\n".join(block).strip() if block else txt
        try:
            structured = parse_discussion_structured(top)
        except Exception:
            structured = {
                "solar_activity": {"summary": "", "forecast": ""},
                "energetic_particle": {"summary": "", "forecast": ""},
                "solar_wind": {"summary": "", "forecast": ""},
                "geospace": {"summary": "", "forecast": ""},
                "_reflowed": top
            }
        return structured, url, top

    return {
        "solar_activity": {"summary": "", "forecast": ""},
        "energetic_particle": {"summary": "", "forecast": ""},
        "solar_wind": {"summary": "", "forecast": ""},
        "geospace": {"summary": "", "forecast": ""},
        "_reflowed": "NOAA forecast discussion unavailable."
    }, None, ""

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

/* three-card summaries */
.tri-grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:.75rem; margin-top:.75rem; }}
.tri-card {{ background:var(--card); border:1px solid var(--border); border-radius:.6rem; padding:.8rem; }}
.tri-card h5 {{ margin:.1rem 0 .4rem 0; color:#cfe3ff; font-size:1rem; }}
.tri-card .body {{ color:#cfe3ff; opacity:.95; line-height:1.35; }}

/* severity colors for R/S/G chips */
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
with tab_overview:
    st.markdown("## Space Weather Dashboard - Overview")

    # --- data pulls ---
    past, current = get_noaa_rsg_now_and_past()
    three = get_3day_summary()
    day1, day2, day3 = three["days"]
    structured_disc, noaa_discussion_src, _raw = get_noaa_forecast_text()
    src_note = noaa_discussion_src.split('/')[-1] if noaa_discussion_src else 'unavailable'

    # --- helpers for BOM grid ---
    def _pill(txt, severity_cls):
        return f"<span class='rs-pill {severity_cls}'>{txt}</span>"

    def col_block(title, r_tuple, s_tuple, g_tuple, subline_html=""):
        r_lbl, r_cls = r_tuple
        s_lbl, s_cls = s_tuple
        g_lbl, g_cls = g_tuple
        return f"""
        <div class="box">
          <h5>{title}</h5>
          <div class="rs-inline">
            {_pill(r_lbl, r_cls)}
            {_pill(s_lbl, s_cls)}
            {_pill(g_lbl, g_cls)}
          </div>
          <div class="subline">{subline_html}</div>
        </div>"""

    # --- Past & Current ---
    past_html = col_block(
        "Past 24 hours",
        (past['r'], _r_class(past['r'])),
        (past['s'], _s_class(past['s'])),
        (past['g'], _g_class(past['g'])),
        "Radio blackouts · Radiation storms · Geomagnetic storms"
    )
    curr_html = col_block(
        "Current",
        (current['r'], _r_class(current['r'])),
        (current['s'], _s_class(current['s'])),
        (current['g'], _g_class(current['g'])),
        "Radio blackouts · Radiation storms · Geomagnetic storms"
    )

    # --- Day 1–3 ---
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

    r3, r3_cls = r_label_and_class_for_day(day3)
    s3, s3_cls = s_label_and_class_for_day(day3)
    g3, g3_cls = g_label_and_class_for_day(day3)
    d3_html = col_block(
        "Day 3 forecast",
        (r3, r3_cls), (s3, s3_cls), (g3, g3_cls),
        f"R1–R2: {day3['r12']}% · R3+: {day3['r3']}% · S1+: {day3['s1']}% · Kp≈{day3['kp'] if day3['kp'] is not None else '~'}"
    )

    # --- render 5-card BOM grid ---
    st.markdown(
        f"<div class='grid-bom'>{past_html}{curr_html}{d1_html}{d2_html}{d3_html}</div>",
        unsafe_allow_html=True
    )
    st.caption(f"Last updated: {last_updated()} · Source: {src_note}")

    # ----------------------------------------------------------------
    # (If needed) CSS for the three summary cards — add this to your
    # global <style> once, then remove this comment & block.
    # .tri-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:.75rem; margin-top:.75rem; }
    # .tri-card { background:var(--card); border:1px solid var(--border); border-radius:.6rem; padding:.8rem; }
    # .tri-card h5 { margin:.1rem 0 .4rem 0; color:#cfe3ff; font-size:1rem; }
    # .tri-card .body { color:#cfe3ff; opacity:.95; line-height:1.35; }
    # ----------------------------------------------------------------
    # --- Impact matrix ---
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
    
    # --- Three short summaries (Solar Activity / Solar Wind / Geospace) ---
    def _short(txt: str, max_chars: int = 260):
        if not txt:
            return ""
        m = re.search(r"(.+?\.)", txt.strip(), re.S)
        s = (m.group(1) if m else txt.strip()).replace("\n", " ")
        return s if len(s) <= max_chars else (s[:max_chars-1].rstrip() + "…")

    sa_summary = (structured_disc.get("solar_activity", {}) or {}).get("summary", "")
    sw_summary = (structured_disc.get("solar_wind", {}) or {}).get("summary", "")
    gs_summary = (structured_disc.get("geospace", {}) or {}).get("summary", "")
    # --- Three full 24 hr Summaries (text columns) ---
    sa_full = (structured_disc.get("solar_activity", {}) or {}).get("summary", "").strip()
    sw_full = (structured_disc.get("solar_wind", {}) or {}).get("summary", "").strip()
    gs_full = (structured_disc.get("geospace", {}) or {}).get("summary", "").strip()

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("### Solar Activity — 24 hr Summary")
        st.markdown(sa_full.replace("\n", "<br>") if sa_full else "—", unsafe_allow_html=True)

    with c2:
        st.markdown("### Solar Wind — 24 hr Summary")
        st.markdown(sw_full.replace("\n", "<br>") if sw_full else "—", unsafe_allow_html=True)

    with c3:
        st.markdown("### Geospace — 24 hr Summary")
        st.markdown(gs_full.replace("\n", "<br>") if gs_full else "—", unsafe_allow_html=True)


    cards_html = f"""
    <div class="tri-grid">
      <div class="tri-card">
        <h5>Solar Activity</h5>
        <div class="body">{_short(sa_summary) or "—"}</div>
      </div>
      <div class="tri-card">
        <h5>Solar Wind</h5>
        <div class="body">{_short(sw_summary) or "—"}</div>
      </div>
      <div class="tri-card">
        <h5>Geospace</h5>
        <div class="body">{_short(gs_summary) or "—"}</div>
      </div>
    </div>
    """
    st.markdown(cards_html, unsafe_allow_html=True)






# ========== Charts Tab ==========
with tab_charts:
    st.markdown("## Space Weather Analytics (Two Columns)")

    time_ranges = {
        "Last 6h": 6*12,       # assuming 5 min intervals
        "Last 24h": 24*12,
        "Full record": None
    }
    selected_range = st.selectbox("Select time range", list(time_ranges.keys()))
    smooth = st.checkbox("Apply 1-hour moving average", value=True)

    def stats_block(times, vals, label, threshold=None):
        if vals is None or len(vals) == 0:
            return ""
        arr = np.array(vals)
        current_val = arr[-1]
        avg = np.mean(arr)
        std = np.std(arr)
        minv = np.min(arr)
        maxv = np.max(arr)
        trend = "↗️ rising" if arr[-1] > arr[0] else ("↘️ falling" if arr[-1] < arr[0] else "⏸️ flat")
        alert = ""
        if threshold is not None and current_val > threshold:
            alert = f"**ALERT: {label} above threshold ({threshold})!**"
        st.markdown(f"""
        **{label} Stats:**  
        - Current: `{current_val:.2e}`  
        - Mean: `{avg:.2e}`  
        - Std Dev: `{std:.2e}`  
        - Min: `{minv:.2e}`  
        - Max: `{maxv:.2e}`  
        - Trend: {trend}  
        {alert}
        """)

    col1, col2 = st.columns(2)

    # ------------- COLUMN 1 -------------
    with col1:
        st.markdown("### Differential Electrons (1-day)")
        url = "https://services.swpc.noaa.gov/json/goes/primary/differential-electrons-1-day.json"
        data = fetch_json(url)
        if data and isinstance(data, list):
            times = [row.get("time_tag") for row in data if "time_tag" in row]
            fluxes = [row.get("flux", 0) for row in data]
            if time_ranges[selected_range]:
                times = times[-time_ranges[selected_range]:]
                fluxes = fluxes[-time_ranges[selected_range]:]
            if smooth and len(fluxes) > 12:
                fluxes = uniform_filter1d(fluxes, size=12)
            stats_block(times, fluxes, "Differential Electrons", threshold=None)
            fig_de = go.Figure()
            fig_de.add_trace(go.Scatter(x=times, y=fluxes, mode="lines", name="Electron Flux"))
            fig_de.update_layout(title="Differential Electrons (1-day)", height=220,
                             margin=dict(l=10, r=10, t=30, b=10),
                             xaxis=dict(title="Time", color="#9fc8ff"),
                             yaxis=dict(title="Flux", color="#9fc8ff"))
            st.plotly_chart(fig_de, use_container_width=True, key="differential_electrons")
        else:
            st.caption("No electron data available.")

        st.markdown("### Differential Protons (1-day)")
        url = "https://services.swpc.noaa.gov/json/goes/primary/differential-protons-1-day.json"
        data = fetch_json(url)
        if data and isinstance(data, list):
            times = [row.get("time_tag") for row in data if "time_tag" in row]
            fluxes = [row.get("flux", 0) for row in data]
            if time_ranges[selected_range]:
                times = times[-time_ranges[selected_range]:]
                fluxes = fluxes[-time_ranges[selected_range]:]
            if smooth and len(fluxes) > 12:
                fluxes = uniform_filter1d(fluxes, size=12)
            stats_block(times, fluxes, "Differential Protons", threshold=None)
            fig_dp = go.Figure()
            fig_dp.add_trace(go.Scatter(x=times, y=fluxes, mode="lines", name="Proton Flux"))
            fig_dp.update_layout(title="Differential Protons (1-day)", height=220,
                             margin=dict(l=10, r=10, t=30, b=10),
                             xaxis=dict(title="Time", color="#9fc8ff"),
                             yaxis=dict(title="Flux", color="#9fc8ff"))
            st.plotly_chart(fig_dp, use_container_width=True, key="differential_protons")
        else:
            st.caption("No proton data available.")

        st.markdown("### Integral Electrons (1-day)")
        url = "https://services.swpc.noaa.gov/json/goes/primary/integral-electrons-1-day.json"
        data = fetch_json(url)
        if data and isinstance(data, list):
            times = [row.get("time_tag") for row in data if "time_tag" in row]
            fluxes = [row.get("flux", 0) for row in data]
            if time_ranges[selected_range]:
                times = times[-time_ranges[selected_range]:]
                fluxes = fluxes[-time_ranges[selected_range]:]
            if smooth and len(fluxes) > 12:
                fluxes = uniform_filter1d(fluxes, size=12)
            stats_block(times, fluxes, "Integral Electrons", threshold=None)
            fig_ie = go.Figure()
            fig_ie.add_trace(go.Scatter(x=times, y=fluxes, mode="lines", name="Integral Electron Flux"))
            fig_ie.update_layout(title="Integral Electrons (1-day)", height=220,
                             margin=dict(l=10, r=10, t=30, b=10),
                             xaxis=dict(title="Time", color="#9fc8ff"),
                             yaxis=dict(title="Flux", color="#9fc8ff"))
            st.plotly_chart(fig_ie, use_container_width=True, key="integral_electrons")
        else:
            st.caption("No integral electron data available.")

        st.markdown("### Integral Protons Plot (1-day)")
        url = "https://services.swpc.noaa.gov/json/goes/primary/integral-protons-plot-1-day.json"
        data = fetch_json(url)
        if data and isinstance(data, list):
            times = [row.get("time_tag") for row in data if "time_tag" in row]
            fluxes = [row.get("flux", 0) for row in data]
            if time_ranges[selected_range]:
                times = times[-time_ranges[selected_range]:]
                fluxes = fluxes[-time_ranges[selected_range]:]
            if smooth and len(fluxes) > 12:
                fluxes = uniform_filter1d(fluxes, size=12)
            stats_block(times, fluxes, "Integral Protons Plot", threshold=None)
            fig_ipp = go.Figure()
            fig_ipp.add_trace(go.Scatter(x=times, y=fluxes, mode="lines", name="Integral Proton Flux"))
            fig_ipp.update_layout(title="Integral Protons Plot (1-day)", height=220,
                             margin=dict(l=10, r=10, t=30, b=10),
                             xaxis=dict(title="Time", color="#9fc8ff"),
                             yaxis=dict(title="Flux", color="#9fc8ff"))
            st.plotly_chart(fig_ipp, use_container_width=True, key="integral_protons_plot")
        else:
            st.caption("No integral proton plot data available.")

    # ------------- COLUMN 2 -------------
    with col2:
        st.markdown("### Magnetometers (1-day)")
        url = "https://services.swpc.noaa.gov/json/goes/primary/magnetometers-1-day.json"
        data = fetch_json(url)
        if data and isinstance(data, list):
            times = [row.get("time_tag") for row in data if "time_tag" in row]
            bx = [row.get("bx_gsm", 0) for row in data]
            by = [row.get("by_gsm", 0) for row in data]
            bz = [row.get("bz_gsm", 0) for row in data]
            if time_ranges[selected_range]:
                times = times[-time_ranges[selected_range]:]
                bx = bx[-time_ranges[selected_range]:]
                by = by[-time_ranges[selected_range]:]
                bz = bz[-time_ranges[selected_range]:]
            stats_block(times, bx, "Magnetometer Bx", threshold=None)
            stats_block(times, by, "Magnetometer By", threshold=None)
            stats_block(times, bz, "Magnetometer Bz", threshold=None)
            fig_mag = go.Figure()
            fig_mag.add_trace(go.Scatter(x=times, y=bx, mode="lines", name="Bx GSM"))
            fig_mag.add_trace(go.Scatter(x=times, y=by, mode="lines", name="By GSM"))
            fig_mag.add_trace(go.Scatter(x=times, y=bz, mode="lines", name="Bz GSM"))
            fig_mag.update_layout(title="Magnetometers (1-day)", height=220,
                             margin=dict(l=10, r=10, t=30, b=10),
                             xaxis=dict(title="Time", color="#9fc8ff"),
                             yaxis=dict(title="nT", color="#9fc8ff"))
            st.plotly_chart(fig_mag, use_container_width=True, key="magnetometers")
        else:
            st.caption("No magnetometer data available.")

        st.markdown("### SUVI Flares (7-day)")
        url = "https://services.swpc.noaa.gov/json/goes/primary/suvi-flares-7-day.json"
        data = fetch_json(url)
        if data and isinstance(data, list):
            times = [row.get("begin_time") for row in data if "begin_time" in row]
            intensities = [row.get("peak_intensity", 0) for row in data]
            if time_ranges[selected_range]:
                times = times[-time_ranges[selected_range]:]
                intensities = intensities[-time_ranges[selected_range]:]
            stats_block(times, intensities, "SUVI Flare Peak Intensity", threshold=None)
            fig_suvi = go.Figure()
            fig_suvi.add_trace(go.Bar(x=times, y=intensities, name="SUVI Flare Intensity"))
            fig_suvi.update_layout(title="SUVI Flares (7-day)", height=220,
                             margin=dict(l=10, r=10, t=30, b=10),
                             xaxis=dict(title="Time", color="#9fc8ff"),
                             yaxis=dict(title="Peak Intensity", color="#9fc8ff"))
            st.plotly_chart(fig_suvi, use_container_width=True, key="suvi_flares")
        else:
            st.caption("No SUVI flares data available.")

        st.markdown("### X-ray Background (7-day)")
        url = "https://services.swpc.noaa.gov/json/goes/primary/xray-background-7-day.json"
        data = fetch_json(url)
        if data and isinstance(data, list):
            times = [row.get("time_tag") for row in data if "time_tag" in row]
            fluxes = [row.get("flux", 0) for row in data]
            if time_ranges[selected_range]:
                times = times[-time_ranges[selected_range]:]
                fluxes = fluxes[-time_ranges[selected_range]:]
            if smooth and len(fluxes) > 12:
                fluxes = uniform_filter1d(fluxes, size=12)
            stats_block(times, fluxes, "X-ray Background", threshold=1e-7)
            fig_xrb = go.Figure()
            fig_xrb.add_trace(go.Scatter(x=times, y=fluxes, mode="lines", name="X-ray Background Flux"))
            fig_xrb.update_layout(title="X-ray Background (7-day)", height=220,
                             margin=dict(l=10, r=10, t=30, b=10),
                             xaxis=dict(title="Time", color="#9fc8ff"),
                             yaxis=dict(title="Flux", color="#9fc8ff"))
            st.plotly_chart(fig_xrb, use_container_width=True, key="xray_background")
        else:
            st.caption("No X-ray background chart data available.")

        st.markdown("### X-rays (6-hour)")
        url = "https://services.swpc.noaa.gov/json/goes/primary/xrays-6-hour.json"
        data = fetch_json(url)
        if data and isinstance(data, list):
            times = [row.get("time_tag") for row in data if "time_tag" in row]
            fluxes = [row.get("flux", 0) for row in data]
            if time_ranges[selected_range]:
                times = times[-time_ranges[selected_range]:]
                fluxes = fluxes[-time_ranges[selected_range]:]
            if smooth and len(fluxes) > 12:
                fluxes = uniform_filter1d(fluxes, size=12)
            stats_block(times, fluxes, "X-rays (6-hour)", threshold=1e-7)
            fig_xr6 = go.Figure()
            fig_xr6.add_trace(go.Scatter(x=times, y=fluxes, mode="lines", name="X-ray Flux"))
            fig_xr6.update_layout(title="X-rays (6-hour)", height=220,
                             margin=dict(l=10, r=10, t=30, b=10),
                             xaxis=dict(title="Time", color="#9fc8ff"),
                             yaxis=dict(title="Flux", color="#9fc8ff"))
            st.plotly_chart(fig_xr6, use_container_width=True, key="xrays_6hour")
        else:
            st.caption("No X-rays data available.")

    st.caption(f"Last updated: {last_updated()}")

# ========== Forecasts Tab ==========
# ========== Forecasts Tab ==========
with tab_forecast:
    st.markdown("## Three-Day Forecast")

    # Pull structured text + 3-day numbers
    structured_disc, noaa_discussion_src, _raw = get_noaa_forecast_text()
    src_note = noaa_discussion_src.split('/')[-1] if noaa_discussion_src else 'unavailable'
    three = get_3day_summary()
    day1, day2, day3 = three["days"]

    # --- Top: Day 1 • Day 2 • Day 3 quick matrix ---
    def _kp_str(v):
        return "~" if v is None else f"{v:.1f}"

    def day_card(col, title, d):
        with col:
            st.markdown(f"### {title}")
            st.markdown(
                f"- **Geomagnetic (G):** {d['g']}  ·  **Kp≈** {_kp_str(d['kp'])}\n"
                f"- **R1–R2:** {d['r12']}%  ·  **R3+:** {d['r3']}%\n"
                f"- **S1+:** {d['s1']}%"
            )

    c1, c2, c3 = st.columns(3)
    day_card(c1, "Day 1", day1)
    day_card(c2, "Day 2", day2)
    day_card(c3, "Day 3", day3)

    st.divider()

    # --- Narrative (3-Day) — show only "Forecast" text, not 24 hr summaries ---
    st.markdown("### Narrative (3-Day)")
    sections = [
        ("Solar Activity", "solar_activity"),
        ("Energetic Particle", "energetic_particle"),
        ("Solar Wind", "solar_wind"),
        ("Geospace", "geospace"),
    ]
    for title, key in sections:
        fc = (structured_disc.get(key, {}) or {}).get("forecast", "").strip()
        if fc:
            st.markdown(f"#### {title}")
            st.markdown(fc.replace("\n", "<br>"), unsafe_allow_html=True)

    st.caption(f"Source: {src_note}")


# ========== Aurora Tab ==========
with tab_aurora:
    #Rebuild all inputs locally so we don't depend on other-tab variables
    structured_disc, noaa_discussion_src, _raw = get_noaa_forecast_text()
    src_note = noaa_discussion_src.split('/')[-1] if noaa_discussion_src else 'unavailable'
    bom_aurora_text = get_bom_aurora()

    # Build a self-contained HTML block for NOAA text
    def _sec_html(title, sec):
        sec = sec or {}
        has_any = (sec.get("summary") or sec.get("forecast"))
        if not has_any:
            return ""
        s = []
        s.append(f"<h4 style='margin:.2rem 0 .4rem 0;color:#cfe3ff'>{title}</h4>")
        if sec.get("summary"):
            s.append("<div class='subttl' style='margin-top:.2rem;'>24 hr Summary</div>")
            s.append(f"<div style='color:#cfe3ff;margin:.2rem 0 0.8rem 0;'>{sec['summary'].replace('\\n','<br>')}</div>")
        if sec.get("forecast"):
            s.append("<div class='subttl' style='margin-top:.2rem;'>Forecast (3-day)</div>")
            s.append(f"<div style='color:#cfe3ff;margin:.2rem 0 0.8rem 0;'>{sec['forecast'].replace('\\n','<br>')}</div>")
        return "\n".join(s)

    content_html = "".join([
        _sec_html("Solar Activity", structured_disc.get("solar_activity")),
        _sec_html("Energetic Particle", structured_disc.get("energetic_particle")),
        _sec_html("Solar Wind", structured_disc.get("solar_wind")),
        _sec_html("Geospace", structured_disc.get("geospace")),
    ]) or (structured_disc.get("_reflowed", "") or "").replace("\n", "<br>")

    st.markdown(f"""
    <div class='section' role='region' aria-label='NOAA Forecast & Aurora'>
      <div class='subttl'>NOAA Forecast
        <span style="opacity:.7;font-size:.9em;">({src_note})</span>
      </div>
      <div style='margin-top:.6rem;'>{content_html}</div>
      <hr style='border-color:rgba(139,233,253,.18)'>
      <div class='subttl'>BOM Aurora</div>
      <div style='white-space:pre-line;color:#cfe3ff;margin-top:.5rem;'>{bom_aurora_text}</div>
    </div>
    """, unsafe_allow_html=True)

# ========== Expert Data Tab (Charts) ==========
with tab_expert:
    st.markdown("## GOES / L1 Expert Data (Expanded, Charts)")
    col1, col2 = st.columns(2)

    # ---------- Column 1 ----------
    with col1:
        st.markdown("### Differential Electrons (3-day)")
        elec_data = fetch_json("https://services.swpc.noaa.gov/json/goes/primary/differential-electrons-3-day.json")
        if elec_data and isinstance(elec_data, list):
            # Example: Plot electron flux vs time, if available
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
    summary_text = make_summary(current, next24)
    structured_disc, noaa_discussion_src, noaa_discussion_raw = get_noaa_forecast_text()

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

        # derive day1 labels/classes for "Next 24 h" row
        r_lbl, r_cls = r_label_and_class_for_day(day1)
        s_lbl, s_cls = s_label_and_class_for_day(day1)
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
    st.info("This dashboard displays real-time space weather data from NOAA and BOM, including solar activity, geomagnetic storms, and aurora outlooks. \
        Use the tabs above to view detailed charts, expert data, and export management-grade PDFs. See the sidebar for settings and accessibility options.")

    st.markdown("""
    ### Key Metrics Explained
    - **R scale**: Radio blackout risk due to solar flares
    - **S scale**: Solar radiation storms
    - **G scale**: Geomagnetic storm risk (Kp index)
    - **Aurora**: Visibility/disruption risk from BOM
    """)
    st.markdown("### Data Sources")
    st.markdown("- NOAA SWPC feeds\n- BOM Aurora API")
    st.markdown("### Accessibility")
    st.markdown("High-contrast and text-only modes available. All metrics and charts include ARIA labels.")
    st.markdown("### Feedback")
    st.markdown("For feature requests or bug reports, please submit an issue on [GitHub](https://github.com/novellgeek/space-weather/issues).")

# ========== Footer ==========
st.caption(f"Server time: {last_updated()}  •  Refresh page to update feeds.")
