import os
import base64
import re
from datetime import datetime, timedelta

import plotly.graph_objects as go
import requests
import streamlit as st
from fpdf import FPDF
from streamlit.components.v1 import html

# ==========================
# Setup
# ==========================
st.set_page_config(page_title="Space Weather (NOAA + BOM Aurora)", layout="wide")
UA = {"User-Agent": "SpaceWeatherDashboard/2.3 (+streamlit)"}

# ---------- Optional BOM wrapper (env var) ----------
HAVE_BOM = True
BOM_ERR = ""
try:
    from pyspaceweather import SpaceWeather  # local helper module
except Exception as e:
    HAVE_BOM = False
    BOM_ERR = f"pyspaceweather unavailable: {e}"

BOM_API_KEY = os.getenv("BOM_API_KEY", "").strip()
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

# ==========================
# Sidebar: Auto-refresh + BOM status
# ==========================
with st.sidebar:
    st.markdown("### Settings")
    refresh_min = st.slider(
        "Auto-refresh (minutes)", min_value=0, max_value=30, value=10,
        help="Set to 0 to disable automatic refresh."
    )
    # Lightweight, dependency-free auto-refresh via JS
    if refresh_min and refresh_min > 0:
        interval_ms = int(refresh_min * 60 * 1000)
        html(
            f"<script>setTimeout(function(){{ window.location.reload(); }}, {interval_ms});</script>",
            height=0,
        )

    st.markdown("### BOM API")
    if bom:
        st.success("BOM aurora: using BOM_API_KEY from environment.")
    else:
        st.info("BOM aurora disabled.")
        if BOM_ERR:
            st.caption(BOM_ERR)
        st.caption("To enable: export BOM_API_KEY in your environment.")

# ==========================
# Helpers & scales
# ==========================
@st.cache_data(ttl=600, show_spinner=False)
def fetch_json(url, timeout=20):
    r = requests.get(url, timeout=timeout, headers=UA)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=600, show_spinner=False)
def fetch_text(url, timeout=20):
    r = requests.get(url, timeout=timeout, headers=UA)
    r.raise_for_status()
    return r.text

def clamp_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        try:
            return float(str(x).strip())
        except Exception:
            return default

def r_scale(xray_flux_wm2):
    if xray_flux_wm2 >= 1e-4: return ("R2", "yellow")
    if xray_flux_wm2 >= 1e-5: return ("R1", "green")
    return ("R0", "green")

def s_scale(proton_pfu_10mev):
    if proton_pfu_10mev >= 10: return ("S2", "yellow")
    if proton_pfu_10mev >= 1: return ("S1", "green")
    return ("S0", "green")

def g_scale_from_kp(kp):
    k = clamp_float(kp, 0)
    if k >= 7: return ("G3", "red")
    if k >= 5: return ("G2", "yellow")
    if k >= 4: return ("G1", "green")
    return ("G0", "green")

# ==========================
# Reflow SWPC text → paragraphs
# ==========================
def reflow_bulletin(txt: str) -> str:
    """
    Re-wrap SWPC hard-wrapped text into paragraphs.
    - Collapses consecutive non-empty lines
    - Keeps blank lines as paragraph breaks
    - Drops metadata lines starting with ':' or '#'
    """
    lines = [ln.rstrip() for ln in txt.splitlines()]
    paras, buf = [], []
    for ln in lines:
        if ln.startswith(":") or ln.startswith("#"):
            continue
        if ln.strip() == "":
            if buf:
                paras.append(" ".join(buf).strip()); buf = []
        else:
            buf.append(ln.strip())
    if buf:
        paras.append(" ".join(buf).strip())
    return "\n\n".join(paras)

# ==========================
# Data: BOM Aurora
# ==========================
def get_bom_aurora():
    if bom is None:
        return f"Aurora info unavailable. ({BOM_ERR or 'pyspaceweather not installed or API key missing'})"
    try:
        outlooks = bom.get_aurora_outlook()
        watches  = bom.get_aurora_watch()
        alerts   = bom.get_aurora_alert()
        lines = []
        if outlooks: lines.append(f"**Aurora Outlook:** {getattr(outlooks[0],'comments','(no text)')}")
        if watches:  lines.append(f"**Aurora Watch:** {getattr(watches[0],'comments','(no text)')}")
        if alerts:   lines.append(f"**Aurora Alert:** {getattr(alerts[0],'description','(no text)')}")
        return "\n".join(lines) if lines else "No aurora alerts/outlooks."
    except Exception as e:
        return f"Aurora info unavailable. ({e})"

# ==========================
# Data: NOAA current & past
# ==========================
def get_noaa_rsg_now_and_past():
    # Kp 1-minute
    try:
        kp = fetch_json("https://services.swpc.noaa.gov/json/planetary_k_index_1m.json")
        k_now = clamp_float(kp[-1].get("kp_index", 0)) if kp else 0.0
        last = kp[-24:] if kp and len(kp) >= 24 else kp
        k_past = max(clamp_float(v.get("kp_index", 0)) for v in last) if last else k_now
    except Exception:
        k_now = k_past = 0.0

    # GOES X-ray (latest list)
    try:
        xr = fetch_json("https://services.swpc.noaa.gov/json/goes/primary/xray-flares-latest.json")
        x_now = clamp_float(xr[-1].get("flux", 0)) if xr else 0.0
        last = xr[-24:] if xr and len(xr) >= 24 else xr
        x_past = max(clamp_float(v.get("flux", 0)) for v in last) if last else x_now
    except Exception:
        x_now = x_past = 0.0

    # Proton >10 MeV (1-day)
    try:
        pr = fetch_json("https://services.swpc.noaa.gov/json/goes/primary/proton-flux-1-day.json")
        p_now = clamp_float(pr[-1].get("flux", 0)) if pr else 0.0
        last = pr[-24:] if pr and len(pr) >= 24 else pr
        p_past = max(clamp_float(v.get("flux", 0)) for v in last) if last else p_now
    except Exception:
        p_now = p_past = 0.0

    r_now, r_now_col = r_scale(x_now)
    r_past, r_past_col = r_scale(x_past)
    s_now, s_now_col = s_scale(p_now)
    s_past, s_past_col = s_scale(p_past)
    g_now, g_now_col = g_scale_from_kp(k_now)
    g_past, g_past_col = g_scale_from_kp(k_past)

    current = {"r": r_now, "r_txt": "Radio blackouts", "r_status": "No" if r_now == "R0" else "Active", "r_color": r_now_col,
               "s": s_now, "s_txt": "Radiation storms", "s_status": "No" if s_now == "S0" else "Active", "s_color": s_now_col,
               "g": g_now, "g_txt": "Geomagnetic storms", "g_status": "No" if g_now == "G0" else "Active", "g_color": g_now_col}
    past = {"r": r_past, "r_txt": "Radio blackouts", "r_status": "No" if r_past == "R0" else "Active", "r_color": r_past_col,
            "s": s_past, "s_txt": "Radiation storms", "s_status": "No" if s_past == "S0" else "Active", "s_color": s_past_col,
            "g": g_past, "g_txt": "Geomagnetic storms", "g_status": "No" if g_past == "G0" else "Active", "g_color": g_past_col}
    return past, current

# ==========================
# Data: NOAA discussion (fixed + reflow)
# ==========================
def get_noaa_forecast_text():
    """
    Returns (html_reflowed, source_url, raw_top_block)
    """
    urls = [
        "https://services.swpc.noaa.gov/text/discussion.txt",            # correct product
        "https://services.swpc.noaa.gov/text/forecast-discussion.txt",   # legacy/alt
        "https://services.swpc.noaa.gov/text/3-day-forecast.txt",        # fallback
    ]
    for url in urls:
        try:
            txt = fetch_text(url)
            # keep the first main block (short)
            lines = [ln.rstrip() for ln in txt.splitlines()]
            block = []
            for ln in lines:
                if ln.strip():
                    block.append(ln)
                if ln.startswith("III.") or ln.lower().startswith("synopsis"):
                    break
            top = "\n".join(block).strip() if block else txt
            return reflow_bulletin(top), url, top
        except Exception:
            continue
    return "NOAA forecast discussion unavailable.", None, ""

# ==========================
# Data: NOAA 3-day text → take only NEXT 24H
# ==========================
def parse_three_day_for_next24(txt: str):
    """Return a dict for the first 24h column. If missing, fall back to day-2."""
    r12 = r3p = s1p = None
    kpmax_day1 = kpmax_day2 = None
    clean = " ".join(txt.split())

    m = re.search(r"(?:R1\s*[-–]\s*R2)\s+(\d+)%\s+(\d+)%\s+(\d+)%.*?(?:R3\s*or\s*greater)\s+(\d+)%\s+(\d+)%\s+(\d+)%", clean, re.I)
    if m:
        r12_day1, r12_day2, _d3, r3_day1, r3_day2, _d3b = map(int, m.groups())
        r12, r3p = r12_day1, r3_day1
        r12_fallback, r3p_fallback = r12_day2, r3_day2
    else:
        r12_fallback = r3p_fallback = None

    ms = re.search(r"S1\s*or\s*greater\s+(\d+)%\s+(\d+)%\s+(\d+)%", clean, re.I)
    if ms:
        s1_day1, s1_day2, _ = map(int, ms.groups())
        s1p = s1_day1
        s1_fallback = s1_day2
    else:
        s1_fallback = None

    triplets = re.findall(r"\d{2}-\d{2}UT\s+(\d\.\d{2})\s+(\d\.\d{2})\s+(\d\.\d{2})", clean)
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
            kpmax_day1, kpmax_day2 = k, k

    if r12 is None and r12_fallback is not None: r12 = r12_fallback
    if r3p is None and r3p_fallback is not None: r3p = r3p_fallback
    if s1p is None and s1_fallback is not None: s1p = s1_fallback
    kpmax = kpmax_day1 if kpmax_day1 is not None else kpmax_day2

    r_bucket = "R0"
    if (r12 or 0) >= 10: r_bucket = "R1"
    if (r3p or 0) >= 1:  r_bucket = "R2"
    s_bucket = "S0"
    if (s1p or 0) >= 10: s_bucket = "S1"
    if kpmax is not None:
        g_bucket, _ = g_scale_from_kp(kpmax)
    else:
        g_bucket = "G0"

    return {
        "r_bucket": r_bucket, "r12_prob": r12 or 0, "r3_prob": r3p or 0,
        "s_bucket": s_bucket, "s1_prob": s1p or 0,
        "g_bucket": g_bucket, "kp_max": f"{kpmax:.2f}" if kpmax is not None else "—"
    }

@st.cache_data(ttl=600, show_spinner=False)
def get_next24_summary():
    try:
        txt = fetch_text("https://services.swpc.noaa.gov/text/3-day-forecast.txt")
        return parse_three_day_for_next24(txt)
    except Exception:
        return {"r_bucket":"R0","r12_prob":0,"r3_prob":0,"s_bucket":"S0","s1_prob":0,"g_bucket":"G0","kp_max":"—"}

# ==========================
# Extra telemetry
# ==========================
def get_goes_telemetry():
    wind_kms = "—"; bt = "—"; bz = "—"; f10_7 = "—"
    try:
        mag = fetch_json("https://services.swpc.noaa.gov/json/dscovr/real_time_mag-1-day.json")
        if mag:
            last = mag[-1]; bt = str(last.get("bt", "—")); bz = str(last.get("bz_gsm", "—"))
    except Exception: pass
    try:
        plasma = fetch_json("https://services.swpc.noaa.gov/json/dscovr/real_time_plasma-1-day.json")
        if plasma:
            last = plasma[-1]; wind_kms = str(last.get("speed", "—"))
    except Exception: pass
    try:
        f10 = fetch_json("https://services.swpc.noaa.gov/json/f10.7cm/observed.json")
        if f10:
            f10_7 = str(f10[-1].get("flux", "—"))
    except Exception: pass
    return wind_kms, bt, bz, f10_7

# ==========================
# Tiny summary
# ==========================
def make_summary(current, next24):
    return (f"Now: {current['r']}/{current['s']}/{current['g']}. "
            f"Next 24 h: {next24['g_bucket']} (Kp≈{next24['kp_max']}), "
            f"R1–R2 {next24['r12_prob']}% / R3+ {next24['r3_prob']}%, S1+ {next24['s1_prob']}%.")

# ==========================
# PDF export
# ==========================
def pdf_download_button(noaa_discussion_raw, past, current, next24, bom_aurora, summary_text):
    def safe_ascii(text):
        return (text or "").replace("–", "-").encode("ascii", "replace").decode("ascii")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt="Space Weather Dashboard (NOAA + BOM Aurora)", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=safe_ascii(summary_text))
    pdf.multi_cell(0, 10, txt=safe_ascii(noaa_discussion_raw))
    pdf.multi_cell(0, 10, txt=safe_ascii(bom_aurora))
    pdf.cell(100, 10, txt="Past 24h (NOAA):", ln=True)
    pdf.cell(100, 10, txt=safe_ascii(f"{past['r']} {past['r_txt']}: {past['r_status']}"), ln=True)
    pdf.cell(100, 10, txt=safe_ascii(f"{past['s']} {past['s_txt']}: {past['s_status']}"), ln=True)
    pdf.cell(100, 10, txt=safe_ascii(f"{past['g']} {past['g_txt']}: {past['g_status']}"), ln=True)
    pdf.cell(100, 10, txt="Current (NOAA):", ln=True)
    pdf.cell(100, 10, txt=safe_ascii(f"{current['r']} {current['r_txt']}: {current['r_status']}"), ln=True)
    pdf.cell(100, 10, txt=safe_ascii(f"{current['s']} {current['s_txt']}: {current['s_status']}"), ln=True)
    pdf.cell(100, 10, txt=safe_ascii(f"{current['g']} {current['g_txt']}: {current['g_status']}"), ln=True)
    pdf.cell(100, 10, txt="Next 24h (NOAA):", ln=True)
    pdf.cell(100, 10, txt=safe_ascii(
        f"{next24['g_bucket']} (Kp~{next24['kp_max']}), "
        f"R1-R2 {next24['r12_prob']}%, R3+ {next24['r3_prob']}%, S1+ {next24['s1_prob']}%"
    ), ln=True)
    pdf.multi_cell(0, 10, txt="For explanation of the R, S and G scales, see NOAA Space Weather Scales. BOM aurora alerts supplement NOAA data.")
    try:
        pdf_output = pdf.output(dest='S').encode('latin1')
    except Exception:
        pdf_output = pdf.output(dest='S').encode('latin1', errors='ignore')
    b64 = base64.b64encode(pdf_output).decode()
    st.markdown(f'<a href="data:application/pdf;base64,{b64}" download="space_weather.pdf">Export PDF</a>', unsafe_allow_html=True)

# ==========================
# UI Styles
# ==========================
st.markdown("""
<style>
:root { --neon: #8be9fd; --glow: 0 0 18px rgba(139,233,253,.25); }
.main { background: radial-gradient(1200px 600px at 30% -10%, rgba(71,131,255,.08), transparent) #0d1419; }
h1,h2,h3 { font-family: 'Orbitron', Arial, sans-serif; letter-spacing:.06em; }
h1 { color: var(--neon); text-shadow: var(--glow); }
.section { background:#111a21; border:1px solid rgba(139,233,253,.25); border-radius:14px;
  box-shadow: 0 0 24px rgba(0,0,0,.4), inset 0 0 80px rgba(139,233,253,.06);
  padding: 18px 20px; margin-bottom: 18px; }
.subttl { color:#a8c7ff; font-weight:700; letter-spacing:.05em; }
.kv { color:#dbe7ff; font-family: ui-monospace, Menlo, Consolas, "Courier New", monospace; }
.badge-col { display:flex; gap:12px; }
.neon-badge { width:70px; height:70px; border-radius:14px; background:linear-gradient(180deg,#1b252d,#151c22);
  display:flex; align-items:center; justify-content:center; box-shadow: 0 0 22px rgba(139,233,253,.2), inset 0 0 22px rgba(255,255,255,.03);
  border:1px solid rgba(139,233,253,.25); font-weight:800; font-size:24px; color:#bfffdc; }
.neon-badge.yellow { color:#ffe49b; } .neon-badge.red { color:#ffb3b3; }
.footer { color:#8aa; font-size:.9rem; }
</style>
""", unsafe_allow_html=True)

# ==========================
# Build dashboard
# ==========================
st.markdown("<h1 style='text-align:left;'>SPACE WEATHER:</h1>", unsafe_allow_html=True)

# Charts
col1, col2, col3 = st.columns([1.1, 1.1, 1])

x_time, x_vals = [], []
p_time, p_vals = [], []
kp_time, kp_vals = [], []

try:
    pr_series = fetch_json("https://services.swpc.noaa.gov/json/goes/primary/proton-flux-1-day.json")
    for row in pr_series:
        p_time.append(row.get("time_tag")); p_vals.append(clamp_float(row.get("flux", 0)))
except Exception:
    pass

try:
    xr_series = fetch_json("https://services.swpc.noaa.gov/json/goes/primary/xray-flares-latest.json")
    for row in xr_series[-24:]:
        x_time.append(row.get("time_tag")); x_vals.append(clamp_float(row.get("flux", 0)))
except Exception:
    pass

try:
    kp_series = fetch_json("https://services.swpc.noaa.gov/json/planetary_k_index_1m.json")
    tail = kp_series[-1000:] if len(kp_series) > 1000 else kp_series
    for row in tail:
        kp_time.append(row.get("time_tag")); kp_vals.append(clamp_float(row.get("kp_index", 0)))
except Exception:
    pass

with col1:
    st.markdown("<div class='section'><div class='subttl'>SOLAR X-RAY FLUX</div>", unsafe_allow_html=True)
    if x_time:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_time, y=x_vals, mode="lines", name="X-ray Flux"))
        fig.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=220,
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          xaxis=dict(showgrid=False, color="#9fc8ff"),
                          yaxis=dict(showgrid=True, gridcolor="rgba(160,190,255,.15)", color="#9fc8ff"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.caption("No X-ray series available.")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='section'><div class='subttl'>SOLAR PROTON FLUX</div>", unsafe_allow_html=True)
    if p_time:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=p_time, y=p_vals, mode="lines", name="Proton Flux"))
        fig2.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=220,
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          xaxis=dict(showgrid=False, color="#9fc8ff"),
                          yaxis=dict(type="log", showgrid=True, gridcolor="rgba(160,190,255,.15)", color="#9fc8ff"))
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
    else:
        st.caption("No proton series available.")
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='section'><div class='subttl'>KP INDEX</div>", unsafe_allow_html=True)
    if kp_time:
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=kp_time, y=kp_vals, name="Kp"))
        fig3.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=220,
                           paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           xaxis=dict(showgrid=False, color="#9fc8ff"),
                           yaxis=dict(range=[0,9], showgrid=True, gridcolor="rgba(160,190,255,.15)", color="#9fc8ff"))
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
    else:
        st.caption("No Kp series available.")
    st.markdown("</div>", unsafe_allow_html=True)

# Discussion + BOM (discussion reflowed, full-width)
noaa_discussion_html, noaa_discussion_src, noaa_discussion_raw = get_noaa_forecast_text()
bom_aurora_text = get_bom_aurora()
st.markdown(f"""
<div class='section'>
  <div class='subttl'>NOAA Forecast Discussion
    <span style="opacity:.6;font-size:.9em;">({noaa_discussion_src.split('/')[-1] if noaa_discussion_src else 'unavailable'})</span>
  </div>
  <div style='color:#cfe3ff;margin-top:.5rem;'>{noaa_discussion_html}</div>
  <hr style='border-color:rgba(139,233,253,.18)'>
  <div class='subttl'>BOM Aurora</div>
  <div style='white-space:pre-line;color:#cfe3ff;margin-top:.5rem;'>{bom_aurora_text}</div>
</div>
""", unsafe_allow_html=True)

# Badges
past, current = get_noaa_rsg_now_and_past()
st.markdown("<div class='section'>", unsafe_allow_html=True)
cA, cSpacer, cD = st.columns([1.2, 0.4, 1.2])
with cA:
    st.markdown("<div class='subttl'>Latest Observed</div>", unsafe_allow_html=True)
    st.markdown("<div class='badge-col'>", unsafe_allow_html=True)
    st.markdown(f"<div class='neon-badge { 'yellow' if current['r_color']=='yellow' else 'red' if current['r_color']=='red' else ''}'>{current['r']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='neon-badge { 'yellow' if current['s_color']=='yellow' else 'red' if current['s_color']=='red' else ''}'>{current['s']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='neon-badge { 'yellow' if current['g_color']=='yellow' else 'red' if current['g_color']=='red' else ''}'>{current['g']}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
with cD:
    st.markdown("<div class='subttl'>24-Hour Observed Maximums</div>", unsafe_allow_html=True)
    st.markdown("<div class='badge-col'>", unsafe_allow_html=True)
    st.markdown(f"<div class='neon-badge { 'yellow' if past['r_color']=='yellow' else 'red' if past['r_color']=='red' else ''}'>{past['r']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='neon-badge { 'yellow' if past['s_color']=='yellow' else 'red' if past['s_color']=='red' else ''}'>{past['s']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='neon-badge { 'yellow' if past['g_color']=='yellow' else 'red' if past['g_color']=='red' else ''}'>{past['g']}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Next 24h (UTC) only
next24 = get_next24_summary()
wind_kms, bt, bz, f10_7 = get_goes_telemetry()

c1, c2 = st.columns([1.2, 1.2])
with c1:
    st.markdown("<div class='section'><div class='subttl'>GOES / L1 Measurements</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='kv'>
      Solar Wind Speed: {wind_kms} km/sec<br>
      Solar Magnetic Fields: Bt: {bt} nT, Bz: {bz} nT<br>
      Noon 10.7cm Radio Flux: {f10_7} sfu
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown(f"<div class='section'><div class='subttl'>Prediction — Next 24 hours (UTC)</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='kv'>
      Minor Flare (R1–R2): {next24['r12_prob']}%<br>
      Major Flare (R3+): {next24['r3_prob']}%<br>
      Solar Proton Flux (S1+): {next24['s1_prob']}%<br>
      Geomagnetic Activity: {next24['g_bucket']} (Kp≈{next24['kp_max']})
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Quick summary line
summary_text = make_summary(current, next24)
st.markdown(f"<div class='footer'>Summary: {summary_text}</div>", unsafe_allow_html=True)

# PDF (uses RAW discussion text for completeness)
pdf_download_button(
    noaa_discussion_raw=noaa_discussion_raw or noaa_discussion_html,
    past=past,
    current=current,
    next24=next24,
    bom_aurora=bom_aurora_text,
    summary_text=summary_text
)

# Footer (timestamps)
try:
    last_server = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
except Exception:
    last_server = "—"
st.caption(f"Server time: {last_server}  •  Refresh page to update feeds.")



##51585962-2fdd-4cf5-9d9e-74cdd09e3bab##