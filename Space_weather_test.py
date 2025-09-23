import os
import base64
import re
from datetime import datetime

import plotly.graph_objects as go
import requests
import streamlit as st
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
        if BOM_ERR:
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

# ========== Data Fetchers ==========
def get_bom_aurora():
    if bom is None:
        return f"Aurora info unavailable. ({BOM_ERR or 'pyspaceweather not installed or API key missing'})"
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

@st.cache_data(ttl=600, show_spinner=False)
def get_noaa_forecast_text():
    urls = [
        "https://services.swpc.noaa.gov/text/discussion.txt",
        "https://services.swpc.noaa.gov/text/forecast-discussion.txt",
        "https://services.swpc.noaa.gov/text/3-day-forecast.txt",
    ]
    for url in urls:
        try:
            txt = fetch_text(url)
            lines = [ln.rstrip() for ln in txt.splitlines()]
            block = []
            for ln in lines:
                if ln.strip():
                    block.append(ln)
                if ln.startswith("III.") or ln.lower().startswith("synopsis"):
                    break
            top = "\n".join(block).strip() if block else txt
            structured = parse_discussion_structured(top)
            return structured, url, top
        except Exception:
            continue
    return {
        "solar_activity": {"summary": "", "forecast": ""},
        "energetic_particle": {"summary": "", "forecast": ""},
        "solar_wind": {"summary": "", "forecast": ""},
        "geospace": {"summary": "", "forecast": ""},
        "_reflowed": "NOAA forecast discussion unavailable."
    }, None, ""

def parse_discussion_structured(raw_txt: str):
    lines = _strip_meta_lines(raw_txt)

    sec_alias = {
        "solar activity": "solar_activity",
        "energetic particle": "energetic_particle",
        "solar wind": "solar_wind",
        "geospace": "geospace",
        "geo space": "geospace",
        "geo-space": "geospace",
    }
    main_header_re = re.compile(r"^(Solar Activity|Energetic Particle|Solar Wind|Geospace|GEO ?Space)\b", re.I)
    sub_header_re  = re.compile(r"^\.(24\s*hr\s*summary|forecast|forcast)\b", re.I)

    data = {
        "solar_activity": {"summary": "", "forecast": ""},
        "energetic_particle": {"summary": "", "forecast": ""},
        "solar_wind": {"summary": "", "forecast": ""},
        "geospace": {"summary": "", "forecast": ""},
    }

    current_sec_key = None
    current_sub = None
    buf = []

def parse_three_day_for_next24(txt: str):
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
        "g_bucket": g_bucket, "kp_max": f"{kpmax:.2f}" if kpmax is not None else "~"
    }

@st.cache_data(ttl=600, show_spinner=False)

def get_next24_summary():
    try:
        txt = fetch_text("https://services.swpc.noaa.gov/text/3-day-forecast.txt")
        return parse_three_day_for_next24(txt)
    except Exception:
        return {"r_bucket":"R0","r12_prob":0,"r3_prob":0,"s_bucket":"S0","s1_prob":0,"g_bucket":"G0","kp_max":"~"}

def get_goes_telemetry():
    wind_kms = "~"; bt = "~"; bz = "~"; f10_7 = "~"
    try:
        mag = fetch_json("https://services.swpc.noaa.gov/json/dscovr/real_time_mag-1-day.json")
        if mag:
            last = mag[-1]; bt = str(last.get("bt", "~")); bz = str(last.get("bz_gsm", "~"))
    except Exception: pass
    try:
        plasma = fetch_json("https://services.swpc.noaa.gov/json/dscovr/real_time_plasma-1-day.json")
        if plasma:
            last = plasma[-1]; wind_kms = str(last.get("speed", "~"))
    except Exception: pass
    try:
        f10 = fetch_json("https://services.swpc.noaa.gov/json/f10.7cm/observed.json")
        if f10:
            f10_7 = str(f10[-1].get("flux", "~"))
    except Exception: pass
    return wind_kms, bt, bz, f10_7

def make_summary(current, next24):
    return (f"Now: {current['r']}/{current['s']}/{current['g']}. "
            f"Next 24 h: {next24['g_bucket']} (Kp~{next24['kp_max']}), "
            f"R1–R2 {next24['r12_prob']}% / R3+ {next24['r3_prob']}%, S1+ {next24['s1_prob']}%.")

def export_management_pdf(
    noaa_discussion_raw,
    past, current, next24,
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

    C_TEAL = (30, 115, 190) if high_contrast else (79, 158, 255)
    C_SLATE = (20, 20, 20) if high_contrast else (40, 47, 56)
    C_GRAY  = (60, 60, 60) if high_contrast else (105, 115, 125)

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
        f"Next 24 h (UTC): {next24['g_bucket']} (Kp~{next24['kp_max']}), R1- R2 {next24['r12_prob']}% / R3+ {next24['r3_prob']}%, S1+ {next24['s1_prob']}%.",
        "BOM Aurora: " + (bom_aurora.splitlines()[0] if bom_aurora else "-"),
    ]:
        pdf.set_x(10); pdf.cell(5, 7, "•")
        pdf.multi_cell(0, 7, b)
    pdf.ln(2)

    # Key Metrics table
    def metrics_row(label, r, s, g, bold=False):
        pdf.set_font("Helvetica", "B" if bold else "", 11)
        fill = 250 if high_contrast else 245
        grid = 120 if high_contrast else 220
        pdf.set_fill_color(fill, fill, fill); pdf.set_draw_color(grid, grid, grid)
        pdf.cell(60, 9, label, border=1, align="L", fill=True)
        pdf.cell(43, 9, r, border=1, align="C", fill=True)
        pdf.cell(43, 9, s, border=1, align="C", fill=True)
        pdf.cell(44, 9, g, border=1, align="C", fill=True)
        pdf.ln(9)

    H("Key Metrics (NOAA R/S/G)", size=16)
    metrics_row("Scope", "R - Radio Blackouts", "S - Radiation Storms", "G - Geomagnetic Storms", bold=True)
    metrics_row("Current",
                f"{current['r']} ({'Active' if current['r']!='R0' else 'No'})",
                f"{current['s']} ({'Active' if current['s']!='S0' else 'No'})",
                f"{current['g']} ({'Active' if current['g']!='G0' else 'No'})")
    metrics_row("Past 24 h",
                f"{past['r']} ({'Active' if past['r']!='R0' else 'No'})",
                f"{past['s']} ({'Active' if past['s']!='S0' else 'No'})",
                f"{past['g']} ({'Active' if past['g']!='G0' else 'No'})")
    metrics_row("Next 24 h",
                f"{next24['r_bucket']} (R1- R2 {next24['r12_prob']}% / R3+ {next24['r3_prob']}%)",
                f"{next24['s_bucket']} (S1+ {next24['s1_prob']}%)",
                f"{next24['g_bucket']} (Kp~{next24['kp_max']})")
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

    # NOAA Discussion full text (sanitized)
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
/* ...rest of your CSS, unchanged... */
</style>""", unsafe_allow_html=True)

# ========== Tabs ==========
tab_overview, tab_charts, tab_forecast, tab_aurora, tab_expert, tab_pdf, tab_help = st.tabs([
    "Overview", "Charts", "Forecasts", "Aurora", "Expert Data", "PDF Export", "Help & Info"
])

# ========== Overview Tab ==========
with tab_overview:
    st.markdown("## Space Weather Dashboard - Overview")
    past, current = get_noaa_rsg_now_and_past()
    next24 = get_next24_summary()
    summary_text = make_summary(current, next24)
    st.markdown(f"**Summary:** {summary_text}")
    st.caption(f"Last updated: {last_updated()}")

    def badge(label, level_key, aria):
        classes = f"neon-badge level-{level_key} pattern-{level_key if level_key in ['low','med','high','veryhigh'] else 'low'}"
        return f"<div class='{classes}' role='img' aria-label='{aria}' title='{aria}'><span>{label}</span></div>"
    st.markdown("<div class='badge-col'>" +
        badge(current['r'], current['lvl'], f"R scale now {current['r']}") +
        badge(current['s'], current['lvl_s'], f"S scale now {current['s']}") +
        badge(current['g'], current['lvl_g'], f"G scale now {current['g']}") +
        "</div>", unsafe_allow_html=True)

# ========== Charts Tab ==========
with tab_charts:
    st.markdown("## Recent Space Weather Trends")
    # Fetch data series
    with st.spinner("Loading charts..."):
        # Solar X-ray
        x_time, x_vals = [], []
        try:
            xr_series = fetch_json("https://services.swpc.noaa.gov/json/goes/primary/xray-flares-latest.json")
            for row in xr_series[-24:]:
                x_time.append(row.get("time_tag")); x_vals.append(clamp_float(row.get("flux", 0)))
        except Exception: pass

        fig = None
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

        # Solar Proton
        p_time, p_vals = [], []
        try:
            pr_series = fetch_json("https://services.swpc.noaa.gov/json/goes/primary/proton-flux-1-day.json")
            for row in pr_series:
                p_time.append(row.get("time_tag")); p_vals.append(clamp_float(row.get("flux", 0)))
        except Exception: pass

        fig2 = None
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

        # KP Index
        kp_time, kp_vals = [], []
        try:
            kp_series = fetch_json("https://services.swpc.noaa.gov/json/planetary_k_index_1m.json")
            tail = kp_series[-1000:] if len(kp_series) > 1000 else kp_series
            for row in tail:
                kp_time.append(row.get("time_tag")); kp_vals.append(clamp_float(row.get("kp_index", 0)))
        except Exception: pass

        fig3 = None
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

# ========== Forecasts Tab ==========
with tab_forecast:
    st.markdown("## NOAA Forecast Discussion")
    structured_disc, noaa_discussion_src, noaa_discussion_raw = get_noaa_forecast_text()
    src_note = noaa_discussion_src.split('/')[-1] if noaa_discussion_src else 'unavailable'

    def _sec_html(title, sec):
        has_any = (sec.get("summary") or sec.get("forecast"))
        if not has_any:
            return ""
        s = []
        s.append(f"<h4 style='margin:.2rem 0 .4rem 0;color:#cfe3ff'>{title}</h4>")
        if sec.get("summary"):
            s.append("<div class='subttl' style='margin-top:.2rem;'>24 hr Summary</div>")
            s.append(f"<div style='color:#cfe3ff;margin:.2rem 0 0.8rem 0;'>{sec['summary'].replace('\\n','<br>')}</div>")
        if sec.get("forecast"):
            s.append("<div class='subttl' style='margin-top:.2rem;'>Forecast</div>")
            s.append(f"<div style='color:#cfe3ff;margin:.2rem 0 0.8rem 0;'>{sec['forecast'].replace('\\n','<br>')}</div>")
        return "\n".join(s)

    sec_html = []
    sec_html.append(_sec_html("Solar Activity", structured_disc["solar_activity"]))
    sec_html.append(_sec_html("Energetic Particle", structured_disc["energetic_particle"]))
    sec_html.append(_sec_html("Solar Wind", structured_disc["solar_wind"]))
    sec_html.append(_sec_html("Geospace", structured_disc["geospace"]))
    content_html = "\n".join([h for h in sec_html if h]) or structured_disc["_reflowed"].replace("\n","<br>")
    st.markdown(f"""
    <div class='section' role='region' aria-label='NOAA Forecast Discussion'>
      <div class='subttl'>NOAA Forecast Discussion
        <span style="opacity:.7;font-size:.9em;">({src_note})</span>
      </div>
      <div style='margin-top:.6rem;'>{content_html}</div>
    </div>
    """, unsafe_allow_html=True)

# ========== Aurora Tab ==========
with tab_aurora:
    st.markdown("## Aurora Outlook (BOM)")
    bom_aurora_text = get_bom_aurora()
    st.markdown(f"<pre>{bom_aurora_text}</pre>", unsafe_allow_html=True)
    st.caption(f"Last updated: {last_updated()}")

# ========== Expert Data Tab ==========
with tab_expert:
    st.markdown("## GOES / L1 Expert Data")
    wind_kms, bt, bz, f10_7 = get_goes_telemetry()
    st.markdown(f"""
    - **Solar Wind Speed:** {wind_kms} km/s
    - **Magnetic Fields:** Bt: {bt} nT, Bz: {bz} nT
    - **10.7cm Radio Flux:** {f10_7} sfu
    """)
    st.caption(f"Last updated: {last_updated()}")

# ========== PDF Export Tab ==========
with tab_pdf:
    st.markdown("## Export Management PDF")
    past, current = get_noaa_rsg_now_and_past()
    next24 = get_next24_summary()
    bom_aurora_text = get_bom_aurora()
    summary_text = make_summary(current, next24)
    structured_disc, noaa_discussion_src, noaa_discussion_raw = get_noaa_forecast_text()
    st.info("You can customize which sections and charts to include in your PDF below.")
    include_xray = st.checkbox("Include Solar X-ray Chart", True)
    include_proton = st.checkbox("Include Proton Chart", True)
    include_kp = st.checkbox("Include Kp Chart", True)
    # Use charts from Charts tab if rendered
    fig_xray = fig if 'fig' in locals() and include_xray else None
    fig_proton = fig2 if 'fig2' in locals() and include_proton else None
    fig_kp = fig3 if 'fig3' in locals() and include_kp else None
    if st.button("Generate PDF"):
        pdf_base64 = export_management_pdf(
            noaa_discussion_raw=noaa_discussion_raw or structured_disc["_reflowed"],
            past=past,
            current=current,
            next24=next24,
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