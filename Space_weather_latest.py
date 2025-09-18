import os
import base64
import re
from datetime import datetime

import plotly.graph_objects as go
import requests
import streamlit as st
from fpdf import FPDF
from streamlit.components.v1 import html

# ==========================
# Setup
# ==========================
st.set_page_config(page_title="Space Weather (NOAA + BOM Aurora)", layout="wide")
UA = {"User-Agent": "SpaceWeatherDashboard/3.2 (+streamlit)"}

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
# Sidebar: Auto-refresh + Accessibility + BOM status
# ==========================
with st.sidebar:
    st.markdown("### Settings")
    refresh_min = st.slider(
        "Auto-refresh (minutes)", min_value=0, max_value=30, value=10,
        help="Set to 0 to disable automatic refresh."
    )
    if refresh_min and refresh_min > 0:
        interval_ms = int(refresh_min * 60 * 1000)
        html(f"<script>setTimeout(function(){{ window.location.reload(); }}, {interval_ms});</script>", height=0)

    st.markdown("### Accessibility")
    high_contrast = st.toggle("High-contrast mode", value=True)
    font_scale = st.slider("Font scale", 1.0, 1.6, 1.2, 0.05, help="Increase overall text size")
    label_style = st.selectbox("Badge label style", ["Text + Color (default)", "Text-only"])

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

# ==========================
# PDF sanitizer (Option A)
# ==========================
def _pdf_sanitize(s: str) -> str:
    if s is None:
        return ""
    repl = {
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u2212": "-",  # minus
        "\u2018": "'", "\u2019": "'", "\u201A": ",",
        "\u201C": '"', "\u201D": '"', "\u201E": '"',
        "\u2009": " ", "\u200A": " ", "\u200B": "",  # thin/zero spaces
        "\u00A0": " ",  # nbsp
        "\u2026": "...",  # ellipsis
        "≈": "~"
    }
    out = "".join(repl.get(ch, ch) for ch in s)
    return out.encode("latin-1", "replace").decode("latin-1")

# ==========================
# SWPC discussion parsing → sections & paragraphs
# ==========================
def _strip_meta_lines(txt: str) -> list[str]:
    # remove :Product, :Issued, and comment lines
    return [ln.rstrip() for ln in txt.splitlines() if not (ln.startswith(":") or ln.startswith("#"))]

def _join_lines_to_paragraphs(lines: list[str]) -> str:
    paras, buf = [], []
    for ln in lines:
        if not ln.strip():
            if buf:
                paras.append(" ".join(buf).strip()); buf = []
        else:
            buf.append(ln.strip())
    if buf:
        paras.append(" ".join(buf).strip())
    return "\n\n".join(paras)

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

    def flush():
        nonlocal buf, current_sec_key, current_sub
        if current_sec_key and current_sub and buf:
            text = _join_lines_to_paragraphs(buf).strip()
            if data[current_sec_key][current_sub]:
                data[current_sec_key][current_sub] += "\n\n" + text
            else:
                data[current_sec_key][current_sub] = text
        buf = []

    for ln in lines:
        if not ln.strip():
            buf.append("")
            continue

        mh = main_header_re.match(ln)
        if mh:
            flush()
            name_norm = mh.group(1).lower()
            current_sec_key = sec_alias.get(name_norm, None)
            current_sub = None
            buf = []
            continue

        sh = sub_header_re.match(ln)
        if sh:
            flush()
            tag = sh.group(1).lower().replace(" ", "")
            if tag.startswith("24hr"):
                current_sub = "summary"
            else:
                current_sub = "forecast"
            tail = ln[sh.end():].strip(".:- ").strip()
            if tail:
                buf.append(tail)
            continue

        if current_sec_key:
            if current_sub is None:
                current_sub = "summary"
            buf.append(ln)

    flush()
    data["_reflowed"] = _join_lines_to_paragraphs(lines)
    return data

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
        if outlooks: lines.append(f"Aurora Outlook: {getattr(outlooks[0],'comments','(no text)')}")
        if watches:  lines.append(f"Aurora Watch: {getattr(watches[0],'comments','(no text)')}")
        if alerts:   lines.append(f"Aurora Alert: {getattr(alerts[0],'description','(no text)')}")
        return "\n".join(lines) if lines else "No aurora alerts/outlooks."
    except Exception as e:
        return f"Aurora info unavailable. ({e})"

# ==========================
# Data: NOAA current & past
# ==========================
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
        pr = fetch_json("https://services.swpc.noaa.gov/json/goes/primary/proton-flux-1-day.json")
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

# ==========================
# Data: NOAA discussion (structured)
# ==========================
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

# ==========================
# Data: NOAA 3-day text → take only NEXT 24H
# ==========================
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

# ==========================
# Extra telemetry
# ==========================
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

# ==========================
# Tiny summary
# ==========================
def make_summary(current, next24):
    return (f"Now: {current['r']}/{current['s']}/{current['g']}. "
            f"Next 24 h: {next24['g_bucket']} (Kp~{next24['kp_max']}), "
            f"R1–R2 {next24['r12_prob']}% / R3+ {next24['r3_prob']}%, S1+ {next24['s1_prob']}%.")

# ==========================
# Management-grade PDF exporter (sanitized, charts optional)
# ==========================
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
            self.cell(0, 8, _pdf_sanitize("Space Weather - Executive Brief"), align="L")

        def footer(self):
            self.set_y(-15)
            self.set_draw_color(200, 200, 200); self.line(10, self.get_y(), 200, self.get_y())
            self.set_y(-12); self.set_text_color(*C_GRAY); self.set_font("Helvetica", "", 10)
            ts = datetime.utcnow().strftime("Generated %Y-%m-%d %H:%M UTC")
            self.cell(0, 8, _pdf_sanitize(f"{ts}  ·  Page {self.page_no()}"), align="R")

    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=18)

    def H(title, size=18):
        pdf.set_text_color(*C_TEAL); pdf.set_font("Helvetica", "B", size); pdf.cell(0, 10, _pdf_sanitize(title), ln=1)
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
        pdf.set_x(10); pdf.cell(5, 7, _pdf_sanitize("•"))
        pdf.multi_cell(0, 7, _pdf_sanitize(b))
    pdf.ln(2)

    # Key Metrics table
    def metrics_row(label, r, s, g, bold=False):
        pdf.set_font("Helvetica", "B" if bold else "", 11)
        fill = 250 if high_contrast else 245
        grid = 120 if high_contrast else 220
        pdf.set_fill_color(fill, fill, fill); pdf.set_draw_color(grid, grid, grid)
        pdf.cell(60, 9, _pdf_sanitize(label), border=1, align="L", fill=True)
        pdf.cell(43, 9, _pdf_sanitize(r), border=1, align="C", fill=True)
        pdf.cell(43, 9, _pdf_sanitize(s), border=1, align="C", fill=True)
        pdf.cell(44, 9, _pdf_sanitize(g), border=1, align="C", fill=True)
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
    pdf.multi_cell(0, 7, _pdf_sanitize(summary_text))

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
        pdf.multi_cell(0, 5, _pdf_sanitize("Note: Charts embedded using Plotly + Kaleido. If missing, install kaleido to enable chart export."))

    # NOAA Discussion full text (sanitized)
    if noaa_discussion_raw:
        pdf.add_page(); H("NOAA Discussion (Full Text Excerpt)", size=16)
        pdf.set_font("Helvetica", "", 12)
        for para in (noaa_discussion_raw or "").split("\n\n"):
            pdf.multi_cell(0, 7, _pdf_sanitize(para.strip())); pdf.ln(1)

    # Aurora summary
    if bom_aurora:
        pdf.add_page(); H("BOM Aurora (Summary)", size=16)
        pdf.set_font("Helvetica", "", 12)
        pdf.multi_cell(0, 7, _pdf_sanitize(bom_aurora))

    try:
        pdf_bytes = pdf.output(dest="S").encode("latin-1")
    except Exception:
        pdf_bytes = pdf.output(dest="S").encode("latin-1", errors="replace")

    b64 = base64.b64encode(pdf_bytes).decode()
    st.markdown(
        f'<a href="data:application/pdf;base64,{b64}" download="{_pdf_sanitize(fname)}">📄 Download Executive PDF</a>',
        unsafe_allow_html=True
    )

# ==========================
# UI Styles (with accessibility variables)
# ==========================
# label-only mode removes color; high-contrast boosts contrast; font_scale applied to :root
st.markdown(f"""
<style>
:root {{
  --scale: {font_scale};
  --neon: {"#00ffff" if high_contrast else "#8be9fd"};
  --fg: {"#ffffff" if high_contrast else "#dbe7ff"};
  --bg: {"#0a0a0a" if high_contrast else "#0d1419"};
  --card: {"#0f0f0f" if high_contrast else "#111a21"};
  --border: {"#ffffff90" if high_contrast else "rgba(139,233,253,.25)"};
  --grid: {"#cccccc55" if high_contrast else "rgba(160,190,255,.15)"};
}}
html, body, .main, .block-container {{ font-size: calc(16px * var(--scale)); }}
.main {{ background: radial-gradient(1200px 600px at 30% -10%, rgba(71,131,255,{0.18 if high_contrast else 0.08}), transparent) var(--bg); }}
h1,h2,h3 {{ font-family: 'Orbitron', Arial, sans-serif; letter-spacing:.06em; }}
h1 {{ color: var(--neon); text-shadow: 0 0 18px rgba(139,233,253,.25); }}
.section {{ background: var(--card); border:1px solid var(--border); border-radius:14px;
  box-shadow: 0 0 24px rgba(0,0,0,.4), inset 0 0 80px rgba(139,233,253,.06);
  padding: 18px 20px; margin-bottom: 18px; }}
.subttl {{ color:#a8c7ff; font-weight:800; letter-spacing:.05em; }}
.kv {{ color: var(--fg); font-family: ui-monospace, Menlo, Consolas, "Courier New", monospace; }}
.badge-col {{ display:flex; gap:12px; flex-wrap:wrap; }}
.neon-badge {{
  width:84px; height:84px; border-radius:14px;
  display:flex; align-items:center; justify-content:center;
  border:2px solid var(--border); font-weight:900; font-size:26px; color:#000;
  background:#e6e6e6; position:relative; overflow:hidden;
}}
/* Patterns for non-color cues */
.pattern-low::before, .pattern-med::before, .pattern-high::before, .pattern-veryhigh::before {{
  content:""; position:absolute; inset:0; opacity:{ "0.25" if label_style.startswith("Text +") else "0.10" };
}}
.pattern-low::before {{
  background: repeating-linear-gradient(45deg, #0000 0 6px, #0003 6px 10px);
}}
.pattern-med::before {{
  background: radial-gradient(#0003 1px, transparent 1px);
  background-size: 10px 10px;
}}
.pattern-high::before {{
  background: repeating-linear-gradient(135deg, #0003 0 4px, #0000 4px 8px);
}}
.pattern-veryhigh::before {{
  background: repeating-linear-gradient(90deg, #0003 0 3px, #0000 3px 6px);
}}
/* Colors (optional; toggle off if Text-only) */
{"".join([
  ".level-low{background:#B7F0B1;color:#0b3d0b;}",
  ".level-med{background:#FFE8A3;color:#5a3c00;}",
  ".level-high{background:#FFC1C1;color:#5a0000;}",
  ".level-veryhigh{background:#FF8A8A;color:#4a0000;}"
]) if label_style.startswith("Text +") else ".level-low,.level-med,.level-high,.level-veryhigh{background:#e6e6e6;color:#111;}"}
.sr-only {{ position:absolute; left:-10000px; width:1px; height:1px; overflow:hidden; }}
.footer {{ color:#8aa; font-size: calc(14px * var(--scale)); }}
</style>
""", unsafe_allow_html=True)

# ==========================
# Build dashboard
# ==========================
st.markdown("<h1 style='text-align:left;'>SPACE WEATHER</h1>", unsafe_allow_html=True)

# Charts
col1, col2, col3 = st.columns([1.1, 1.1, 1])

x_time, x_vals = [], []
p_time, p_vals = [], []
kp_time, kp_vals = [], []

fig = fig2 = fig3 = None  # define upfront

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
    st.markdown("<div class='section' role='region' aria-label='Solar X-ray Flux'><div class='subttl'>SOLAR X-RAY FLUX</div>", unsafe_allow_html=True)
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
    st.markdown("<div class='section' role='region' aria-label='Solar Proton Flux'><div class='subttl'>SOLAR PROTON FLUX</div>", unsafe_allow_html=True)
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
    st.markdown("<div class='section' role='region' aria-label='Kp Index'><div class='subttl'>KP INDEX</div>", unsafe_allow_html=True)
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

# Discussion + BOM (structured headings & paragraphs)
structured_disc, noaa_discussion_src, noaa_discussion_raw = get_noaa_forecast_text()

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

src_note = noaa_discussion_src.split('/')[-1] if noaa_discussion_src else 'unavailable'
sec_html = []
sec_html.append(_sec_html("Solar Activity", structured_disc["solar_activity"]))
sec_html.append(_sec_html("Energetic Particle", structured_disc["energetic_particle"]))
sec_html.append(_sec_html("Solar Wind", structured_disc["solar_wind"]))
sec_html.append(_sec_html("Geospace", structured_disc["geospace"]))
content_html = "\n".join([h for h in sec_html if h]) or structured_disc["_reflowed"].replace("\n","<br>")

bom_aurora_text = get_bom_aurora()
st.markdown(f"""
<div class='section' role='region' aria-label='NOAA Forecast Discussion'>
  <div class='subttl'>NOAA Forecast Discussion
    <span style="opacity:.7;font-size:.9em;">({src_note})</span>
  </div>
  <div style='margin-top:.6rem;'>{content_html}</div>
  <hr style='border-color:rgba(139,233,253,.18)'>
  <div class='subttl'>BOM Aurora</div>
  <div style='white-space:pre-line;color:#cfe3ff;margin-top:.5rem;'>{bom_aurora_text}</div>
</div>
""", unsafe_allow_html=True)

# Badges (Latest & 24h Max) — color-blind safe patterns + text labels
def badge(label, level_key, aria):
    # level_key one of: low, med, high, veryhigh (for R/S/G we mapped above)
    classes = f"neon-badge level-{level_key} pattern-{level_key if level_key in ['low','med','high','veryhigh'] else 'low'}"
    return f"<div class='{classes}' role='img' aria-label='{aria}' title='{aria}'><span>{label}</span></div>"

past, current = get_noaa_rsg_now_and_past()
st.markdown("<div class='section' role='region' aria-label='Observed Conditions'>", unsafe_allow_html=True)
cA, cSpacer, cD = st.columns([1.2, 0.2, 1.2])
with cA:
    st.markdown("<div class='subttl'>Latest Observed</div>", unsafe_allow_html=True)
    st.markdown("<div class='badge-col'>", unsafe_allow_html=True)
    st.markdown(badge(current['r'], current['lvl'], f"R scale now {current['r']} - {current['r_txt']} ({current['r_status']})"), unsafe_allow_html=True)
    st.markdown(badge(current['s'], current['lvl_s'], f"S scale now {current['s']} - {current['s_txt']} ({current['s_status']})"), unsafe_allow_html=True)
    st.markdown(badge(current['g'], current['lvl_g'], f"G scale now {current['g']} - {current['g_txt']} ({current['g_status']})"), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
with cD:
    st.markdown("<div class='subttl'>24-Hour Observed Maximums</div>", unsafe_allow_html=True)
    st.markdown("<div class='badge-col'>", unsafe_allow_html=True)
    st.markdown(badge(past['r'], past['lvl'], f"R scale past 24 hours maximum {past['r']} - {past['r_txt']} ({past['r_status']})"), unsafe_allow_html=True)
    st.markdown(badge(past['s'], past['lvl_s'], f"S scale past 24 hours maximum {past['s']} - {past['s_txt']} ({past['s_status']})"), unsafe_allow_html=True)
    st.markdown(badge(past['g'], past['lvl_g'], f"G scale past 24 hours maximum {past['g']} - {past['g_txt']} ({past['g_status']})"), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Next 24h (UTC) only
next24 = get_next24_summary()
wind_kms, bt, bz, f10_7 = get_goes_telemetry()

c1, c2 = st.columns([1.2, 1.2])
with c1:
    st.markdown("<div class='section' role='region' aria-label='GOES and L1 Measurements'><div class='subttl'>GOES / L1 Measurements</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='kv'>
      Solar Wind Speed: {wind_kms} km/sec<br>
      Solar Magnetic Fields: Bt: {bt} nT, Bz: {bz} nT<br>
      Noon 10.7cm Radio Flux: {f10_7} sfu
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown(f"<div class='section' role='region' aria-label='Prediction Next 24 hours'><div class='subttl'>Prediction — Next 24 hours (UTC)</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='kv'>
      Minor Flare (R1–R2): {next24['r12_prob']}%<br>
      Major Flare (R3+): {next24['r3_prob']}%<br>
      Solar Proton Flux (S1+): {next24['s1_prob']}%<br>
      Geomagnetic Activity: {next24['g_bucket']} (Kp~{next24['kp_max']})
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Quick summary line (plain symbols)
summary_text = make_summary(current, next24)
st.markdown(f"<div class='footer' role='note'>Summary: {summary_text}</div>", unsafe_allow_html=True)

# ======= Executive PDF (sanitized, with optional charts) =======
export_management_pdf(
    noaa_discussion_raw=noaa_discussion_raw or structured_disc["_reflowed"],
    past=past,
    current=current,
    next24=next24,
    bom_aurora=bom_aurora_text,
    summary_text=summary_text,
    fig_xray=fig, fig_proton=fig2, fig_kp=fig3,
    fname="space_weather_management.pdf"
)

# Footer (timestamps)
try:
    last_server = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
except Exception:
    last_server = "-"
st.caption(f"Server time: {last_server}  •  Refresh page to update feeds.")






##51585962-2fdd-4cf5-9d9e-74cdd09e3bab##

##windows powershell: $env:BOM_API_KEY="51585962-2fdd-4cf5-9d9e-74cdd09e3bab"##