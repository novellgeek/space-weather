import streamlit as st
import requests
from datetime import datetime, timedelta
import base64
from fpdf import FPDF

# --- BOM SpaceWeather Wrapper ---
from pyspaceweather import SpaceWeather  # Ensure pyspaceweather.py is in your directory!

BOM_API_KEY = "51585962-2fdd-4cf5-9d9e-74cdd09e3bab"
bom = SpaceWeather(BOM_API_KEY)

st.set_page_config(page_title="Space Weather (NOAA + BOM Aurora)", layout="wide")

# ========== BOM Aurora Alerts ==========
def get_bom_aurora():
    aurora_lines = []
    try:
        outlooks = bom.get_aurora_outlook()
        watches = bom.get_aurora_watch()
        alerts = bom.get_aurora_alert()
        if outlooks:
            o = outlooks[0]
            aurora_lines.append(f"**Aurora Outlook:** {o.comments}")
        if watches:
            w = watches[0]
            aurora_lines.append(f"**Aurora Watch:** {w.comments}")
        if alerts:
            a = alerts[0]
            aurora_lines.append(f"**Aurora Alert:** {a.description}")
        return "\n".join(aurora_lines) if aurora_lines else "No aurora alerts/outlooks."
    except Exception as e:
        return f"Aurora info unavailable. ({e})"

# ========== NOAA R/S/G BLOCKS ==========
def get_noaa_rsg():
    kp_url = "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json"
    xray_url = "https://services.swpc.noaa.gov/json/goes/primary/xray-flares-latest.json"
    proton_url = "https://services.swpc.noaa.gov/json/goes/primary/proton-flux-1-day.json"
    def r_scale(x):
        if x > 1e-4: return ("R2", "yellow")
        if x > 1e-5: return ("R1", "green")
        return ("R0", "green")
    def s_scale(p):
        if p > 10: return ("S2", "yellow")
        if p > 1: return ("S1", "green")
        return ("S0", "green")
    def g_scale(k):
        if k >= 7: return ("G3", "red")
        if k >= 5: return ("G2", "yellow")
        if k >= 4: return ("G1", "green")
        return ("G0", "green")
    try:
        kp = requests.get(kp_url, timeout=10).json()
        k_now = kp[-1]['kp_index']
        k_past = max([v['kp_index'] for v in kp[-24:]])
    except Exception:
        k_now, k_past = 0, 0
    try:
        xr = requests.get(xray_url, timeout=10).json()
        x_now = xr[-1]['flux']
        x_past = max([v['flux'] for v in xr[-24:]])
    except Exception:
        x_now, x_past = 0, 0
    try:
        pr = requests.get(proton_url, timeout=10).json()
        p_now = pr[-1]['flux']
        p_past = max([v['flux'] for v in pr[-24:]])
    except Exception:
        p_now, p_past = 0, 0
    r_now, r_now_col = r_scale(x_now)
    r_past, r_past_col = r_scale(x_past)
    s_now, s_now_col = s_scale(p_now)
    s_past, s_past_col = s_scale(p_past)
    g_now, g_now_col = g_scale(k_now)
    g_past, g_past_col = g_scale(k_past)
    forecast_dates = [(datetime.utcnow() + timedelta(days=i)).strftime("%d-%b") for i in range(3)]
    forecast = []
    for _ in range(3):
        forecast.append({
            "r": r_now, "r_txt": "Radio blackouts", "r_status": "See forecast", "r_color": r_now_col,
            "s": s_now, "s_txt": "Radiation storms", "s_status": "See forecast", "s_color": s_now_col,
            "g": g_now, "g_txt": "Geomagnetic storms", "g_status": "See forecast", "g_color": g_now_col,
        })
    current = {
        "r": r_now, "r_txt": "Radio blackouts", "r_status": "No" if r_now == "R0" else "Active", "r_color": r_now_col,
        "s": s_now, "s_txt": "Radiation storms", "s_status": "No" if s_now == "S0" else "Active", "s_color": s_now_col,
        "g": g_now, "g_txt": "Geomagnetic storms", "g_status": "No" if g_now == "G0" else "Active", "g_color": g_now_col
    }
    past = {
        "r": r_past, "r_txt": "Radio blackouts", "r_status": "No" if r_past == "R0" else "Active", "r_color": r_past_col,
        "s": s_past, "s_txt": "Radiation storms", "s_status": "No" if s_past == "S0" else "Active", "s_color": s_past_col,
        "g": g_past, "g_txt": "Geomagnetic storms", "g_status": "No" if g_past == "G0" else "Active", "g_color": g_past_col
    }
    return past, current, forecast, forecast_dates

# ========== NOAA Forecast Discussion ==========
def get_noaa_forecast_text():
    url = "https://services.swpc.noaa.gov/text/forecast-discussion.txt"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        lines = [line.strip() for line in resp.text.splitlines() if line.strip()]
        # Only return today's discussion (first block)
        if lines:
            first_block = []
            for line in lines:
                first_block.append(line)
                if line.startswith("III.") or line.lower().startswith("synopsis"):
                    break
            return "\n".join(first_block)
        return "NOAA forecast discussion unavailable."
    except Exception as e:
        return f"NOAA forecast discussion unavailable. ({e})"

# ========== PDF Export ==========
def pdf_download_button(noaa_discussion, past, current, forecast, forecast_dates, bom_aurora):
    def safe_ascii(text):
        return text.replace("–", "-").encode("ascii", "replace").decode("ascii")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt="Space Weather Dashboard (NOAA + BOM Aurora)", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=safe_ascii(noaa_discussion))
    pdf.multi_cell(0, 10, txt=safe_ascii(bom_aurora))
    pdf.cell(100, 10, txt="Past 24h (NOAA):", ln=True)
    pdf.cell(100, 10, txt=safe_ascii(f"{past['r']} {past['r_txt']}: {past['r_status']}"), ln=True)
    pdf.cell(100, 10, txt=safe_ascii(f"{past['s']} {past['s_txt']}: {past['s_status']}"), ln=True)
    pdf.cell(100, 10, txt=safe_ascii(f"{past['g']} {past['g_txt']}: {past['g_status']}"), ln=True)
    pdf.cell(100, 10, txt="Current (NOAA):", ln=True)
    pdf.cell(100, 10, txt=safe_ascii(f"{current['r']} {current['r_txt']}: {current['r_status']}"), ln=True)
    pdf.cell(100, 10, txt=safe_ascii(f"{current['s']} {current['s_txt']}: {current['s_status']}"), ln=True)
    pdf.cell(100, 10, txt=safe_ascii(f"{current['g']} {current['g_txt']}: {current['g_status']}"), ln=True)
    pdf.cell(100, 10, txt="Forecast (NOAA):", ln=True)
    for i in range(3):
        f = forecast[i]
        pdf.cell(100, 10, txt=safe_ascii(f"{forecast_dates[i]}: {f['r']} {f['r_txt']}, {f['s']} {f['s_txt']}, {f['g']} {f['g_txt']}"), ln=True)
    pdf.multi_cell(0, 10, txt="For explanation of the R, S and G scales, and a description of risk and impact, see the NOAA Space Weather Scales. BOM aurora alerts supplement NOAA data.")
    pdf_output = pdf.output(dest='S').encode('latin1')
    b64 = base64.b64encode(pdf_output).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="space_weather.pdf">Export PDF</a>'
    st.markdown(href, unsafe_allow_html=True)

# ========== UI Styling ==========
st.markdown("""
    <style>
    .main {background-color: #181a20;}
    h1, h2, h3, .stTextInput label {color:#00ffe7 !important; font-family: 'Orbitron', Arial, sans-serif;}
    .block-container {padding: 2em 1em;}
    .rsg-grid {display: flex; flex-wrap: wrap; gap: 0.5em; justify-content: center;}
    .rsg-block {
        min-width: 160px;
        max-width: 160px;
        min-height: 130px;
        background: #23a455;
        color: white;
        border-radius: 10px;
        padding: 1em;
        margin: 0.2em;
        text-align: center;
        box-shadow: 0 0 8px #00ffe733;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .rsg-title {font-size: 1.2em; font-weight: bold;}
    .rsg-desc {font-size: 1em;}
    .rsg-green {background: #23a455;}
    .rsg-yellow {background: #e3b90c;}
    .rsg-red {background: #c11c1c;}
    </style>
""", unsafe_allow_html=True)

# ========== Dashboard ==========
st.markdown("<h1 style='text-align: center; color: #00ffe7; text-shadow: 0 0 18px #00ffe733;'>Space Weather (NOAA + BOM Aurora)</h1>", unsafe_allow_html=True)

noaa_discussion = get_noaa_forecast_text()
bom_aurora = get_bom_aurora()

st.markdown(f"""
    <div style='background:#232638;border:2px solid #00ffe7;border-radius:10px;padding:1.5em 2em;margin-bottom:2em;box-shadow:0 0 14px #00ffe711;'>
        <span style='color:#00ffe7;font-family:Orbitron,Arial;font-size:1.13em;'>NOAA Forecast Discussion</span>
        <div style='margin-top:1em;color:#d7d8e0;font-size:1.1em;white-space:pre-line'>{noaa_discussion}</div>
        <hr>
        <span style='color:#00ffe7;font-family:Orbitron,Arial;font-size:1.13em;'>BOM Aurora Alerts/Outlook</span>
        <div style='margin-top:1em;font-size:1.1em;white-space:pre-line'>{bom_aurora}</div>
    </div>
""", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center; color: #4f66de;'>Space Weather Conditions (NOAA R/S/G)</h2>", unsafe_allow_html=True)

past, current, forecast, forecast_dates = get_noaa_rsg()

st.markdown("<div class='rsg-grid'>", unsafe_allow_html=True)
st.markdown(f"""
    <div class='rsg-block rsg-{past['r_color']}'>
        <div class='rsg-title'>Past 24 hr</div>
        <div><b>{past['r']}</b></div>
        <div class='rsg-desc'>{past['r_txt']}<br>{past['r_status']}</div>
        <div style='margin-top:0.3em;'><b>{past['s']}</b></div>
        <div class='rsg-desc'>{past['s_txt']}<br>{past['s_status']}</div>
        <div style='margin-top:0.3em;'><b>{past['g']}</b></div>
        <div class='rsg-desc'>{past['g_txt']}<br>{past['g_status']}</div>
    </div>
""", unsafe_allow_html=True)
st.markdown(f"""
    <div class='rsg-block rsg-{current['r_color']}'>
        <div class='rsg-title'>Current</div>
        <div><b>{current['r']}</b></div>
        <div class='rsg-desc'>{current['r_txt']}<br>{current['r_status']}</div>
        <div style='margin-top:0.3em;'><b>{current['s']}</b></div>
        <div class='rsg-desc'>{current['s_txt']}<br>{current['s_status']}</div>
        <div style='margin-top:0.3em;'><b>{current['g']}</b></div>
        <div class='rsg-desc'>{current['g_txt']}<br>{current['g_status']}</div>
    </div>
""", unsafe_allow_html=True)
for i in range(3):
    f = forecast[i]
    st.markdown(f"""
        <div class='rsg-block rsg-{f['r_color']}'>
            <div class='rsg-title'>{forecast_dates[i]} Forecast</div>
            <div><b>{f['r']}</b></div>
            <div class='rsg-desc'>{f['r_txt']}<br>{f['r_status']}</div>
            <div style='margin-top:0.3em;'><b>{f['s']}</b></div>
            <div class='rsg-desc'>{f['s_txt']}<br>{f['s_status']}</div>
            <div style='margin-top:0.3em;'><b>{f['g']}</b></div>
            <div class='rsg-desc'>{f['g_txt']}<br>{f['g_status']}</div>
        </div>
    """, unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
    <br>
    <div style='font-size:1em;color:#222;background:#e2e8f0;border-radius:9px;padding:1em 2em;max-width:800px;margin:auto;'>
        For explanation of the <b>R, S and G</b> scales, and a description of risk and impact, see the
        <a href='https://www.swpc.noaa.gov/noaa-scales-explanation' target='_blank'>NOAA Space Weather Scales</a>.
        <br>
        Colour scaling: <span style='color:green;font-weight:bold;'>Green</span>: Low to Medium,
        <span style='color:orange;font-weight:bold;'>Yellow</span>: High, <span style='color:red;font-weight:bold;'>Red</span>: Very high.
        <br>
        All times UTC. NOAA data live, BOM aurora alerts supplement. Dashboard refreshed on load.
    </div>
""", unsafe_allow_html=True)

pdf_download_button(noaa_discussion, past, current, forecast, forecast_dates, bom_aurora)