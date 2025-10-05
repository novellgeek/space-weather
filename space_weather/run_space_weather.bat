@echo off
REM Launch the Space Weather Streamlit app on port 8503
cd C:\Users\HP\Scripts\Operation\space_weather
start /min streamlit run Space_weather_latest.py --server.port 8503