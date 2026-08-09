# pages/1_Live_Weather.py
import streamlit as st
import httpx
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Live Weather Ingestion", layout="wide")

st.title("📡 Real-Time Meteorological Ingestion Hub")
st.markdown("---")

st.sidebar.markdown("### Location Settings")
latitude = st.sidebar.number_input("Latitude", value=22.9734, format="%.4f")
longitude = st.sidebar.number_input("Longitude", value=78.6569, format="%.4f")

if st.sidebar.button("Fetch Live Ingestion Data"):
    with st.spinner("Connecting to ingestion API..."):
        try:
            # Query backend open-meteo proxy
            response = httpx.get(
                "http://localhost:8000/api/v1/weather/current", 
                params={"latitude": latitude, "longitude": longitude},
                timeout=5.0
            )
            if response.status_code == 200:
                data = response.json()
                st.session_state["live_data"] = data
                st.success("Ingestion successful.")
            else:
                st.error("Error connecting to live weather station service.")
        except Exception as e:
            st.error(f"Network error: {str(e)}")

# Display Data
if "live_data" in st.session_state:
    weather = st.session_state["live_data"]
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Temperature", f"{weather.get('temperature', 'N/A')} °C")
    col2.metric("Relative Humidity", f"{weather.get('relative_humidity', 'N/A')} %")
    col3.metric("Wind Speed", f"{weather.get('wind_speed', 'N/A')} km/h")
    col4.metric("Cloud Cover", f"{weather.get('cloud_cover', 'N/A')} %")
    
    st.markdown("### Regional Climate Risk Advisory")
    if weather.get("relative_humidity", 0) > 80:
        st.warning("⚠️ High Atmospheric Moisture Index: Favorable conditions for convective precipitation initiation.")
    else:
        st.info("ℹ️ Stable regional atmosphere detected. Normal background parameters.")
else:
    st.info("Please trigger the ingestion request from the sidebar panel.")