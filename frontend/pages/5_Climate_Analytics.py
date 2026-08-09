# pages/5_Climate_Analytics.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Climate Analytics", layout="wide")

st.title("📊 Historical Climate Analytics & Trends")
st.markdown("---")

# Constructing historical simulation data
dates = pd.date_range(start="2016-01-01", end="2026-01-01", freq="ME")
rainfall = np.random.normal(loc=150, scale=40, size=len(dates)) + np.sin(np.linspace(0, 50, len(dates))) * 100
rainfall = np.clip(rainfall, 0, None)

analytics_df = pd.DataFrame({"Date": dates, "Precipitation (mm)": rainfall})

col1, col2 = st.columns([1, 3])

with col1:
    st.markdown("### Trend Aggregations")
    metric_choice = st.selectbox("Target Stat", ["Precipitation", "Anomalous Deviations"])
    st.metric("10-Year Historical Mean", f"{rainfall.mean():.2f} mm")
    st.metric("Max Recorded Precip Event", f"{rainfall.max():.2f} mm")

with col2:
    fig = px.line(analytics_df, x="Date", y="Precipitation (mm)", title="10-Year Longitudinal Monsoon Trend")
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)