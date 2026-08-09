# pages/6_Explainable_AI.py
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Explainable AI", layout="wide")

st.title("🧩 Explainable AI (XAI) Dashboard")
st.markdown("---")

st.info("Feature importance values indicate how much each input variable influences rainfall predictions.")

# Mock SHAP value summary
shap_data = {
    "Atmospheric Feature": [
        "Relative Humidity 2m", 
        "Sea Level Pressure (MSL)", 
        "Mean Temperature", 
        "Wind Speed Max", 
        "Day of Year"
    ],
    "Mean Absolute SHAP Value (Impact)": [0.384, 0.291, 0.184, 0.098, 0.043]
}

df_shap = pd.DataFrame(shap_data).sort_values(by="Mean Absolute SHAP Value (Impact)", ascending=True)

fig = px.bar(
    df_shap, 
    x="Mean Absolute SHAP Value (Impact)", 
    y="Atmospheric Feature", 
    orientation="h",
    title="SHAP Global Feature Importance Values",
    color="Mean Absolute SHAP Value (Impact)",
    color_continuous_scale="Purples"
)
fig.update_layout(template="plotly_dark")

st.plotly_chart(fig, use_container_width=True)
st.markdown("""
**Key Finding:** Relative Humidity in the lower boundary layers ($2\text{m}$) has the strongest statistical correlation with monsoon precipitation trends.
""")