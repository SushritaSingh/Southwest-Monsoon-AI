# pages/2_Forecast_Models.py
import streamlit as st
import httpx

st.set_page_config(page_title="Predictive Forecasting Models", layout="wide")

st.title("🔮 Time-Series & Predictive Forecasting Portal")
st.markdown("---")

st.sidebar.markdown("### Forecasting Input Metas")
engine_type = st.sidebar.selectbox("Prediction Engine Class", ["Ensemble Machine Learning", "Deep Learning Recurrent (LSTM/GRU)"])
temp = st.sidebar.slider("Mean Temperature (°C)", 10.0, 45.0, 28.0)
humidity = st.sidebar.slider("Relative Humidity (%)", 20, 100, 80)
wind = st.sidebar.slider("Wind Speed (km/h)", 0.0, 100.0, 15.0)
pressure = st.sidebar.slider("Sea Level Pressure (hPa)", 950.0, 1050.0, 1008.0)
month = st.sidebar.slider("Target Month", 1, 12, 6)

if st.button("Execute Predictive Forecast Run"):
    payload = {
        "temperature_2m_mean": temp,
        "relative_humidity_2m_mean": humidity,
        "wind_speed_10m_max": wind,
        "pressure_msl_mean": pressure,
        "month": month
    }
    
    with st.spinner("Processing statistical model outputs..."):
        try:
            endpoint = "regression" if engine_type == "Ensemble Machine Learning" else "regression" # Maps to same model endpoint for demo
            res = httpx.post(f"http://localhost:8000/api/v1/predict/{endpoint}", json=payload, timeout=5.0)
            if res.status_code == 200:
                data = res.json()
                
                col1, col2 = st.columns(2)
                col1.metric("Predicted Precipitation", f"{data['predicted_precipitation']:.2f} mm")
                col2.metric("Accuracy Level (R²)", f"{data['confidence_r2'] * 100:.1f}%")
                
                st.success(f"Forecast executed successfully using the `{data['model_name']}` engine.")
            else:
                st.error("Unable to execute forecast run. Verify API status.")
        except Exception as e:
            st.error(f"Inference Timeout: {str(e)}")