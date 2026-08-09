# pages/4_GIS_Maps.py
import streamlit as st
import numpy as np
import plotly.express as px
import pydeck as pdk
import pandas as pd
from remote_sensing.processor import GeospatialProcessor

st.set_page_config(page_title="GIS Geospatial Analysis", layout="wide")

st.title("🗺️ Remote Sensing & Spatial Index Profiling")
st.markdown("---")

# Sidebar Controls for Dynamic Geographic Targeting
st.sidebar.markdown("### 🌐 Regional Coordinates Filter")
target_region = st.sidebar.selectbox(
    "Focus Geographic Sector",
    ["Central India Agri-Belt", "Kerala Coastal Gateway", "Gangetic Basin", "Northeast Orographic Zone"]
)

# Coordinates mapping for dynamic viewport centering
region_coords = {
    "Central India Agri-Belt": {"lat": 22.9734, "lon": 78.6569, "zoom": 5.5},
    "Kerala Coastal Gateway": {"lat": 10.8505, "lon": 76.2711, "zoom": 6.5},
    "Gangetic Basin": {"lat": 25.5941, "lon": 85.1376, "zoom": 6.0},
    "Northeast Orographic Zone": {"lat": 25.5788, "lon": 91.8933, "zoom": 6.5}
}

coords = region_coords[target_region]

processor = GeospatialProcessor()

col_controls, col_display = st.columns([1, 2])

with col_controls:
    st.markdown("### 📊 Index Selection & Parameters")
    index_choice = st.selectbox("Select Target Index", ["NDVI (Drought & Vegetation)", "NDWI (Flood & Wetness)"])
    render_3d = st.checkbox("Render 3D Volumetric Topography", value=True)
    run_analysis = st.button("🚀 Render Spatial Grid Metrics", use_container_width=True)

with col_display:
    if run_analysis:
        with st.spinner("Reading geospatial raster matrices..."):
            # Process raster data using GeospatialProcessor engine
            raster_data = processor.process_satellite_tiff("data/satellite_temp.tif")
            
            st.write(f"**Bounding Coordinates:** `{raster_data['bounds']}`")
            
            matrix = np.array(raster_data["ndvi_matrix"] if "NDVI" in index_choice else raster_data["ndwi_matrix"])
            mean_val = raster_data["ndvi_mean"] if "NDVI" in index_choice else raster_data["ndwi_mean"]
            
            st.info(f"💡 Calculated Area Mean Index Value: **{mean_val:.4f}**")
            
            # Heatmap Matrix Rendering via Plotly
            fig = px.imshow(
                matrix, 
                color_continuous_scale="YlGn" if "NDVI" in index_choice else "YlGnBu",
                title=f"Geospatial {index_choice} Density Raster"
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Optional 3D Column Map Layer Overlay
            if render_3d:
                st.markdown("### 🏔️ 3D Surface Index Distribution")
                
                # Convert matrix into geospatial point array surrounding the chosen region
                grid_size = matrix.shape[0]
                lats = np.linspace(coords["lat"] - 1.0, coords["lat"] + 1.0, grid_size)
                lons = np.linspace(coords["lon"] - 1.0, coords["lon"] + 1.0, grid_size)
                lon_grid, lat_grid = np.meshgrid(lons, lats)
                
                df_points = pd.DataFrame({
                    'lat': lat_grid.flatten(),
                    'lon': lon_grid.flatten(),
                    'val': np.clip(matrix.flatten() * 100, 1, 100)
                })

                layer = pdk.Layer(
                    "ColumnLayer",
                    data=df_points,
                    get_position=["lon", "lat"],
                    get_elevation="val",
                    elevation_scale=300,
                    radius=8000,
                    get_fill_color=["val * 2", "180", "255 - val * 2", 180],
                    pickable=True,
                    auto_highlight=True,
                )

                view_state = pdk.ViewState(
                    latitude=coords["lat"],
                    longitude=coords["lon"],
                    zoom=coords["zoom"],
                    pitch=45,
                    bearing=15
                )

                st.pydeck_chart(pdk.Deck(
                    layers=[layer],
                    initial_view_state=view_state,
                    tooltip={"text": "Index Value: {val}"}
                ))
    else:
        st.info("👆 Select your targeting parameters and click **'Render Spatial Grid Metrics'** to inspect the satellite matrix.")