# remote_sensing/earthengine_api.py
"""
Google Earth Engine (GEE) API Integration Module.
Handles connections and data extraction from GEE collections with fallback simulation logic.
"""

import os
from typing import Dict, Any

# Import globally so VS Code linter always recognizes the "ee" module
try:
    import ee
    HAS_EE = True
except ImportError:
    HAS_EE = False


class EarthEngineAPI:
    """Manages connections to Google Earth Engine platforms."""
    
    def __init__(self):
        self.initialized = False
        if HAS_EE:
            try:
                # Checks for environment variables before initialization
                if "EARTHENGINE_SERVICE_ACCOUNT" in os.environ:
                    ee.Initialize()
                    self.initialized = True
            except Exception:
                self.initialized = False

    def fetch_sentinel_image(self, bbox: list, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Retrieves Sentinel metadata for regional bounding box coordinates.
        Uses a robust simulation fallback if GEE environment is uninitialized.
        """
        # 1. Safe simulation boundary check
        if not self.initialized or not HAS_EE:
            return {
                "status": "Simulation Mode",
                "message": "Earth Engine API is running in fallback mode.",
                "dataset": "COPERNICUS/S2_SR"
            }
        
        # 2. Live GEE Connection (Clean, warning-free)
        try:
            # Define region geometry
            aoi = ee.Geometry.Rectangle(bbox)
            collection = (
                ee.ImageCollection("COPERNICUS/S2_SR")
                .filterBounds(aoi)
                .filterDate(start_date, end_date)
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
            )
            
            image = collection.median()
            return {
                "status": "Connected",
                "image_id": image.id().getInfo(),
                "bands": ["B2", "B3", "B4", "B8"]
            }
        except Exception as e:
            return {
                "status": "Error",
                "message": f"GEE Live Ingestion Failed: {str(e)}",
                "dataset": "COPERNICUS/S2_SR"
            }