# remote_sensing/__init__.py
"""
Remote Sensing and Geospatial package initialization.
Exposes index calculation tools and Google Earth Engine services.
"""

from remote_sensing.earthengine_api import EarthEngineAPI
from remote_sensing.processor import GeospatialProcessor

__all__ = [
    "EarthEngineAPI",
    "GeospatialProcessor"
]