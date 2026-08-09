# config/__init__.py
"""
Configuration module initialization for the Weather Intelligence and Climate Decision Support Platform.
Exposes the global settings instance to manage directories, paths, and environment variables dynamically.
"""

from config.settings import settings

# Expose the settings instance at the package level for clean imports
__all__ = ["settings"]