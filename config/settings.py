import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from config import settings

class ProjectSettings(BaseSettings):
    # App Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_RAW_DIR: Path = BASE_DIR / "data" / "raw"
    DATA_PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"
    MODEL_CHECKPOINTS_DIR: Path = BASE_DIR / "data" / "models_checkpoint"
    VECTOR_DB_DIR: Path = BASE_DIR / "data" / "vector_db"
    
    # API & Agent settings
    OPEN_METEO_API_URL: str = "https://api.open-meteo.com/v1/forecast"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    LLM_MODEL_NAME: str = "llama3"
    
    # Ensure directories exist upon configurations initialization
    model_config = SettingsConfigDict(env_file=str(Path(__file__).resolve().parent.parent / ".env"), extra="ignore")

    def create_directories(self):
        for path in [self.DATA_RAW_DIR, self.DATA_PROCESSED_DIR, self.MODEL_CHECKPOINTS_DIR, self.VECTOR_DB_DIR]:
            path.mkdir(parents=True, exist_ok=True)

# Instantiate settings
settings = ProjectSettings()
settings.create_directories()