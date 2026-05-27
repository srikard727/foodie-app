from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    foursquare_api_key: str | None = None

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

FOURSQUARE_RESTAURANT_CATEGORY_ID = "13065"
FOURSQUARE_RADIUS_METERS = 8047  # 5 miles
FOURSQUARE_SEARCH_LIMIT = 20
FOURSQUARE_DETAIL_LIMIT = 5
FOURSQUARE_TIPS_LIMIT = 5
RECOMMENDATION_LIMIT = 3
