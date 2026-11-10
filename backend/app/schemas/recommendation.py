from pydantic import BaseModel, Field, field_validator

class RecommendationRequest(BaseModel):
    message: str = Field(min_length=2, max_length=500)
    latitude: float = Field(ge=-90, le=90, allow_inf_nan=False)
    longitude: float = Field(ge=-180, le=180, allow_inf_nan=False)

    @field_validator("message")
    @classmethod
    def nonempty_message(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 2:
            raise ValueError("Please describe what you would like to eat.")
        return value
