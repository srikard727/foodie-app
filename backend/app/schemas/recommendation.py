from pydantic import BaseModel

class RecommendationRequest(BaseModel):
    message: str