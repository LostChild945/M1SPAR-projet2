from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field


class RecommendItem(BaseModel):
    product_id: str
    score: float = Field(ge=0.0, le=1.0)
    source: Literal["als", "bridge", "popular"]
    category: str


class RecommendResponse(BaseModel):
    user_id: str
    segment: str
    recommendations: list[RecommendItem]
    latency_ms: int
    cache_hit: bool


class SimilarItem(BaseModel):
    product_id: str
    score: float = Field(ge=0.0, le=1.0)
    category: str


class SimilarResponse(BaseModel):
    product_id: str
    similar_products: list[SimilarItem]
    latency_ms: int


class FeedbackRequest(BaseModel):
    user_id: str
    product_id: str
    interaction_type: Literal["click", "purchase", "skip"]
    ab_variant: Literal["control", "treatment"]
    timestamp: Optional[datetime] = None


class ABResultsResponse(BaseModel):
    experiment_id: str
    start_date: str
    n_days: int
    control_ctr: float
    treatment_ctr: float
    lift_pct: float
    p_value: float
    significant: bool
    n_users_control: int
    n_users_treatment: int


class Error(BaseModel):
    detail: str
