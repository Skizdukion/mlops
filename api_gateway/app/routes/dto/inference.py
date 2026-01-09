from pydantic import BaseModel, Field, conint, confloat
from datetime import datetime
from uuid import UUID
from api_gateway.app.constant.model_type import ModelType


class NycDurationPredictionRequest(BaseModel):
    """Request model for taxi duration prediction."""

    tpep_pickup_datetime: datetime = Field(..., description="Pickup time (UTC)")
    passenger_count: conint(ge=1, le=12) = Field(..., description="Passenger Count")
    pulocationid: int = Field(..., description="Pickup location id")
    dolocationid: int = Field(..., description="Estimate dropoff location id")
    model_type: ModelType = Field(
        default=ModelType.xgboost, description="Model to use for prediction"
    )


class NycDurationPredictionResponse(BaseModel):
    prediction_id: UUID = Field(..., description="Unique prediction identifier")
    est_duration: confloat(gt=0) = Field(
        ..., description="Estimated trip duration in seconds"
    )
