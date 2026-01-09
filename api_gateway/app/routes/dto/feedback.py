from pydantic import BaseModel, Field, confloat
from uuid import UUID


class NycDurationFeedbackRequest(BaseModel):
    """Request model for providing feedback on prediction accuracy."""

    prediction_id: UUID = Field(..., description="Unique prediction identifier")
    dolocationid: int = Field(..., description="Real dropoff location id")
    duration: confloat(gt=0) = Field(..., description="Real duration")
