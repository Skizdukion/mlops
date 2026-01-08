from fastapi import APIRouter, status
from pydantic import BaseModel, Field, confloat
from api_gateway.app.storage.repository import SqlLiteDatabase
from api_gateway.app.models.domain import Feedback

router = APIRouter()

class NycDurationFeedbackRequest(BaseModel):
    """Request model for pronunciation assessment."""

    prediction_id: int = Field(..., description="Unique prediction identifier")
    dolocationid: int = Field(..., description="Real dropoff location id")
    duration: confloat(gt=0) = Field(..., description="Real duration")


@router.post("/feedback", status_code=status.HTTP_204_NO_CONTENT)
async def feedback(request: NycDurationFeedbackRequest):
    pass
