from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, HttpUrl, Field, conint, confloat
from typing import Optional, Dict, Any
from datetime import datetime

router = APIRouter()


class NycDurationFeedbackRequest(BaseModel):
    """Request model for pronunciation assessment."""
    prediction_id: str = Field(..., description="Unique prediction identifier")
    dolocationid: int = Field(..., description="Real dropoff location id")
    duration: confloat(gt=0) = Field(..., description="Real duration")


@router.post("/feedback", status_code=status.HTTP_204_NO_CONTENT)
async def feedback(request: NycDurationFeedbackRequest):
    # store feedback
    return
