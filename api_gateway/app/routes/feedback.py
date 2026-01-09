from fastapi import APIRouter, status
from api_gateway.app.services.feedback import feedback_service
from api_gateway.app.routes.dto.feedback import (
    NycDurationFeedbackRequest,
)

router = APIRouter()


@router.post(
    "/feedback",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Submit Trip Feedback",
    description="Submit actual trip feedback including real duration to monitor model performance.",
)
async def feedback(request: NycDurationFeedbackRequest):
    feedback_service.feedback(request)
    return
