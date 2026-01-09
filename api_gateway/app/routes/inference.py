from fastapi import APIRouter
from api_gateway.app.services.inference import inference_service
from api_gateway.app.routes.dto.inference import (
    NycDurationPredictionResponse,
    NycDurationPredictionRequest,
)

router = APIRouter()


@router.post(
    "/predict",
    response_model=NycDurationPredictionResponse,
    summary="Predict Trip Duration",
    description="Predict the estimated duration of a taxi trip based on pickup time, location, and passenger count.",
)
async def prediction(request: NycDurationPredictionRequest):
    payload = request.dict()
    return inference_service.inference(**payload)
