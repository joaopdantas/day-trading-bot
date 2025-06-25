from fastapi import APIRouter
from app.services.prediction_service import predict_trend
from app.models.schemas import PredictionRequest, PredictionResponse

router = APIRouter()

@router.post("/trend", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    prediction = predict_trend(req)
    return PredictionResponse(prediction=prediction)
