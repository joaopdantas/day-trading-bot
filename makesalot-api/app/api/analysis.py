from fastapi import APIRouter, Query
from app.services.analysis_service import compute_moving_average
from app.models.schemas import IndicatorRequest, IndicatorResponse

router = APIRouter()

@router.post("/moving-average", response_model=IndicatorResponse)
def moving_average(req: IndicatorRequest):
    result = compute_moving_average(req)
    return IndicatorResponse(result=result)
