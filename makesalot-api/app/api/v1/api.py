"""
API router that includes all endpoint routers
"""
from fastapi import APIRouter
from app.api.v1.endpoints import technical, predictions, chart

api_router = APIRouter()

api_router.include_router(
    technical.router,
    prefix="/technical",
    tags=["technical"]
)

api_router.include_router(
    predictions.router,
    prefix="/predictions", 
    tags=["predictions"]
)

api_router.include_router(
    chart.router,
    prefix="/chart",
    tags=["chart"]
)