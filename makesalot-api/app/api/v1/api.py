"""
Base API router that includes all endpoint routers
"""
from fastapi import APIRouter
from app.api.v1.endpoints import (
    technical,
    predictions,
    user,
    feedback,
    watchlist
)

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
    user.router,
    prefix="/users",
    tags=["users"]
)

api_router.include_router(
    feedback.router,
    prefix="/feedback",
    tags=["feedback"]
)

api_router.include_router(
    watchlist.router,
    prefix="/watchlist",
    tags=["watchlist"]
)
