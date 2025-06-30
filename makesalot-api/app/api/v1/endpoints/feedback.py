"""
User feedback and analytics endpoints
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
from datetime import datetime

from app.core.config import settings
from app.schemas.feedback import (
    FeedbackCreate,
    FeedbackResponse,
    AnalyticsResponse
)
from app.services.feedback import FeedbackService
from app.api.deps import get_current_user

router = APIRouter()


@router.post("/submit", response_model=FeedbackResponse)
async def submit_feedback(
    feedback: FeedbackCreate,
    current_user=Depends(get_current_user)
):
    """
    Submit user feedback
    """
    try:
        feedback_service = FeedbackService()
        result = await feedback_service.create_feedback(
            user_id=current_user.id,
            feedback=feedback
        )
        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error submitting feedback: {str(e)}"
        )


@router.get("/analytics", response_model=AnalyticsResponse)
async def get_analytics(
    days: Optional[int] = 30,
    current_user=Depends(get_current_user)
):
    """
    Get usage analytics and feedback statistics
    """
    try:
        feedback_service = FeedbackService()
        analytics = await feedback_service.get_analytics(days=days)
        return analytics

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting analytics: {str(e)}"
        )
