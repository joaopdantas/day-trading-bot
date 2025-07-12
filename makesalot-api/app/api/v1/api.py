"""
Complete API Router for MakesALot Trading API
Includes all endpoints needed for the extension
"""
from fastapi import APIRouter

# Import all endpoint routers
from app.api.v1.endpoints.analyze import router as analyze_router
from app.api.v1.endpoints.quote import router as quote_router
from app.api.v1.endpoints.utils import router as utils_router
from app.api.v1.endpoints.strategies_router import router as strategies_router
from app.api.v1.endpoints.chart_data import router as chart_router

# Import existing endpoints if they exist
try:
    from app.api.v1.endpoints import technical
    TECHNICAL_AVAILABLE = True
except (ImportError, AttributeError):
    TECHNICAL_AVAILABLE = False

try:
    from app.api.v1.endpoints import predictions
    PREDICTIONS_AVAILABLE = True
except (ImportError, AttributeError):
    PREDICTIONS_AVAILABLE = False

try:
    from app.api.v1.endpoints import chart
    CHART_AVAILABLE = True
except (ImportError, AttributeError):
    CHART_AVAILABLE = False

# Create main API router
api_router = APIRouter()

# Include all new endpoint routers
api_router.include_router(
    analyze_router,
    prefix="",
    tags=["analysis"]
)

api_router.include_router(
    quote_router,
    prefix="",
    tags=["quotes"]
)

api_router.include_router(
    utils_router,
    prefix="/utils",
    tags=["utilities"]
)

api_router.include_router(
    strategies_router,
    prefix="/strategies",
    tags=["strategies"]
)

api_router.include_router(
    chart_router,
    prefix="/chart",
    tags=["chart"]
)

# Include existing endpoints if available
if TECHNICAL_AVAILABLE:
    api_router.include_router(
        technical.router,
        prefix="/technical",
        tags=["technical-legacy"]
    )

if PREDICTIONS_AVAILABLE:
    api_router.include_router(
        predictions.router,
        prefix="/predictions", 
        tags=["predictions-legacy"]
    )

if CHART_AVAILABLE:
    api_router.include_router(
        chart.router,
        prefix="/chart-legacy",
        tags=["chart-legacy"]
    )