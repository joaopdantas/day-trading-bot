"""
MakesALot Trading API - Main Application Entry Point
"""

import logging
import os
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exception_handlers import http_exception_handler

from app.core.config import settings
from app.api.v1.api import api_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version="2.0.0",
    description="Advanced Trading API with Technical Analysis and ML Predictions",
    openapi_url="/api/v1/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://finance.yahoo.com",
        "https://tradingview.com", 
        "https://www.investing.com",
        "https://www.marketwatch.com",
        "https://www.bloomberg.com",
        "https://www.cnbc.com",
        "chrome-extension://*",
        "moz-extension://*",
        "http://localhost:*",
        "https://localhost:*"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)

# Global exception handler
@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    return await http_exception_handler(request, exc)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global Exception: {type(exc).__name__} - {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "service": "MakesALot Trading API",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
        "environment": os.getenv("DEBUG", "false") == "true" and "development" or "production"
    }

# API stats endpoint
@app.get("/stats")
async def api_stats():
    """Basic API statistics"""
    return {
        "service": "MakesALot Trading API",
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "api_v1": settings.API_V1_STR,
            "docs": "/docs",
            "openapi": "/api/v1/openapi.json"
        },
        "features": [
            "Technical Analysis",
            "ML Trading Strategies", 
            "Real-time Quotes",
            "Chart Data",
            "Symbol Validation",
            "Trading Predictions"
        ],
        "timestamp": datetime.now().isoformat()
    }

# Root endpoint
@app.get("/")
async def root():
    """API root endpoint with basic information"""
    return {
        "message": "Welcome to MakesALot Trading API",
        "version": "2.0.0",
        "documentation": "/docs",
        "health": "/health", 
        "api": settings.API_V1_STR,
        "timestamp": datetime.now().isoformat()
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    """Application startup tasks"""
    logger.info("🚀 MakesALot Trading API starting up...")
    logger.info(f"📊 API URL: {settings.API_V1_STR}")
    logger.info(f"🌐 Host: {settings.HOST}:{settings.PORT}")
    logger.info("✅ Startup complete")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown tasks"""
    logger.info("👋 MakesALot Trading API shutting down...")
    logger.info("✅ Shutdown complete")

if __name__ == "__main__":
    import uvicorn
    
    # Development server configuration
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=os.getenv("DEBUG", "false") == "true",
        log_level="info"
    )