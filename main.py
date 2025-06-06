"""Main application entry point."""

import logging
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api import router
from src.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Sentiment Analysis API...")

    # Download required NLTK data for TextBlob
    try:
        import nltk

        nltk.download("punkt", quiet=True)
        nltk.download("brown", quiet=True)
        logger.info("NLTK data downloaded successfully")
    except Exception as e:
        logger.warning(f"Failed to download NLTK data: {e}")

    logger.info("Application startup complete")

    # Separate the startup and shutdown processes
    yield

    # Shutdown
    logger.info("Shutting down Sentiment Analysis API...")


# Create FastAPI application
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=settings.api_description,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": settings.api_title,
        "version": settings.api_version,
        "description": settings.api_description,
        "docs": "/docs",
        "health": f"{settings.api_prefix}/health",
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=settings.debug, log_level="info")
