"""API routes for sentiment analysis endpoints."""

import logging

from fastapi import APIRouter, HTTPException, Query

from ..clients import FedditAPIError
from ..config import settings
from ..models import SentimentAnalysisResponse
from ..services import sentiment_service

logger = logging.getLogger(__name__)

# Create API router
router = APIRouter(prefix=settings.api_prefix, tags=["sentiment"])


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "sentiment-analysis"}


@router.get(
    "/subfeddits/{subfeddit_name}/sentiment", response_model=SentimentAnalysisResponse
)
async def analyze_subfeddit_sentiment(
    subfeddit_name: str,
    limit: int = Query(
        default=25,
        ge=1,
        le=100,
        description="Maximum number of comments to analyze (1-100)",
    ),
    start_date: str | None = Query(
        default=None,
        description="Filter comments after this date (ISO format: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)",
    ),
    end_date: str | None = Query(
        default=None,
        description="Filter comments before this date (ISO format: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)",
    ),
    sort_order: str | None = Query(
        default=None,
        description="Sort comments by polarity score: 'asc' (most negative first), 'desc' (most positive first), or None (no sorting - chronological order)",
    ),
):
    """
    Analyze sentiment for comments in a specific subfeddit.

    Returns the most recent comments with sentiment analysis including:
    - Polarity score (-1 to 1)
    - Classification (positive or negative)

    Optional features:
    - Filter by date range
    - Sort by sentiment polarity score (None = chronological order, default)
    - Limit number of results

    Sorting behavior:
    - None: Comments in chronological order (no sorting by sentiment)
    - 'desc': Most positive comments first (polarity score 1.0 → -1.0)
    - 'asc': Most negative comments first (polarity score -1.0 → 1.0)
    """
    try:
        logger.info(
            f"Processing sentiment analysis request for subfeddit: {subfeddit_name}"
        )

        # Validate sort_order parameter
        if sort_order is not None and sort_order not in ["asc", "desc"]:
            raise ValueError(
                f"sort_order must be 'asc', 'desc', or None, got: {sort_order}"
            )

        result = await sentiment_service.analyze_subfeddit_sentiment(
            subfeddit_name=subfeddit_name,
            limit=limit,
            start_date=start_date,
            end_date=end_date,
            sort_order=sort_order,
        )

        return result

    except FedditAPIError as e:
        logger.error(f"Feddit API error for {subfeddit_name}: {str(e)}")
        raise HTTPException(
            status_code=503, detail=f"Unable to fetch data from Feddit API: {str(e)}"
        )

    except ValueError as e:
        logger.error(f"Invalid parameter for {subfeddit_name}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid parameter: {str(e)}")

    except Exception as e:
        logger.error(f"Unexpected error analyzing {subfeddit_name}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error occurred while processing request",
        )
