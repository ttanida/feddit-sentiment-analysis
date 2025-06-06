"""Data models for comments and sentiment analysis."""

from pydantic import BaseModel, Field


class SentimentResult(BaseModel):
    """Sentiment analysis result for a comment."""

    polarity_score: float = Field(
        ..., description="Sentiment polarity score between -1 and 1"
    )
    classification: str = Field(
        ..., description="Sentiment classification: positive or negative"
    )


class CommentBase(BaseModel):
    """Base comment model from Feddit API."""

    id: str = Field(..., description="Unique identifier of the comment")
    username: str = Field(..., description="Username who made the comment")
    text: str = Field(..., description="Content of the comment")
    created_at: int = Field(..., description="Unix timestamp when comment was created")


class CommentWithSentiment(CommentBase):
    """Comment model with sentiment analysis results."""

    sentiment: SentimentResult = Field(..., description="Sentiment analysis results")


class SubfedditInfo(BaseModel):
    """Basic subfeddit information."""

    id: str = Field(..., description="Unique identifier of the subfeddit")
    username: str = Field(..., description="Username who started the subfeddit")
    title: str = Field(..., description="Title/topic of the subfeddit")
    description: str = Field(..., description="Description of the subfeddit")


class SentimentAnalysisResponse(BaseModel):
    """Response model for sentiment analysis endpoint."""

    subfeddit: str = Field(..., description="Name of the subfeddit")
    total_comments: int = Field(..., description="Total number of comments returned")
    comments: list[CommentWithSentiment] = Field(
        ..., description="List of comments with sentiment analysis"
    )
    subfeddit_info: SubfedditInfo | None = Field(
        None, description="Subfeddit information if available"
    )
