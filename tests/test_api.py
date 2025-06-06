"""Integration tests for API endpoints."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from main import app
from src.config import settings


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestAPI:
    """Test suite for API endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "service" in data
        assert "version" in data
        assert data["service"] == settings.api_title
        assert data["version"] == settings.api_version

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get(f"{settings.api_prefix}/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "sentiment-analysis"

    @patch("src.services.sentiment_service.analyze_subfeddit_sentiment")
    def test_analyze_sentiment_success(self, mock_analyze, client):
        """Test successful sentiment analysis."""
        # Mock the service response
        mock_response = {
            "subfeddit": "Dummy Topic 1",
            "total_comments": 2,
            "comments": [
                {
                    "id": "1",
                    "username": "user1",
                    "text": "I love this product!",
                    "created_at": 1640995200,
                    "sentiment": {"polarity_score": 0.8, "classification": "positive"},
                },
                {
                    "id": "2",
                    "username": "user2",
                    "text": "This is terrible!",
                    "created_at": 1641081600,
                    "sentiment": {"polarity_score": -0.7, "classification": "negative"},
                },
            ],
            "subfeddit_info": None,
        }

        mock_analyze.return_value = mock_response

        response = client.get(f"{settings.api_prefix}/subfeddits/Dummy%20Topic%201/sentiment")
        assert response.status_code == 200

        data = response.json()
        assert data["subfeddit"] == "Dummy Topic 1"
        assert data["total_comments"] == 2
        assert len(data["comments"]) == 2

        # Check sentiment data
        assert data["comments"][0]["sentiment"]["classification"] == "positive"
        assert data["comments"][1]["sentiment"]["classification"] == "negative"

    def test_analyze_sentiment_with_parameters(self, client):
        """Test sentiment analysis with query parameters."""
        with patch("src.services.sentiment_service.analyze_subfeddit_sentiment") as mock_analyze:
            mock_analyze.return_value = {
                "subfeddit": "Dummy Topic 1",
                "total_comments": 0,
                "comments": [],
                "subfeddit_info": None,
            }

            response = client.get(
                f"{settings.api_prefix}/subfeddits/Dummy%20Topic%201/sentiment",
                params={
                    "limit": 10,
                    "sort_order": "asc",
                    "start_date": "2022-01-01",
                    "end_date": "2022-12-31",
                },
            )

            assert response.status_code == 200

            # Verify the service was called with correct parameters
            mock_analyze.assert_called_once_with(
                subfeddit_name="Dummy Topic 1",
                limit=10,
                start_date="2022-01-01",
                end_date="2022-12-31",
                sort_order="asc",
            )

    def test_analyze_sentiment_no_sorting(self, client):
        """Test sentiment analysis with no sorting parameter."""
        with patch("src.services.sentiment_service.analyze_subfeddit_sentiment") as mock_analyze:
            mock_response = {
                "subfeddit": "Dummy Topic 1",
                "total_comments": 2,
                "comments": [
                    {
                        "id": "1",
                        "username": "user1",
                        "text": "Great!",
                        "created_at": 1640995200,
                        "sentiment": {"polarity_score": 0.8, "classification": "positive"},
                    },
                    {
                        "id": "2",
                        "username": "user2",
                        "text": "Okay.",
                        "created_at": 1641081600,
                        "sentiment": {"polarity_score": 0.2, "classification": "positive"},
                    },
                ],
                "subfeddit_info": None,
            }

            mock_analyze.return_value = mock_response

            response = client.get(f"{settings.api_prefix}/subfeddits/Dummy%20Topic%201/sentiment")
            assert response.status_code == 200

            data = response.json()
            assert data["subfeddit"] == "Dummy Topic 1"
            assert data["total_comments"] == 2

            # Verify the service was called with None sort_order (default)
            mock_analyze.assert_called_once_with(
                subfeddit_name="Dummy Topic 1",
                limit=25,
                start_date=None,
                end_date=None,
                sort_order=None,
            )

    def test_analyze_sentiment_explicit_none_sorting(self, client):
        """Test sentiment analysis with explicitly passed None sort_order."""
        with patch("src.services.sentiment_service.analyze_subfeddit_sentiment") as mock_analyze:
            mock_analyze.return_value = {
                "subfeddit": "Dummy Topic 1",
                "total_comments": 1,
                "comments": [
                    {
                        "id": "1",
                        "username": "user1",
                        "text": "Test comment",
                        "created_at": 1640995200,
                        "sentiment": {"polarity_score": 0.0, "classification": "neutral"},
                    }
                ],
                "subfeddit_info": None,
            }

            response = client.get(
                f"{settings.api_prefix}/subfeddits/Dummy%20Topic%201/sentiment",
                params={"sort_order": "None"},  # Explicitly pass None as string
            )
            assert response.status_code == 400  # Should be rejected as invalid

    def test_analyze_sentiment_invalid_parameters(self, client):
        """Test sentiment analysis with invalid parameters."""
        # Test invalid limit (too low)
        response = client.get(f"{settings.api_prefix}/subfeddits/Dummy%20Topic%201/sentiment", params={"limit": 0})
        assert response.status_code == 422

        # Test invalid limit (too high)
        response = client.get(f"{settings.api_prefix}/subfeddits/Dummy%20Topic%201/sentiment", params={"limit": 101})
        assert response.status_code == 422

        # Test invalid sort_order
        response = client.get(
            f"{settings.api_prefix}/subfeddits/Dummy%20Topic%201/sentiment", params={"sort_order": "invalid"}
        )
        assert response.status_code == 400

        data = response.json()
        assert "sort_order must be" in data["detail"]

    @patch("src.services.sentiment_service.analyze_subfeddit_sentiment")
    def test_analyze_sentiment_feddit_error(self, mock_analyze, client):
        """Test handling of Feddit API errors."""
        from src.clients import FedditAPIError

        mock_analyze.side_effect = FedditAPIError("Feddit API unavailable")

        response = client.get(f"{settings.api_prefix}/subfeddits/Dummy%20Topic%201/sentiment")
        assert response.status_code == 503

        data = response.json()
        assert "Unable to fetch data from Feddit API" in data["detail"]

    @patch("src.services.sentiment_service.analyze_subfeddit_sentiment")
    def test_analyze_sentiment_validation_error(self, mock_analyze, client):
        """Test handling of validation errors."""
        mock_analyze.side_effect = ValueError("Invalid date format")

        response = client.get(f"{settings.api_prefix}/subfeddits/Dummy%20Topic%201/sentiment")
        assert response.status_code == 400

        data = response.json()
        assert "Invalid parameter" in data["detail"]
