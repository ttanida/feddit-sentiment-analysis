"""Unit tests for SentimentService."""

from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

from src.clients import FedditAPIError
from src.models import CommentBase, CommentWithSentiment, SentimentResult
from src.services.sentiment_service import SentimentService


class TestSentimentService:
    """Test suite for SentimentService class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = SentimentService()

    def test_validate_parameters_default_limit(self):
        """Test parameter validation with default limit."""
        result = self.service._SentimentService__validate_parameters(None, "desc")
        assert result == 25  # default_comment_limit

    def test_validate_parameters_custom_limit(self):
        """Test parameter validation with custom limit."""
        result = self.service._SentimentService__validate_parameters(50, "desc")
        assert result == 50

    def test_validate_parameters_max_limit_exceeded(self):
        """Test parameter validation when limit exceeds maximum."""
        result = self.service._SentimentService__validate_parameters(150, "desc")
        assert result == 100  # max_comment_limit

    def test_validate_parameters_invalid_limit(self):
        """Test parameter validation with invalid limit."""
        with pytest.raises(ValueError, match="Limit must be greater than 0"):
            self.service._SentimentService__validate_parameters(0, "desc")

    def test_validate_parameters_invalid_sort_order(self):
        """Test parameter validation with invalid sort order."""
        with pytest.raises(
            ValueError, match="sort_order must be 'asc', 'desc', or None"
        ):
            self.service._SentimentService__validate_parameters(25, "invalid")

    def test_validate_parameters_none_sort_order(self):
        """Test parameter validation with None sort order."""
        result = self.service._SentimentService__validate_parameters(25, None)
        assert result == 25  # Should not raise an error

    def test_parse_date_parameters_valid_dates(self):
        """Test parsing valid date parameters."""
        start_date, end_date = self.service._parse_date_parameters(
            "2022-01-01", "2022-12-31"
        )

        assert isinstance(start_date, datetime)
        assert isinstance(end_date, datetime)
        assert start_date.year == 2022
        assert end_date.year == 2022

    def test_parse_date_parameters_none_dates(self):
        """Test parsing None date parameters."""
        start_date, end_date = self.service._parse_date_parameters(None, None)

        assert start_date is None
        assert end_date is None

    def test_parse_date_parameters_invalid_format(self):
        """Test parsing invalid date format."""
        with pytest.raises(ValueError, match="Invalid date format"):
            self.service._parse_date_parameters("invalid-date", None)

    def test_sort_comments_desc(self):
        """Test sorting comments in descending order."""
        comments = [
            CommentWithSentiment(
                id="1",
                username="user1",
                text="test",
                created_at=1640995200,
                sentiment=SentimentResult(
                    polarity_score=0.2, classification="positive"
                ),
            ),
            CommentWithSentiment(
                id="2",
                username="user2",
                text="test",
                created_at=1640995200,
                sentiment=SentimentResult(
                    polarity_score=0.8, classification="positive"
                ),
            ),
            CommentWithSentiment(
                id="3",
                username="user3",
                text="test",
                created_at=1640995200,
                sentiment=SentimentResult(
                    polarity_score=-0.5, classification="negative"
                ),
            ),
        ]

        result = self.service._sort_comments(comments, "desc")

        assert len(result) == 3
        assert result[0].sentiment.polarity_score == 0.8
        assert result[1].sentiment.polarity_score == 0.2
        assert result[2].sentiment.polarity_score == -0.5

    def test_sort_comments_asc(self):
        """Test sorting comments in ascending order."""
        comments = [
            CommentWithSentiment(
                id="1",
                username="user1",
                text="test",
                created_at=1640995200,
                sentiment=SentimentResult(
                    polarity_score=0.2, classification="positive"
                ),
            ),
            CommentWithSentiment(
                id="2",
                username="user2",
                text="test",
                created_at=1640995200,
                sentiment=SentimentResult(
                    polarity_score=-0.5, classification="negative"
                ),
            ),
        ]

        result = self.service._sort_comments(comments, "asc")

        assert len(result) == 2
        assert result[0].sentiment.polarity_score == -0.5
        assert result[1].sentiment.polarity_score == 0.2

    @pytest.mark.asyncio
    async def test_fetch_and_analyze_comments_success(self):
        """Test successful fetching and analyzing of comments."""
        mock_comments = [
            CommentBase(
                id="1", username="user1", text="Great product!", created_at=1640995200
            )
        ]

        with patch.object(
            self.service.feddit_client, "get_comments", new_callable=AsyncMock
        ) as mock_get_comments, patch.object(
            self.service.sentiment_analyzer, "analyze_text"
        ) as mock_analyze:
            mock_get_comments.return_value = mock_comments
            mock_analyze.return_value = SentimentResult(
                polarity_score=0.8, classification="positive"
            )

            result = await self.service._fetch_and_analyze_comments(
                "test_subfeddit", 25
            )

            assert len(result) == 1
            assert result[0].id == "1"
            assert result[0].sentiment.polarity_score == 0.8
            assert result[0].sentiment.classification == "positive"

            mock_get_comments.assert_called_once_with(
                subfeddit_name="test_subfeddit", limit=25
            )
            mock_analyze.assert_called_once_with("Great product!")

    @pytest.mark.asyncio
    async def test_fetch_and_analyze_comments_empty_result(self):
        """Test fetching comments with empty result."""
        with patch.object(
            self.service.feddit_client, "get_comments", new_callable=AsyncMock
        ) as mock_get_comments:
            mock_get_comments.return_value = []

            result = await self.service._fetch_and_analyze_comments(
                "test_subfeddit", 25
            )

            assert len(result) == 0

    @pytest.mark.asyncio
    async def test_fetch_and_analyze_comments_api_error(self):
        """Test handling of Feddit API error."""
        with patch.object(
            self.service.feddit_client, "get_comments", new_callable=AsyncMock
        ) as mock_get_comments:
            mock_get_comments.side_effect = FedditAPIError("API unavailable")

            with pytest.raises(FedditAPIError):
                await self.service._fetch_and_analyze_comments("test_subfeddit", 25)

    @pytest.mark.asyncio
    async def test_analyze_subfeddit_sentiment_success(self):
        """Test successful sentiment analysis of subfeddit."""
        mock_comments = [
            CommentBase(id="1", username="user1", text="Great!", created_at=1640995200)
        ]

        with patch.object(
            self.service.feddit_client, "get_comments", new_callable=AsyncMock
        ) as mock_get_comments, patch.object(
            self.service.feddit_client, "get_subfeddit_info", new_callable=AsyncMock
        ) as mock_get_info, patch.object(
            self.service.sentiment_analyzer, "analyze_text"
        ) as mock_analyze:
            mock_get_comments.return_value = mock_comments
            mock_get_info.return_value = None
            mock_analyze.return_value = SentimentResult(
                polarity_score=0.8, classification="positive"
            )

            result = await self.service.analyze_subfeddit_sentiment("test_subfeddit")

            assert result.subfeddit == "test_subfeddit"
            assert result.total_comments == 1
            assert len(result.comments) == 1
            assert result.comments[0].sentiment.polarity_score == 0.8

    @pytest.mark.asyncio
    async def test_analyze_subfeddit_sentiment_no_comments(self):
        """Test sentiment analysis with no comments found."""
        with patch.object(
            self.service.feddit_client, "get_comments", new_callable=AsyncMock
        ) as mock_get_comments:
            mock_get_comments.return_value = []

            result = await self.service.analyze_subfeddit_sentiment("empty_subfeddit")

            assert result.subfeddit == "empty_subfeddit"
            assert result.total_comments == 0
            assert len(result.comments) == 0

    @pytest.mark.asyncio
    async def test_analyze_subfeddit_sentiment_with_date_filtering(self):
        """Test sentiment analysis with date parameters."""
        with patch.object(
            self.service, "_fetch_and_analyze_comments", new_callable=AsyncMock
        ) as mock_fetch, patch.object(
            self.service.feddit_client, "get_subfeddit_info", new_callable=AsyncMock
        ) as mock_get_info:
            mock_fetch.return_value = []
            mock_get_info.return_value = None

            result = await self.service.analyze_subfeddit_sentiment(
                "test_subfeddit", start_date="2022-01-01", end_date="2022-12-31"
            )

            assert result.subfeddit == "test_subfeddit"
            mock_fetch.assert_called_once_with(
                "test_subfeddit",
                25,
                datetime(2022, 1, 1, 0, 0),
                datetime(2022, 12, 31, 0, 0),
            )

    @pytest.mark.asyncio
    async def test_analyze_subfeddit_sentiment_no_sorting(self):
        """Test sentiment analysis with no sorting (None sort_order)."""
        mock_comments = [
            CommentBase(id="1", username="user1", text="Great!", created_at=1640995200),
            CommentBase(id="2", username="user2", text="Okay.", created_at=1641081600),
        ]

        with patch.object(
            self.service.feddit_client, "get_comments", new_callable=AsyncMock
        ) as mock_get_comments, patch.object(
            self.service.feddit_client, "get_subfeddit_info", new_callable=AsyncMock
        ) as mock_get_info, patch.object(
            self.service.sentiment_analyzer, "analyze_text"
        ) as mock_analyze:
            mock_get_comments.return_value = mock_comments
            mock_get_info.return_value = None
            # Different sentiment scores to test order preservation
            mock_analyze.side_effect = [
                SentimentResult(polarity_score=0.8, classification="positive"),
                SentimentResult(polarity_score=0.2, classification="positive"),
            ]

            result = await self.service.analyze_subfeddit_sentiment(
                "test_subfeddit", sort_order=None
            )

            assert result.subfeddit == "test_subfeddit"
            assert result.total_comments == 2
            assert len(result.comments) == 2
            # Should maintain chronological order (no sorting by sentiment)
            assert result.comments[0].id == "1"  # First chronologically
            assert result.comments[1].id == "2"  # Second chronologically
            assert result.comments[0].sentiment.polarity_score == 0.8
            assert result.comments[1].sentiment.polarity_score == 0.2

    @pytest.mark.asyncio
    async def test_fetch_with_date_aware_pagination_skip_batches(self):
        """Test that smart pagination skips batches before start_date."""
        # Mock comments: first batch all before start_date, second batch has matches
        batch1_comments = [
            CommentBase(
                id="1", username="user1", text="Old comment", created_at=1609459200
            ),  # 2021-01-01
        ]
        batch2_comments = [
            CommentBase(
                id="2", username="user2", text="New comment", created_at=1640995200
            ),  # 2022-01-01
        ]

        with patch.object(
            self.service.feddit_client, "get_comments", new_callable=AsyncMock
        ) as mock_get_comments, patch.object(
            self.service.sentiment_analyzer, "analyze_text"
        ) as mock_analyze:
            # Return different batches on different calls
            mock_get_comments.side_effect = [
                batch1_comments,
                batch2_comments,
                [],
            ]  # Empty third call
            mock_analyze.return_value = SentimentResult(
                polarity_score=0.5, classification="positive"
            )

            start_date = datetime(2021, 12, 1)  # Should skip first batch
            result = await self.service._fetch_with_date_aware_pagination(
                "test", 25, start_date, None
            )

            assert len(result) == 1
            assert (
                result[0].id == "2"
            )  # Only the second batch comment should be included

            # Should have called get_comments twice (skip first batch, process second)
            assert (
                mock_get_comments.call_count == 3
            )  # batch1, batch2, empty batch to stop

    @pytest.mark.asyncio
    async def test_fetch_with_date_aware_pagination_stop_at_end_date(self):
        """Test that smart pagination stops when reaching end_date."""
        # Mock comments: first batch within range, second batch after end_date
        batch1_comments = [
            CommentBase(
                id="1", username="user1", text="Within range", created_at=1640995200
            ),  # 2022-01-01
        ]
        batch2_comments = [
            CommentBase(
                id="2", username="user2", text="After end date", created_at=1672531200
            ),  # 2023-01-01
        ]

        with patch.object(
            self.service.feddit_client, "get_comments", new_callable=AsyncMock
        ) as mock_get_comments, patch.object(
            self.service.sentiment_analyzer, "analyze_text"
        ) as mock_analyze:
            mock_get_comments.side_effect = [batch1_comments, batch2_comments]
            mock_analyze.return_value = SentimentResult(
                polarity_score=0.5, classification="positive"
            )

            end_date = datetime(2022, 6, 1)  # Should stop before second batch
            result = await self.service._fetch_with_date_aware_pagination(
                "test", 25, None, end_date
            )

            assert len(result) == 1
            assert (
                result[0].id == "1"
            )  # Only the first batch comment should be included

            # Should have called get_comments twice but stopped when hitting end_date
            assert mock_get_comments.call_count == 2

    @pytest.mark.asyncio
    async def test_fetch_with_date_aware_pagination_respects_limit(self):
        """Test that smart pagination respects the final limit parameter."""
        # Mock many comments to test limit
        batch_comments = [
            CommentBase(
                id=f"{i}",
                username=f"user{i}",
                text=f"Comment {i}",
                created_at=1640995200 + i,
            )
            for i in range(50)  # 50 comments in batch
        ]

        with patch.object(
            self.service.feddit_client, "get_comments", new_callable=AsyncMock
        ) as mock_get_comments, patch.object(
            self.service.sentiment_analyzer, "analyze_text"
        ) as mock_analyze:
            mock_get_comments.return_value = batch_comments
            mock_analyze.return_value = SentimentResult(
                polarity_score=0.5, classification="positive"
            )

            # Request only 10 comments
            result = await self.service._fetch_with_date_aware_pagination(
                "test", 10, None, None
            )

            assert len(result) == 10  # Should be limited to 10 despite 50 available

            # Should have stopped early since we found enough comments
            assert mock_get_comments.call_count == 1
