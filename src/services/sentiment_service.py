"""Service layer for sentiment analysis operations."""

import logging
from datetime import datetime

from dateutil.parser import parse as parse_date

from ..clients import FedditAPIError, feddit_client
from ..config import settings
from ..models import CommentBase, CommentWithSentiment, SentimentAnalysisResponse
from ..sentiment import sentiment_analyzer

logger = logging.getLogger(__name__)


class SentimentService:
    """Service for handling sentiment analysis requests."""

    def __init__(self):
        """Initialize the sentiment service."""
        self.feddit_client = feddit_client
        self.sentiment_analyzer = sentiment_analyzer

    def __validate_parameters(self, limit: int | None, sort_order: str | None) -> int:
        """
        Validate and process input parameters.

        Args:
            limit: Maximum number of comments to analyze
            sort_order: Sort order ('asc', 'desc', or None)

        Returns:
            Validated limit value

        Raises:
            ValueError: If invalid parameters provided
        """
        # Validate and set default limit
        if limit is None:
            validated_limit = settings.default_comment_limit
        elif limit > settings.max_comment_limit:
            validated_limit = settings.max_comment_limit
        elif limit <= 0:
            raise ValueError("Limit must be greater than 0")
        else:
            validated_limit = limit

        # Validate sort order
        if sort_order is not None and sort_order not in ["asc", "desc"]:
            raise ValueError("sort_order must be 'asc', 'desc', or None")

        return validated_limit

    def _filter_comments_by_date(
        self,
        comments: list[CommentWithSentiment],
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[CommentWithSentiment]:
        """
        Filter comments by date range.

        Args:
            comments: List of comments to filter
            start_date: Filter comments after this date
            end_date: Filter comments before this date

        Returns:
            Filtered list of comments
        """
        if not start_date and not end_date:
            return comments

        filtered_comments = []

        for comment in comments:
            comment_date = datetime.fromtimestamp(comment.created_at)

            # Check if comment is within date range
            if start_date and comment_date < start_date:
                continue
            if end_date and comment_date > end_date:
                continue

            filtered_comments.append(comment)

        return filtered_comments

    def _sort_comments(
        self, comments: list[CommentWithSentiment], sort_order: str = "desc"
    ) -> list[CommentWithSentiment]:
        """
        Sort comments by specified criteria.

        Args:
            comments: List of comments to sort
            sort_order: Sort order ('asc' or 'desc')

        Returns:
            Sorted list of comments
        """
        reverse = sort_order.lower() == "desc"

        return sorted(
            comments, key=lambda x: x.sentiment.polarity_score, reverse=reverse
        )

    def _parse_date_parameters(
        self, start_date: str | None, end_date: str | None
    ) -> tuple[datetime | None, datetime | None]:
        """
        Parse date string parameters into datetime objects.

        Args:
            start_date: Start date string in ISO format
            end_date: End date string in ISO format

        Returns:
            Tuple of parsed start and end dates

        Raises:
            ValueError: If date format is invalid
        """
        parsed_start_date = None
        parsed_end_date = None

        try:
            if start_date:
                parsed_start_date = parse_date(start_date)
            if end_date:
                parsed_end_date = parse_date(end_date)
        except Exception as e:
            raise ValueError(f"Invalid date format: {str(e)}")

        return parsed_start_date, parsed_end_date

    async def _fetch_and_analyze_comments(
        self,
        subfeddit_name: str,
        validated_limit: int,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[CommentWithSentiment]:
        """
        Fetch comments from Feddit API with smart date-aware pagination and perform sentiment analysis.

        Args:
            subfeddit_name: Name of the subfeddit
            validated_limit: Maximum number of comments to return after filtering
            start_date: Filter comments after this date (for smart pagination)
            end_date: Filter comments before this date (for smart pagination)

        Returns:
            List of comments with sentiment analysis

        Raises:
            FedditAPIError: If failed to fetch comments from Feddit
        """
        # If no date filtering is needed, just fetch the comments and perform sentiment analysis
        if not start_date and not end_date:
            try:
                base_comments = await self.feddit_client.get_comments(
                    subfeddit_name=subfeddit_name, limit=validated_limit
                )
            except FedditAPIError as e:
                logger.error(f"Failed to fetch comments for {subfeddit_name}: {str(e)}")
                raise

            if not base_comments:
                return []

            # Perform sentiment analysis
            return self._analyze_comments_sentiment(base_comments)

        # Smart date-aware pagination when date filtering is needed
        return await self._fetch_with_date_aware_pagination(
            subfeddit_name, validated_limit, start_date, end_date
        )

    async def _fetch_with_date_aware_pagination(
        self,
        subfeddit_name: str,
        validated_limit: int,
        start_date: datetime | None,
        end_date: datetime | None,
    ) -> list[CommentWithSentiment]:
        """
        Fetch comments using smart date-aware pagination.

        This method fetches comments in batches and uses date checking to efficiently
        find comments within the specified date range before applying the final limit.
        """
        batch_size = 100  # Process comments in batches of 100
        skip = 0
        all_matching_comments: list[CommentBase] = []

        logger.info(
            f"Using smart pagination for date range: {start_date} to {end_date}"
        )

        while True:
            # Fetch a batch of comments
            try:
                batch_comments = await self.feddit_client.get_comments(
                    subfeddit_name=subfeddit_name, skip=skip, limit=batch_size
                )
            except FedditAPIError as e:
                logger.error(
                    f"Failed to fetch comments batch (skip={skip}) for {subfeddit_name}: {str(e)}"
                )

                # If we have some comments already, return them instead of failing completely
                if all_matching_comments:
                    logger.warning(
                        f"Returning {len(all_matching_comments)} partial results due to API error"
                    )
                    break
                else:
                    raise  # No comments collected yet, re-raise the error

            # If no more comments, we're done
            if not batch_comments:
                logger.info(f"No more comments found at skip={skip}")
                break

            # Convert timestamps to datetime objects for comparison
            batch_dates = [
                datetime.fromtimestamp(comment.created_at) for comment in batch_comments
            ]
            first_comment_date = batch_dates[0]
            last_comment_date = batch_dates[-1]

            logger.debug(
                f"Batch {skip//batch_size + 1}: {len(batch_comments)} comments, dates {first_comment_date} to {last_comment_date}"
            )

            # Check if we should skip this entire batch
            if start_date and last_comment_date < start_date:
                # All comments in this batch are before start_date, skip to next batch
                logger.debug(
                    "Entire batch is before start_date, skipping to next batch"
                )
                skip += batch_size
                continue

            if end_date and first_comment_date > end_date:
                # All comments in this batch are after end_date, we're done
                logger.debug("Reached comments after end_date, stopping")
                break

            # Filter comments in this batch by date range
            filtered_batch = []
            for comment, comment_date in zip(batch_comments, batch_dates):
                # Check if comment is within date range
                if start_date and comment_date < start_date:
                    continue
                if end_date and comment_date > end_date:
                    continue

                filtered_batch.append(comment)

            # Add matching comments from this batch
            all_matching_comments.extend(filtered_batch)

            # If we have enough comments after filtering, we can stop
            if len(all_matching_comments) >= validated_limit:
                logger.info(
                    f"Found enough matching comments ({len(all_matching_comments)}), stopping"
                )
                break

            # Move to next batch
            skip += batch_size

            # Safety check to prevent infinite loops
            if skip > 10000:  # Reasonable safety limit
                logger.warning(
                    "Reached safety limit of 10000 comments, stopping pagination"
                )
                break

        # Apply final limit and perform sentiment analysis
        final_comments = all_matching_comments[:validated_limit]
        logger.info(
            f"Found {len(all_matching_comments)} matching comments, returning {len(final_comments)} after limit"
        )

        return self._analyze_comments_sentiment(final_comments)

    def _analyze_comments_sentiment(
        self, comments: list[CommentBase]
    ) -> list[CommentWithSentiment]:
        """
        Perform sentiment analysis on a list of comments.

        Args:
            comments: List of base comments to analyze

        Returns:
            List of comments with sentiment analysis
        """
        comments_with_sentiment = []
        for comment in comments:
            sentiment_result = self.sentiment_analyzer.analyze_text(comment.text)
            comment_with_sentiment = CommentWithSentiment(
                id=comment.id,
                username=comment.username,
                text=comment.text,
                created_at=comment.created_at,
                sentiment=sentiment_result,
            )
            comments_with_sentiment.append(comment_with_sentiment)
        return comments_with_sentiment

    def _process_comments(
        self,
        comments: list[CommentWithSentiment],
        start_date: datetime | None,
        end_date: datetime | None,
        sort_order: str | None,
    ) -> list[CommentWithSentiment]:
        """
        Apply filtering and sorting to comments.

        Args:
            comments: List of comments to process
            start_date: Filter comments after this date
            end_date: Filter comments before this date
            sort_order: Sort order ('asc', 'desc', or None for no sorting)

        Returns:
            Processed list of comments
        """
        # Filter by date range if specified
        filtered_comments = self._filter_comments_by_date(
            comments, start_date, end_date
        )

        # Sort comments only if sort_order is specified
        if sort_order is not None:
            sorted_comments = self._sort_comments(filtered_comments, sort_order)
            return sorted_comments

        # Return in original order (chronological) if no sorting requested
        return filtered_comments

    async def analyze_subfeddit_sentiment(
        self,
        subfeddit_name: str,
        limit: int = None,
        start_date: str | None = None,
        end_date: str | None = None,
        sort_order: str | None = None,
    ) -> SentimentAnalysisResponse:
        """
        Analyze sentiment for comments in a subfeddit.

        Args:
            subfeddit_name: Name of the subfeddit
            limit: Maximum number of comments to analyze
            start_date: Filter comments after this date (ISO format)
            end_date: Filter comments before this date (ISO format)
            sort_order: Sort order by polarity score ('asc', 'desc', or None for chronological order)

        Returns:
            SentimentAnalysisResponse with analyzed comments

        Raises:
            FedditAPIError: If failed to fetch comments from Feddit
            ValueError: If invalid parameters provided
        """
        # Validate parameters
        validated_limit = self.__validate_parameters(limit, sort_order)

        # Parse date parameters
        parsed_start_date, parsed_end_date = self._parse_date_parameters(
            start_date, end_date
        )

        logger.info(
            f"Analyzing sentiment for subfeddit: {subfeddit_name} (limit: {validated_limit})"
        )

        # Fetch comments with smart date-aware pagination and analyze sentiment
        comments_with_sentiment = await self._fetch_and_analyze_comments(
            subfeddit_name, validated_limit, parsed_start_date, parsed_end_date
        )

        # Handle empty results
        if not comments_with_sentiment:
            logger.warning(f"No comments found for subfeddit: {subfeddit_name}")
            return SentimentAnalysisResponse(
                subfeddit=subfeddit_name,
                total_comments=0,
                comments=[],
                subfeddit_info=None,
            )

        # Apply sorting if requested (date filtering already done during fetch)
        processed_comments = (
            self._sort_comments(comments_with_sentiment, sort_order)
            if sort_order
            else comments_with_sentiment
        )

        # Get subfeddit info
        subfeddit_info = await self.feddit_client.get_subfeddit_info(subfeddit_name)

        logger.info(
            f"Successfully analyzed {len(processed_comments)} comments for {subfeddit_name}"
        )

        return SentimentAnalysisResponse(
            subfeddit=subfeddit_name,
            total_comments=len(processed_comments),
            comments=processed_comments,
            subfeddit_info=subfeddit_info,
        )


# Global service instance
sentiment_service = SentimentService()
