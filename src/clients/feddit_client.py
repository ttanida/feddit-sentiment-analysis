"""HTTP client for interacting with the Feddit API."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any

import httpx

from ..config import settings
from ..models import CommentBase, SubfedditInfo

logger = logging.getLogger(__name__)


class FedditAPIError(Exception):
    """Exception raised for Feddit API errors."""

    pass


class FedditClient:
    """Async HTTP client for Feddit API."""

    def __init__(self):
        """Initialize the Feddit client."""
        self.base_url = settings.feddit_base_url
        self.timeout = settings.feddit_timeout
        self.max_retries = settings.feddit_max_retries

        # Caching for subfeddits
        self._subfeddits_cache: list[dict[str, Any]] | None = None
        self._cache_timestamp: datetime | None = None
        self._cache_ttl = timedelta(minutes=10)  # Cache for 10 minutes

        # Efficient lookups
        self._name_to_id_cache: dict[str, int] = {}
        self._name_to_info_cache: dict[str, SubfedditInfo] = {}

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        retries: int = 0,
    ) -> dict[str, Any]:
        """
        Make HTTP request to Feddit API with retry logic.

        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Query parameters
            retries: Current retry count

        Returns:
            JSON response data

        Raises:
            FedditAPIError: If request fails after max retries
        """
        url = f"{self.base_url}{endpoint}"

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.request(method, url, params=params)
                response.raise_for_status()
                return response.json()

        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error {e.response.status_code} for {url}: {e.response.text}"
            )
            if retries < self.max_retries:
                await asyncio.sleep(2**retries)  # Exponential backoff
                return await self._make_request(method, endpoint, params, retries + 1)
            raise FedditAPIError(f"HTTP {e.response.status_code}: {e.response.text}")

        except httpx.RequestError as e:
            logger.error(f"Request error for {url}: {str(e)}")
            if retries < self.max_retries:
                await asyncio.sleep(2**retries)
                return await self._make_request(method, endpoint, params, retries + 1)
            raise FedditAPIError(f"Request failed: {str(e)}")

    def __is_cache_valid(self) -> bool:
        """Check if the subfeddits cache is still valid."""
        if self._cache_timestamp is None or self._subfeddits_cache is None:
            return False
        return datetime.now() - self._cache_timestamp < self._cache_ttl

    def __update_lookup_caches(self, subfeddits: list[dict[str, Any]]) -> None:
        """Update the efficient lookup caches."""
        self._name_to_id_cache.clear()
        self._name_to_info_cache.clear()

        for subfeddit in subfeddits:
            title = subfeddit.get("title", "").lower()
            if title:
                # Cache ID lookup
                self._name_to_id_cache[title] = subfeddit["id"]

                # Cache SubfedditInfo
                self._name_to_info_cache[title] = SubfedditInfo(
                    id=str(subfeddit["id"]),
                    username=subfeddit["username"],
                    title=subfeddit["title"],
                    description=subfeddit["description"],
                )

    async def _get_subfeddits(self) -> list[dict[str, Any]]:
        """
        Get list of available subfeddits with caching.

        Returns:
            List of subfeddit data
        """
        # Return cached data if valid
        if self.__is_cache_valid():
            logger.debug("Using cached subfeddits data")
            return self._subfeddits_cache

        try:
            logger.debug("Fetching fresh subfeddits data from API")
            data = await self._make_request("GET", "/api/v1/subfeddits/")
            subfeddits = data.get("subfeddits", [])

            # Update cache
            self._subfeddits_cache = subfeddits
            self._cache_timestamp = datetime.now()

            # Update efficient lookup caches
            self.__update_lookup_caches(subfeddits)

            logger.debug(f"Cached {len(subfeddits)} subfeddits")
            return subfeddits

        except FedditAPIError:
            logger.error("Failed to fetch subfeddits")
            # Return cached data if available, even if expired
            if self._subfeddits_cache is not None:
                logger.warning("Returning expired cache due to API error")
                return self._subfeddits_cache
            return []

    async def get_subfeddit_info(self, subfeddit_name: str) -> SubfedditInfo | None:
        """
        Get information about a specific subfeddit by name.

        Args:
            subfeddit_name: Name/title of the subfeddit

        Returns:
            SubfedditInfo object or None if not found
        """
        name_lower = subfeddit_name.lower()

        # Try cache first
        if self.__is_cache_valid() and name_lower in self._name_to_info_cache:
            logger.debug(f"Using cached info for subfeddit: {subfeddit_name}")
            return self._name_to_info_cache[name_lower]

        # Cache miss - refresh and try again
        try:
            await self._get_subfeddits()  # This will update the cache
            return self._name_to_info_cache.get(name_lower)
        except Exception as e:
            logger.error(f"Error getting subfeddit info for {subfeddit_name}: {str(e)}")
            return None

    async def _get_subfeddit_id_by_name(self, subfeddit_name: str) -> int | None:
        """
        Get subfeddit ID by name/title.

        Args:
            subfeddit_name: Name/title of the subfeddit

        Returns:
            Subfeddit ID or None if not found
        """
        name_lower = subfeddit_name.lower()

        # Try cache first
        if self.__is_cache_valid() and name_lower in self._name_to_id_cache:
            logger.debug(f"Using cached ID for subfeddit: {subfeddit_name}")
            return self._name_to_id_cache[name_lower]

        # Cache miss - refresh and try again
        try:
            await self._get_subfeddits()  # This will update the cache
            return self._name_to_id_cache.get(name_lower)
        except Exception as e:
            logger.error(f"Error getting subfeddit ID for {subfeddit_name}: {str(e)}")
            return None

    async def get_comments(
        self, subfeddit_name: str, skip: int = 0, limit: int = 25
    ) -> list[CommentBase]:
        """
        Get comments from a subfeddit by name.

        Args:
            subfeddit_name: Name/title of the subfeddit
            skip: Number of comments to skip
            limit: Maximum number of comments to return

        Returns:
            List of CommentBase objects
        """
        try:
            # First get the subfeddit ID
            subfeddit_id = await self._get_subfeddit_id_by_name(subfeddit_name)
            if subfeddit_id is None:
                raise FedditAPIError(f"Subfeddit '{subfeddit_name}' not found")

            # Then fetch comments using the ID
            params = {"subfeddit_id": subfeddit_id, "skip": skip, "limit": limit}
            data = await self._make_request("GET", "/api/v1/comments/", params=params)

            comments = []
            for comment_data in data.get("comments", []):
                comments.append(
                    CommentBase(
                        id=str(comment_data["id"]),
                        username=comment_data["username"],
                        text=comment_data["text"],
                        created_at=comment_data["created_at"],
                    )
                )

            return comments

        except FedditAPIError as e:
            logger.error(f"Failed to fetch comments for {subfeddit_name}: {str(e)}")
            raise


# Global client instance
feddit_client = FedditClient()
