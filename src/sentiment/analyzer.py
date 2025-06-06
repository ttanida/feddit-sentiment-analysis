"""
Sentiment analysis implementation using TextBlob.

This implementation uses TextBlob for sentiment analysis as the focus of this challenge
is on engineering practices and API design rather than ML model optimization.

TextBlob provides:
- Simple sentiment analysis
- No external API dependencies
- Fast processing with custom caching layer
- Binary classification (positive/negative only)

For production systems requiring higher accuracy, consider alternatives like:
- External APIs (OpenAI, Google Cloud Natural Language, AWS Comprehend)
- Self-hosted transformer models (BERT, RoBERTa, DistilBERT)
- Custom trained models on domain-specific data
- Ensemble approaches combining multiple models (e.g. VADER, BERT, etc.)
"""

import logging

from textblob import TextBlob

from ..config import settings
from ..models import SentimentResult
from ..utils import sentiment_cache

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Sentiment analyzer using TextBlob with binary classification.

    TextBlob was chosen for this engineering demonstration because:
    1. Simple integration
    2. No external API dependencies or rate limits
    3. Consistent performance for development and testing
    4. Built-in polarity scoring (-1 to +1 scale)

    Classification logic:
    - Polarity >= 0: Positive
    - Polarity < 0: Negative

    The modular design allows easy replacement with more sophisticated
    models when higher accuracy is required.
    """

    def __init__(self):
        """Initialize the sentiment analyzer."""
        pass

    def __classify_sentiment(self, polarity_score: float) -> str:
        """
        Classify sentiment based on polarity score.

        Args:
            polarity_score: The polarity score from TextBlob

        Returns:
            Classification string: 'positive' or 'negative'
        """
        if polarity_score >= 0:
            return "positive"
        else:
            return "negative"

    def analyze_text(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of a single text using TextBlob.

        TextBlob uses a pre-trained model based on movie reviews and provides
        polarity scores from -1 (negative) to +1 (positive).

        Args:
            text: The text to analyze

        Returns:
            SentimentResult with polarity score and classification
        """
        # Check cache first to avoid redundant analysis
        cache_key = sentiment_cache.create_key(text)
        cached_result = sentiment_cache.get(cache_key)

        if cached_result:
            logger.debug(f"Cache hit for text: {text[:50]}...")
            return cached_result

        try:
            # Perform sentiment analysis with TextBlob
            blob = TextBlob(text)
            polarity_score = blob.sentiment.polarity

            # Classify as positive (>= 0) or negative (< 0)
            classification = self.__classify_sentiment(polarity_score)

            result = SentimentResult(
                polarity_score=polarity_score, classification=classification
            )

            # Cache the result for improved performance
            sentiment_cache.set(cache_key, result, settings.cache_ttl_seconds)

            logger.debug(
                f"Analyzed sentiment for text: {text[:50]}... -> {classification} ({polarity_score:.3f})"
            )
            return result

        except Exception as e:
            logger.error(
                f"Error analyzing sentiment for text: {text[:50]}... - {str(e)}"
            )
            # Return positive sentiment as fallback to maintain service reliability
            return SentimentResult(polarity_score=0.0, classification="positive")

    def analyze_batch(self, texts: list[str]) -> list[SentimentResult]:
        """
        Analyze sentiment for a batch of texts.

        Args:
            texts: List of texts to analyze

        Returns:
            List of SentimentResult objects
        """
        results = []
        for text in texts:
            results.append(self.analyze_text(text))
        return results


# Global analyzer instance
sentiment_analyzer = SentimentAnalyzer()
