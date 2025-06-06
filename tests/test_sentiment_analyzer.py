"""Tests for sentiment analyzer."""

from src.models import SentimentResult
from src.sentiment.analyzer import SentimentAnalyzer


class TestSentimentAnalyzer:
    """Test suite for SentimentAnalyzer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = SentimentAnalyzer()

    def test_analyze_positive_text(self):
        """Test analyzing positive sentiment text."""
        text = "I love this amazing product! It's fantastic and wonderful!"
        result = self.analyzer.analyze_text(text)

        assert isinstance(result, SentimentResult)
        assert result.polarity_score > 0
        assert result.classification == "positive"

    def test_analyze_negative_text(self):
        """Test analyzing negative sentiment text."""
        text = "This is terrible and awful! I hate it completely."
        result = self.analyzer.analyze_text(text)

        assert isinstance(result, SentimentResult)
        assert result.polarity_score < 0
        assert result.classification == "negative"

    def test_analyze_mixed_sentiment_text(self):
        """Test analyzing text with mixed/ambiguous sentiment."""
        text = "This is a regular product. It works as expected."
        result = self.analyzer.analyze_text(text)

        assert isinstance(result, SentimentResult)
        assert result.classification in ["positive", "negative"]  # Only positive or negative allowed

    def test_analyze_empty_text(self):
        """Test analyzing empty text."""
        result = self.analyzer.analyze_text("")

        assert isinstance(result, SentimentResult)
        assert result.polarity_score == 0.0
        assert result.classification == "positive"  # 0.0 is classified as positive (>= 0)

    def test_analyze_batch(self):
        """Test batch sentiment analysis."""
        texts = ["I love this!", "This is terrible!", "This is okay."]

        results = self.analyzer.analyze_batch(texts)

        assert len(results) == 3
        assert all(isinstance(r, SentimentResult) for r in results)

        # First should be positive
        assert results[0].polarity_score > 0
        # Second should be negative
        assert results[1].polarity_score < 0

    def test_cache_functionality(self):
        """Test that caching works correctly."""
        text = "This is a test for caching functionality."

        # First analysis
        result1 = self.analyzer.analyze_text(text)

        # Second analysis should use cache
        result2 = self.analyzer.analyze_text(text)

        assert result1.polarity_score == result2.polarity_score
        assert result1.classification == result2.classification

    def test_polarity_score_range(self):
        """Test that polarity scores are within expected range."""
        texts = ["Amazing wonderful fantastic!", "Terrible awful horrible!", "This is regular text."]

        for text in texts:
            result = self.analyzer.analyze_text(text)
            assert -1.0 <= result.polarity_score <= 1.0

    def test_error_handling_fallback(self):
        """Test that analyzer handles errors gracefully with fallback."""
        # Test with problematic input that might cause TextBlob to fail
        # This is more of a safety test to ensure the fallback behavior works
        from unittest.mock import patch

        with patch("src.sentiment.analyzer.TextBlob") as mock_textblob:
            mock_textblob.side_effect = Exception("TextBlob error")

            result = self.analyzer.analyze_text("test text")

            # Should return fallback values as per implementation
            assert result.polarity_score == 0.0
            assert result.classification == "positive"
