"""
Unit tests for src/data_scraping.py
"""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from src.data_scraping import scrape_fintech_reviews


class TestScrapeReviews:
    """Tests for the scrape_fintech_reviews function."""

    @patch('src.data_scraping.reviews')
    def test_returns_dataframe(self, mock_reviews):
        """Should return a pandas DataFrame."""
        mock_reviews.return_value = ([{
            'reviewId': '1',
            'content': 'Great app',
            'score': 5,
            'at': pd.Timestamp('2024-01-01'),
        }], None)

        result = scrape_fintech_reviews({"TestBank": "com.test.app"}, count=1)
        assert isinstance(result, pd.DataFrame)

    @patch('src.data_scraping.reviews')
    def test_correct_columns(self, mock_reviews):
        """Should have the expected columns."""
        mock_reviews.return_value = ([{
            'reviewId': 'r1',
            'content': 'Nice',
            'score': 4,
            'at': pd.Timestamp('2024-06-15'),
        }], None)

        result = scrape_fintech_reviews({"CBE": "com.test"}, count=1)
        expected_cols = {'bank', 'reviewId', 'reviewText', 'rating', 'date', 'source'}
        assert set(result.columns) == expected_cols

    @patch('src.data_scraping.reviews')
    def test_bank_name_is_set(self, mock_reviews):
        """Should assign the correct bank name to each review."""
        mock_reviews.return_value = ([{
            'reviewId': 'r1',
            'content': 'Works well',
            'score': 5,
            'at': pd.Timestamp('2024-01-01'),
        }], None)

        result = scrape_fintech_reviews({"Dashen": "com.dashen"}, count=1)
        assert result.iloc[0]['bank'] == 'Dashen'

    @patch('src.data_scraping.reviews')
    def test_source_is_google_play(self, mock_reviews):
        """Source column should always be 'Google Play'."""
        mock_reviews.return_value = ([{
            'reviewId': 'r1',
            'content': 'OK',
            'score': 3,
            'at': pd.Timestamp('2024-01-01'),
        }], None)

        result = scrape_fintech_reviews({"BOA": "com.boa"}, count=1)
        assert result.iloc[0]['source'] == 'Google Play'

    @patch('src.data_scraping.reviews')
    def test_retries_on_failure(self, mock_reviews):
        """Should retry on failure and succeed on later attempt."""
        mock_reviews.side_effect = [
            Exception("Network error"),
            ([{
                'reviewId': 'r1',
                'content': 'Finally works',
                'score': 5,
                'at': pd.Timestamp('2024-01-01'),
            }], None),
        ]

        result = scrape_fintech_reviews({"CBE": "com.cbe"}, count=1, retries=3)
        assert len(result) == 1

    @patch('src.data_scraping.reviews')
    def test_empty_dataframe_on_total_failure(self, mock_reviews):
        """Should return empty DataFrame if all retries fail."""
        mock_reviews.side_effect = Exception("Always fails")

        result = scrape_fintech_reviews({"CBE": "com.cbe"}, count=1, retries=2)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
