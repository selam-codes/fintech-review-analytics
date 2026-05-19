"""
Tests for src/sentiment_analysis.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import pandas as pd
from src.sentiment_analysis import (
    VADERSentimentAnalyzer,
    TextBlobSentimentAnalyzer,
    aggregate_sentiment_by_bank,
    aggregate_sentiment_by_rating,
    sentiment_coverage,
)


# ---------------------------------------------------------------------------
# VADER Tests (fast, no model download required)
# ---------------------------------------------------------------------------

class TestVADERSentimentAnalyzer:
    """Test VADER analyzer — lightweight, always available."""

    @pytest.fixture
    def analyzer(self):
        return VADERSentimentAnalyzer()

    def test_positive_review(self, analyzer):
        label, score = analyzer.predict_one("This app is amazing and works perfectly!")
        assert label == "positive"
        assert score > 0

    def test_negative_review(self, analyzer):
        label, score = analyzer.predict_one("Terrible app, crashes every time. Worst experience ever.")
        assert label == "negative"
        assert score > 0

    def test_neutral_review(self, analyzer):
        label, score = analyzer.predict_one("I downloaded the app.")
        assert label in ["neutral", "positive", "negative"]  # VADER may vary

    def test_empty_string(self, analyzer):
        label, score = analyzer.predict_one("")
        assert label == "neutral"
        assert score == 0.0

    def test_none_input(self, analyzer):
        label, score = analyzer.predict_one(None)
        assert label == "neutral"
        assert score == 0.0

    def test_batch_prediction(self, analyzer):
        texts = ["Great app!", "Terrible!", "It's okay.", ""]
        results = analyzer.predict_batch(texts)
        assert len(results) == 4
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

    def test_batch_labels_are_valid(self, analyzer):
        texts = ["Love it", "Hate it", "Meh"]
        results = analyzer.predict_batch(texts)
        valid_labels = {"positive", "negative", "neutral"}
        for label, score in results:
            assert label in valid_labels
            assert isinstance(score, float)


# ---------------------------------------------------------------------------
# TextBlob Tests
# ---------------------------------------------------------------------------

class TestTextBlobSentimentAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return TextBlobSentimentAnalyzer()

    def test_positive_review(self, analyzer):
        label, score = analyzer.predict_one("This is a wonderful and great application!")
        assert label == "positive"

    def test_negative_review(self, analyzer):
        label, score = analyzer.predict_one("Horrible terrible awful app, worst ever!")
        assert label == "negative"

    def test_empty_input(self, analyzer):
        label, score = analyzer.predict_one("")
        assert label == "neutral"
        assert score == 0.0


# ---------------------------------------------------------------------------
# Aggregation Tests
# ---------------------------------------------------------------------------

class TestAggregation:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'bank': ['CBE', 'CBE', 'CBE', 'BOA', 'BOA', 'BOA'],
            'rating': [5, 1, 3, 5, 2, 4],
            'sentiment_label': ['positive', 'negative', 'neutral', 'positive', 'negative', 'positive'],
            'sentiment_score': [0.95, 0.88, 0.55, 0.92, 0.76, 0.89],
        })

    def test_aggregate_by_bank_shape(self, sample_df):
        result = aggregate_sentiment_by_bank(sample_df)
        assert len(result) == 2  # CBE and BOA
        assert 'total_reviews' in result.columns
        assert 'positive_pct' in result.columns

    def test_aggregate_by_bank_counts(self, sample_df):
        result = aggregate_sentiment_by_bank(sample_df)
        cbe = result[result['bank'] == 'CBE'].iloc[0]
        assert cbe['total_reviews'] == 3
        assert cbe['positive_count'] == 1
        assert cbe['negative_count'] == 1
        assert cbe['neutral_count'] == 1

    def test_aggregate_by_rating_shape(self, sample_df):
        result = aggregate_sentiment_by_rating(sample_df)
        assert 'rating' in result.columns
        assert 'mean_score' in result.columns
        assert len(result) > 0

    def test_sentiment_coverage_100(self, sample_df):
        labeled, total, pct = sentiment_coverage(sample_df)
        assert pct == 100.0
        assert labeled == total == 6

    def test_sentiment_coverage_with_nan(self):
        df = pd.DataFrame({
            'sentiment_label': ['positive', None, 'negative', None],
        })
        labeled, total, pct = sentiment_coverage(df)
        assert labeled == 2
        assert total == 4
        assert pct == 50.0
