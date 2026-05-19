"""
Unit tests for src/preprocessing.py
"""

import pandas as pd
import pytest
from src.preprocessing import (
    clean_text,
    tokenize_and_lemmatize,
    normalize_schema,
    preprocess_dataframe,
    robust_clean,
)


# ─── clean_text tests ────────────────────────────────────────────

class TestCleanText:
    def test_lowercases_text(self):
        assert clean_text("HELLO World") == "hello world"

    def test_removes_punctuation(self):
        result = clean_text("Great app! Works well.")
        assert "!" not in result
        assert "." not in result

    def test_removes_digits(self):
        result = clean_text("Version 3 is great in 2024")
        assert "3" not in result
        assert "2024" not in result

    def test_handles_nan(self):
        assert clean_text(float('nan')) == ""

    def test_handles_none_like(self):
        assert clean_text(None) == ""

    def test_handles_numeric_input(self):
        result = clean_text(12345)
        assert isinstance(result, str)

    def test_empty_string(self):
        assert clean_text("") == ""


# ─── tokenize_and_lemmatize tests ────────────────────────────────

class TestTokenizeAndLemmatize:
    def test_removes_stopwords(self):
        result = tokenize_and_lemmatize("this is a very good application")
        assert "this" not in result.split()
        assert "is" not in result.split()

    def test_lemmatizes_words(self):
        result = tokenize_and_lemmatize("the banks are running smoothly")
        assert "bank" in result.split()

    def test_filters_short_tokens(self):
        result = tokenize_and_lemmatize("I am so ok no")
        # Tokens with len <= 2 should be removed
        for token in result.split():
            assert len(token) > 2

    def test_empty_string(self):
        result = tokenize_and_lemmatize("")
        assert result == ""


# ─── normalize_schema tests ──────────────────────────────────────

class TestNormalizeSchema:
    def test_renames_reviewText_to_review(self):
        df = pd.DataFrame({"reviewText": ["good"], "rating": [5]})
        result = normalize_schema(df)
        assert "review" in result.columns
        assert "reviewText" not in result.columns

    def test_renames_content_to_review(self):
        df = pd.DataFrame({"content": ["nice"], "rating": [4]})
        result = normalize_schema(df)
        assert "review" in result.columns

    def test_renames_Bank_to_bank(self):
        df = pd.DataFrame({"Bank": ["CBE"], "review": ["ok"]})
        result = normalize_schema(df)
        assert "bank" in result.columns
        assert "Bank" not in result.columns

    def test_renames_score_to_rating(self):
        df = pd.DataFrame({"score": [5], "review": ["good"]})
        result = normalize_schema(df)
        assert "rating" in result.columns

    def test_renames_at_to_date(self):
        df = pd.DataFrame({"at": ["2024-01-01"], "review": ["test"]})
        result = normalize_schema(df)
        assert "date" in result.columns

    def test_no_rename_when_target_exists(self):
        """Should not rename if the target column already exists."""
        df = pd.DataFrame({"reviewText": ["a"], "review": ["b"]})
        result = normalize_schema(df)
        assert "reviewText" in result.columns
        assert "review" in result.columns


# ─── preprocess_dataframe tests ──────────────────────────────────

class TestPreprocessDataframe:
    def test_adds_clean_and_processed_columns(self):
        df = pd.DataFrame({"review": ["The application is great!"]})
        result = preprocess_dataframe(df)
        assert "clean_text" in result.columns
        assert "processed_content" in result.columns

    def test_handles_missing_column_gracefully(self):
        df = pd.DataFrame({"other_col": ["test"]})
        result = preprocess_dataframe(df, text_column="missing")
        # Should return df without crashing
        assert result is not None


# ─── robust_clean tests ──────────────────────────────────────────

class TestRobustClean:
    def test_drops_missing_review(self):
        df = pd.DataFrame({
            "review": ["good app", None, "nice"],
            "rating": [5, 4, 3],
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        })
        result = robust_clean(df)
        assert len(result) == 2

    def test_drops_short_reviews(self):
        df = pd.DataFrame({
            "review": ["ab", "good app", "x"],
            "rating": [3, 5, 1],
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        })
        result = robust_clean(df)
        assert len(result) == 1
        assert result.iloc[0]["review"].strip() == "good app"

    def test_normalizes_dates(self):
        df = pd.DataFrame({
            "review": ["test review"],
            "rating": [5],
            "date": ["January 15, 2024"],
        })
        result = robust_clean(df)
        assert result.iloc[0]["date"] == "2024-01-15"

    def test_resets_index(self):
        df = pd.DataFrame({
            "review": [None, "valid review", None],
            "rating": [1, 5, 3],
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        })
        result = robust_clean(df)
        assert list(result.index) == list(range(len(result)))
