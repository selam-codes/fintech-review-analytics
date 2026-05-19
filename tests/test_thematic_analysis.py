"""
Tests for src/thematic_analysis.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import pandas as pd
import numpy as np
from src.thematic_analysis import (
    extract_tfidf_keywords,
    assign_theme,
    assign_themes_to_df,
    summarize_themes_per_bank,
    get_theme_keywords_evidence,
    THEME_KEYWORDS,
)


# ---------------------------------------------------------------------------
# Theme Assignment Tests
# ---------------------------------------------------------------------------

class TestAssignTheme:
    def test_login_theme(self):
        text = "cannot login password error otp verification failed"
        theme = assign_theme(text)
        assert theme == "Account Access & Login"

    def test_transaction_theme(self):
        text = "transfer slow loading error payment failed"
        theme = assign_theme(text)
        assert theme == "Transaction Performance"

    def test_support_theme(self):
        text = "customer support help complaint service terrible"
        theme = assign_theme(text)
        assert theme == "Customer Support"

    def test_ui_theme(self):
        text = "beautiful design interface clean simple easy love great"
        theme = assign_theme(text)
        assert theme == "UI & App Design"

    def test_feature_request_theme(self):
        text = "please fix bug add feature update improve missing"
        theme = assign_theme(text)
        assert theme == "Feature Requests & Bugs"

    def test_empty_string(self):
        theme = assign_theme("")
        assert theme == "Other / General"

    def test_none_input(self):
        theme = assign_theme(None)
        assert theme == "Other / General"

    def test_unrelated_text(self):
        theme = assign_theme("xyz abc 123 qwerty")
        assert theme == "Other / General"


# ---------------------------------------------------------------------------
# TF-IDF Extraction Tests
# ---------------------------------------------------------------------------

class TestTfidfExtraction:
    @pytest.fixture
    def sample_texts(self):
        return [
            "login error password failed access account",
            "transfer slow payment error failed",
            "great app beautiful design easy use",
            "customer support help resolve issue",
            "login failed verification otp code",
        ] * 10  # Repeat for min_df threshold

    def test_returns_list_of_tuples(self, sample_texts):
        keywords = extract_tfidf_keywords(sample_texts, top_n=10)
        assert isinstance(keywords, list)
        assert all(isinstance(kw, tuple) and len(kw) == 2 for kw in keywords)

    def test_top_n_respected(self, sample_texts):
        keywords = extract_tfidf_keywords(sample_texts, top_n=5)
        assert len(keywords) <= 5

    def test_scores_are_positive(self, sample_texts):
        keywords = extract_tfidf_keywords(sample_texts, top_n=10)
        for term, score in keywords:
            assert score >= 0


# ---------------------------------------------------------------------------
# DataFrame Theme Assignment Tests
# ---------------------------------------------------------------------------

class TestAssignThemesToDf:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'bank': ['CBE', 'CBE', 'BOA', 'BOA', 'Dashen'],
            'processed_content': [
                'login password error failed',
                'great app easy use beautiful',
                'transfer slow payment error',
                'customer support help resolve',
                'please fix bug update feature',
            ]
        })

    def test_adds_theme_column(self, sample_df):
        result = assign_themes_to_df(sample_df)
        assert 'identified_theme' in result.columns

    def test_all_rows_have_theme(self, sample_df):
        result = assign_themes_to_df(sample_df)
        assert result['identified_theme'].notna().all()

    def test_themes_are_valid(self, sample_df):
        result = assign_themes_to_df(sample_df)
        valid = set(THEME_KEYWORDS.keys()) | {"Other / General"}
        for theme in result['identified_theme']:
            assert theme in valid


# ---------------------------------------------------------------------------
# Theme Summary Tests
# ---------------------------------------------------------------------------

class TestSummarizeThemes:
    def test_summary_has_required_columns(self):
        df = pd.DataFrame({
            'bank': ['CBE', 'CBE', 'BOA'],
            'identified_theme': ['UI & App Design', 'Account Access & Login', 'Transaction Performance'],
        })
        summary = summarize_themes_per_bank(df)
        assert 'bank' in summary.columns
        assert 'count' in summary.columns
        assert 'pct' in summary.columns

    def test_percentages_sum_to_100(self):
        df = pd.DataFrame({
            'bank': ['CBE'] * 10,
            'identified_theme': ['UI & App Design'] * 6 + ['Transaction Performance'] * 4,
        })
        summary = summarize_themes_per_bank(df)
        total_pct = summary['pct'].sum()
        assert abs(total_pct - 100.0) < 0.5  # floating point tolerance
