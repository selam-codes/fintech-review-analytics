"""
Sentiment Analysis Module
=========================

Tool Selection Rationale
------------------------
Primary Model: distilbert-base-uncased-finetuned-sst-2-english (HuggingFace Transformers)
  - Pre-trained on Stanford Sentiment Treebank (SST-2), specifically fine-tuned
    for binary sentiment classification (POSITIVE / NEGATIVE).
  - Provides calibrated confidence scores (softmax probabilities).
  - Neutral classification heuristic: if max confidence < 0.70, label as "neutral".
    This threshold captures ambiguous or mixed-sentiment reviews.
  - Chosen over VADER/TextBlob for higher accuracy on informal, short-form review text
    where sarcasm, slang, and code-switching are common.

Comparison Baselines (documented below):
  - VADER: Rule-based, lexicon approach. Fast but struggles with negation, domain jargon.
  - TextBlob: Pattern-based polarity. Lightweight but less accurate on review-specific text.
  Both are included for result comparison and validation, as recommended.
"""

import pandas as pd
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# 1. DistilBERT Transformer Sentiment Analyzer (Primary)
# ---------------------------------------------------------------------------

class TransformerSentimentAnalyzer:
    """
    Classify review text as positive, negative, or neutral using
    distilbert-base-uncased-finetuned-sst-2-english.

    Neutral heuristic: max softmax probability < neutral_threshold.
    """

    def __init__(self, neutral_threshold=0.70):
        from transformers import pipeline
        self.classifier = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True,
            max_length=512,
        )
        self.neutral_threshold = neutral_threshold

    def predict_one(self, text):
        """Classify a single review string. Returns (label, score)."""
        if not isinstance(text, str) or len(text.strip()) == 0:
            return "neutral", 0.0

        # DistilBERT has a 512 token limit; truncation is handled by the pipeline
        result = self.classifier(text[:512])[0]
        label = result["label"].lower()  # 'positive' or 'negative'
        score = round(result["score"], 4)

        # Apply neutral threshold
        if score < self.neutral_threshold:
            return "neutral", score

        return label, score

    def predict_batch(self, texts, batch_size=32):
        """
        Classify a list/Series of review texts.
        Returns list of (label, score) tuples.
        """
        results = []
        text_list = list(texts)

        for i in tqdm(range(0, len(text_list), batch_size),
                      desc="DistilBERT sentiment", unit="batch"):
            batch = text_list[i : i + batch_size]
            # Handle empty/null strings
            clean_batch = [
                t[:512] if isinstance(t, str) and len(t.strip()) > 0 else "neutral"
                for t in batch
            ]
            preds = self.classifier(clean_batch)
            for j, pred in enumerate(preds):
                orig = batch[j]
                if not isinstance(orig, str) or len(orig.strip()) == 0:
                    results.append(("neutral", 0.0))
                elif pred["score"] < self.neutral_threshold:
                    results.append(("neutral", round(pred["score"], 4)))
                else:
                    results.append((pred["label"].lower(), round(pred["score"], 4)))

        return results


# ---------------------------------------------------------------------------
# 2. VADER Sentiment Analyzer (Baseline Comparison)
# ---------------------------------------------------------------------------

class VADERSentimentAnalyzer:
    """
    Rule-based sentiment analysis using NLTK's VADER.
    Compound score mapping:
      compound >= 0.05  -> positive
      compound <= -0.05 -> negative
      otherwise         -> neutral
    """

    def __init__(self):
        import nltk
        nltk.download('vader_lexicon', quiet=True)
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        self.analyzer = SentimentIntensityAnalyzer()

    def predict_one(self, text):
        if not isinstance(text, str) or len(text.strip()) == 0:
            return "neutral", 0.0

        scores = self.analyzer.polarity_scores(text)
        compound = scores['compound']

        if compound >= 0.05:
            return "positive", round(compound, 4)
        elif compound <= -0.05:
            return "negative", round(abs(compound), 4)
        else:
            return "neutral", round(abs(compound), 4)

    def predict_batch(self, texts, batch_size=None):
        """Process all texts (batch_size ignored for VADER — it's already fast)."""
        return [self.predict_one(t) for t in tqdm(texts, desc="VADER sentiment")]


# ---------------------------------------------------------------------------
# 3. TextBlob Sentiment Analyzer (Baseline Comparison)
# ---------------------------------------------------------------------------

class TextBlobSentimentAnalyzer:
    """
    Pattern-based sentiment using TextBlob polarity.
    Polarity mapping:
      polarity > 0.1   -> positive
      polarity < -0.1  -> negative
      otherwise         -> neutral
    """

    def __init__(self):
        from textblob import TextBlob  # noqa: F401 — validate import
        self._TextBlob = TextBlob

    def predict_one(self, text):
        if not isinstance(text, str) or len(text.strip()) == 0:
            return "neutral", 0.0

        blob = self._TextBlob(text)
        polarity = blob.sentiment.polarity

        if polarity > 0.1:
            return "positive", round(polarity, 4)
        elif polarity < -0.1:
            return "negative", round(abs(polarity), 4)
        else:
            return "neutral", round(abs(polarity), 4)

    def predict_batch(self, texts, batch_size=None):
        return [self.predict_one(t) for t in tqdm(texts, desc="TextBlob sentiment")]


# ---------------------------------------------------------------------------
# 4. Aggregation Utilities
# ---------------------------------------------------------------------------

def aggregate_sentiment_by_bank(df, label_col='sentiment_label', score_col='sentiment_score'):
    """
    Aggregate sentiment scores grouped by bank.
    Returns a DataFrame with mean score, label distribution, and counts.
    """
    agg = df.groupby('bank').agg(
        total_reviews=(label_col, 'count'),
        mean_score=(score_col, 'mean'),
        positive_count=(label_col, lambda x: (x == 'positive').sum()),
        negative_count=(label_col, lambda x: (x == 'negative').sum()),
        neutral_count=(label_col, lambda x: (x == 'neutral').sum()),
    ).reset_index()

    agg['positive_pct'] = round(agg['positive_count'] / agg['total_reviews'] * 100, 1)
    agg['negative_pct'] = round(agg['negative_count'] / agg['total_reviews'] * 100, 1)
    agg['neutral_pct'] = round(agg['neutral_count'] / agg['total_reviews'] * 100, 1)

    return agg


def aggregate_sentiment_by_rating(df, label_col='sentiment_label', score_col='sentiment_score'):
    """
    Aggregate mean sentiment score by star rating, optionally per bank.
    Useful for validating sentiment model: 5-star reviews should be mostly positive.
    """
    agg = df.groupby(['bank', 'rating']).agg(
        count=(label_col, 'count'),
        mean_score=(score_col, 'mean'),
        positive_pct=(label_col, lambda x: round((x == 'positive').mean() * 100, 1)),
        negative_pct=(label_col, lambda x: round((x == 'negative').mean() * 100, 1)),
    ).reset_index()

    return agg


def sentiment_coverage(df, label_col='sentiment_label'):
    """Calculate what percentage of reviews received a sentiment label."""
    total = len(df)
    labeled = df[label_col].notna().sum()
    pct = round(labeled / total * 100, 1) if total > 0 else 0.0
    return labeled, total, pct
