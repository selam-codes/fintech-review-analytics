"""
Thematic Analysis Module
========================

Approach & Grouping Logic
-------------------------
1. **Keyword Extraction**: TF-IDF (unigrams + bigrams) is used to surface the
   most distinctive terms per bank. scikit-learn's TfidfVectorizer with
   sublinear TF and English stop-words provides a clean signal.

2. **Theme Grouping**: Keywords are mapped to business-relevant themes using a
   curated keyword-to-theme dictionary. This rule-based approach was chosen for:
   - Interpretability: every theme assignment is traceable to specific keywords.
   - Domain relevance: themes are fintech-specific (not generic LDA topics).
   - Reproducibility: deterministic mapping, no random seed sensitivity.

3. **Theme Definitions** (3–5 per bank):
   - "Account Access & Login"   — login, password, OTP, verification, access
   - "Transaction Performance"  — transfer, slow, loading, crash, error, fail
   - "Customer Support"         — support, help, response, complaint, service
   - "UI & App Design"          — interface, design, update, feature, user-friendly
   - "Feature Requests & Bugs"  — bug, fix, add, need, request, improve, missing

4. **Optional LDA**: An LDA-based topic model is also provided for unsupervised
   theme discovery / validation against the rule-based approach.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from collections import defaultdict


# ---------------------------------------------------------------------------
# Theme Definitions (keyword → theme mapping)
# ---------------------------------------------------------------------------

THEME_KEYWORDS = {
    "Account Access & Login": [
        "login", "log", "password", "otp", "verification", "verify", "access",
        "account", "register", "registration", "sign", "fingerprint",
        "biometric", "pin", "authentication", "locked", "lock", "unlock",
        "session", "expired", "code", "sms",
    ],
    "Transaction Performance": [
        "transfer", "transaction", "slow", "loading", "load", "crash",
        "error", "fail", "failed", "timeout", "hang", "freeze", "stuck",
        "pending", "delay", "speed", "fast", "quick", "performance",
        "payment", "pay", "send", "receive", "balance", "money",
        "network", "connection", "server", "working",
    ],
    "Customer Support": [
        "support", "help", "response", "complaint", "service", "call",
        "customer", "agent", "staff", "branch", "office", "feedback",
        "resolve", "resolved", "assist", "contact", "phone",
    ],
    "UI & App Design": [
        "interface", "design", "ui", "ux", "user", "friendly", "beautiful",
        "layout", "theme", "dark", "mode", "look", "clean", "simple",
        "easy", "intuitive", "modern", "nice", "good", "great", "love",
        "best", "excellent", "wonderful", "amazing", "smooth",
    ],
    "Feature Requests & Bugs": [
        "bug", "fix", "add", "need", "request", "improve", "improvement",
        "missing", "feature", "update", "version", "new", "wish",
        "suggest", "suggestion", "problem", "issue", "please",
        "change", "upgrade", "option",
    ],
}


# ---------------------------------------------------------------------------
# 1. TF-IDF Keyword Extraction
# ---------------------------------------------------------------------------

def extract_tfidf_keywords(texts, top_n=30, ngram_range=(1, 2), max_features=5000):
    """
    Extract the top-N keywords/bigrams from a corpus using TF-IDF.

    Args:
        texts: iterable of preprocessed text strings
        top_n: number of top keywords to return
        ngram_range: tuple for unigram/bigram range
        max_features: max vocabulary size

    Returns:
        list of (term, tfidf_score) sorted descending
    """
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=True,
        stop_words='english',
        min_df=3,
        max_df=0.85,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    # Mean TF-IDF score across all documents
    mean_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
    top_indices = mean_scores.argsort()[::-1][:top_n]

    keywords = [(feature_names[i], round(mean_scores[i], 4)) for i in top_indices]
    return keywords


def extract_keywords_per_bank(df, text_col='processed_content', top_n=30):
    """
    Extract top TF-IDF keywords for each bank separately.

    Returns:
        dict: {bank_name: [(keyword, score), ...]}
    """
    bank_keywords = {}
    for bank in df['bank'].unique():
        bank_texts = df[df['bank'] == bank][text_col].dropna().tolist()
        if len(bank_texts) < 10:
            bank_keywords[bank] = []
            continue
        bank_keywords[bank] = extract_tfidf_keywords(bank_texts, top_n=top_n)
    return bank_keywords


# ---------------------------------------------------------------------------
# 2. Rule-Based Theme Assignment
# ---------------------------------------------------------------------------

def assign_theme(text, theme_keywords=None):
    """
    Assign the best-matching theme to a single review based on keyword overlap.

    Logic:
    - Tokenize the review into words.
    - Count how many keywords from each theme appear in the review.
    - The theme with the highest count wins. Ties broken by order in dict.
    - If no keywords match, assign "Other / General".

    Returns:
        str: theme name
    """
    if theme_keywords is None:
        theme_keywords = THEME_KEYWORDS

    if not isinstance(text, str) or len(text.strip()) == 0:
        return "Other / General"

    words = set(text.lower().split())
    theme_scores = {}

    for theme, keywords in theme_keywords.items():
        score = sum(1 for kw in keywords if kw in words)
        theme_scores[theme] = score

    best_theme = max(theme_scores, key=theme_scores.get)
    if theme_scores[best_theme] == 0:
        return "Other / General"

    return best_theme


def assign_themes_to_df(df, text_col='processed_content', theme_keywords=None):
    """
    Add an 'identified_theme' column to the dataframe.
    Uses the processed (lemmatized) text for matching.
    """
    df['identified_theme'] = df[text_col].apply(
        lambda t: assign_theme(t, theme_keywords)
    )
    return df


# ---------------------------------------------------------------------------
# 3. Per-Bank Theme Summary
# ---------------------------------------------------------------------------

def summarize_themes_per_bank(df, theme_col='identified_theme'):
    """
    Produce a summary table of theme distribution per bank.

    Returns:
        pd.DataFrame with bank, theme, count, percentage
    """
    summary = (
        df.groupby(['bank', theme_col])
        .size()
        .reset_index(name='count')
    )
    # Add percentage within each bank
    bank_totals = df.groupby('bank').size().reset_index(name='bank_total')
    summary = summary.merge(bank_totals, on='bank')
    summary['pct'] = round(summary['count'] / summary['bank_total'] * 100, 1)
    summary = summary.drop(columns='bank_total')
    summary = summary.sort_values(['bank', 'count'], ascending=[True, False])
    return summary


def get_theme_keywords_evidence(bank_keywords, theme_keywords=None):
    """
    For each bank, show which TF-IDF keywords support each theme.
    This provides the 'keyword examples' evidence required.

    Args:
        bank_keywords: dict from extract_keywords_per_bank()

    Returns:
        dict: {bank: {theme: [matching_keywords]}}
    """
    if theme_keywords is None:
        theme_keywords = THEME_KEYWORDS

    evidence = {}
    for bank, kw_list in bank_keywords.items():
        evidence[bank] = {}
        kw_terms = [kw[0] for kw in kw_list]  # just the term strings

        for theme, theme_kws in theme_keywords.items():
            matching = []
            for term in kw_terms:
                # Check if any theme keyword is a substring of the TF-IDF term
                for tkw in theme_kws:
                    if tkw in term:
                        matching.append(term)
                        break
            if matching:
                evidence[bank][theme] = matching

    return evidence


# ---------------------------------------------------------------------------
# 4. Optional: LDA Topic Modeling
# ---------------------------------------------------------------------------

def run_lda_topics(texts, n_topics=5, top_n_words=10, max_features=3000):
    """
    Fit an LDA model for unsupervised topic discovery.

    Returns:
        list of lists: each inner list contains the top words for one topic
        LDA model and vectorizer for further use
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        min_df=3,
        max_df=0.85,
    )
    doc_term_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=20,
        learning_method='online',
    )
    lda.fit(doc_term_matrix)

    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_word_indices = topic.argsort()[::-1][:top_n_words]
        top_words = [feature_names[i] for i in top_word_indices]
        topics.append(top_words)

    return topics, lda, vectorizer
