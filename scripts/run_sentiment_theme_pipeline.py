"""
run_sentiment_theme_pipeline.py
-------------------------------
End-to-end pipeline for Task 2: Sentiment & Thematic Analysis.

Pipeline Steps:
  1. Load preprocessed reviews from data/raw/reviews.csv
  2. Apply text preprocessing (clean, tokenize, lemmatize)
  3. Run DistilBERT sentiment classification
  4. Run VADER sentiment classification (comparison baseline)
  5. Extract TF-IDF keywords per bank
  6. Assign themes to each review
  7. Print aggregated results
  8. Save final CSV with columns:
     review_id, review_text, sentiment_label, sentiment_score, identified_theme

Usage:
  python scripts/run_sentiment_theme_pipeline.py
  python scripts/run_sentiment_theme_pipeline.py --input data/raw/reviews.csv
  python scripts/run_sentiment_theme_pipeline.py --skip-transformer  # VADER only (fast)
"""

import os
import sys
import argparse
import time

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from src.preprocessing import preprocess_dataframe, normalize_schema
from src.sentiment_analysis import (
    TransformerSentimentAnalyzer,
    VADERSentimentAnalyzer,
    aggregate_sentiment_by_bank,
    aggregate_sentiment_by_rating,
    sentiment_coverage,
)
from src.thematic_analysis import (
    extract_keywords_per_bank,
    assign_themes_to_df,
    summarize_themes_per_bank,
    get_theme_keywords_evidence,
    run_lda_topics,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Task 2: Sentiment & Thematic Analysis Pipeline")
    parser.add_argument('--input', type=str, default=None,
                        help='Path to input CSV (default: data/raw/reviews.csv)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to output CSV (default: data/sentiment_themes_results.csv)')
    parser.add_argument('--skip-transformer', action='store_true',
                        help='Skip DistilBERT and use VADER only (faster)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for transformer inference (default: 32)')
    return parser.parse_args()


def main():
    args = parse_args()

    project_root = os.path.join(os.path.dirname(__file__), '..')
    input_path = args.input or os.path.join(project_root, 'data', 'raw', 'reviews.csv')
    output_path = args.output or os.path.join(project_root, 'data', 'sentiment_themes_results.csv')

    print("=" * 60)
    print("  Task 2: Sentiment & Thematic Analysis Pipeline")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Load data
    # ------------------------------------------------------------------
    print(f"\n[1/6] Loading data from {input_path}...")
    if not os.path.exists(input_path):
        print(f"  ERROR: File not found: {input_path}")
        print("  Run 'python scripts/scrape_reviews.py' first.")
        sys.exit(1)

    df = pd.read_csv(input_path)
    df = normalize_schema(df)
    print(f"  Loaded {len(df)} reviews across {df['bank'].nunique()} banks")
    print(f"  Columns: {list(df.columns)}")

    # ------------------------------------------------------------------
    # Step 2: Preprocess text
    # ------------------------------------------------------------------
    print("\n[2/6] Preprocessing text (clean → tokenize → lemmatize)...")
    start = time.time()
    df = preprocess_dataframe(df, text_column='review')
    elapsed = time.time() - start
    print(f"  Preprocessing complete in {elapsed:.1f}s")
    print(f"  Sample processed text: '{df['processed_content'].iloc[0][:80]}...'")

    # ------------------------------------------------------------------
    # Step 3: Sentiment Analysis
    # ------------------------------------------------------------------
    if args.skip_transformer:
        print("\n[3/6] Sentiment Analysis (VADER only — transformer skipped)...")
        vader = VADERSentimentAnalyzer()
        results = vader.predict_batch(df['review'].tolist())
        df['sentiment_label'] = [r[0] for r in results]
        df['sentiment_score'] = [r[1] for r in results]
        print("  VADER analysis complete.")
    else:
        # Primary: DistilBERT
        print("\n[3/6] Sentiment Analysis (DistilBERT transformer)...")
        start = time.time()
        transformer = TransformerSentimentAnalyzer(neutral_threshold=0.70)
        results = transformer.predict_batch(df['review'].tolist(), batch_size=args.batch_size)
        df['sentiment_label'] = [r[0] for r in results]
        df['sentiment_score'] = [r[1] for r in results]
        elapsed = time.time() - start
        print(f"  DistilBERT analysis complete in {elapsed:.1f}s")

        # Comparison: VADER
        print("\n  Running VADER for comparison...")
        vader = VADERSentimentAnalyzer()
        vader_results = vader.predict_batch(df['review'].tolist())
        df['vader_label'] = [r[0] for r in vader_results]
        df['vader_score'] = [r[1] for r in vader_results]

        # Agreement report
        agreement = (df['sentiment_label'] == df['vader_label']).mean() * 100
        print(f"  DistilBERT vs VADER agreement: {agreement:.1f}%")

    # Coverage check (KPI: 90%+ of reviews should have sentiment)
    labeled, total, pct = sentiment_coverage(df)
    status = "✓ KPI MET" if pct >= 90 else "⚠ BELOW 90%"
    print(f"\n  Sentiment Coverage: {labeled}/{total} ({pct}%) {status}")

    # ------------------------------------------------------------------
    # Step 4: Sentiment Aggregation
    # ------------------------------------------------------------------
    print("\n[4/6] Aggregating sentiment scores...")

    print("\n  --- Sentiment by Bank ---")
    bank_agg = aggregate_sentiment_by_bank(df)
    print(bank_agg.to_string(index=False))

    print("\n  --- Mean Sentiment by Bank × Star Rating ---")
    rating_agg = aggregate_sentiment_by_rating(df)
    print(rating_agg.to_string(index=False))

    # ------------------------------------------------------------------
    # Step 5: Thematic Analysis (TF-IDF + Rule-Based Themes)
    # ------------------------------------------------------------------
    print("\n[5/6] Thematic Analysis...")

    # Extract TF-IDF keywords per bank
    print("\n  Extracting TF-IDF keywords per bank...")
    bank_keywords = extract_keywords_per_bank(df, text_col='processed_content', top_n=30)
    for bank, kws in bank_keywords.items():
        top5 = ", ".join([f"{kw[0]} ({kw[1]})" for kw in kws[:5]])
        print(f"    {bank} top-5: {top5}")

    # Assign themes
    print("\n  Assigning themes to reviews...")
    df = assign_themes_to_df(df, text_col='processed_content')

    # Theme summary
    print("\n  --- Theme Distribution per Bank ---")
    theme_summary = summarize_themes_per_bank(df)
    print(theme_summary.to_string(index=False))

    # Keyword evidence for themes
    print("\n  --- Theme Keyword Evidence ---")
    evidence = get_theme_keywords_evidence(bank_keywords)
    for bank, themes in evidence.items():
        print(f"\n  {bank}:")
        for theme, kws in themes.items():
            print(f"    {theme}: {', '.join(kws[:5])}")

    # Count distinct themes per bank (KPI: 3+ per bank)
    print("\n  --- Distinct Themes per Bank ---")
    for bank in df['bank'].unique():
        bank_themes = df[df['bank'] == bank]['identified_theme'].nunique()
        status = "✓ KPI MET" if bank_themes >= 3 else "⚠ Below 3"
        print(f"    {bank}: {bank_themes} themes {status}")

    # ------------------------------------------------------------------
    # Step 6: Optional LDA Topic Modeling
    # ------------------------------------------------------------------
    print("\n  Running LDA topic modeling (5 topics) for validation...")
    all_texts = df['processed_content'].dropna().tolist()
    if len(all_texts) > 50:
        topics, _, _ = run_lda_topics(all_texts, n_topics=5, top_n_words=8)
        for i, topic_words in enumerate(topics):
            print(f"    LDA Topic {i+1}: {', '.join(topic_words)}")
    else:
        print("    Skipped: not enough documents for LDA.")

    # ------------------------------------------------------------------
    # Step 7: Save results
    # ------------------------------------------------------------------
    print(f"\n[6/6] Saving results to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create review_id if not present
    if 'review_id' not in df.columns:
        df.insert(0, 'review_id', range(1, len(df) + 1))

    # Select output columns
    output_cols = ['review_id', 'review', 'bank', 'rating', 'date',
                   'sentiment_label', 'sentiment_score', 'identified_theme']
    # Add VADER columns if they exist
    if 'vader_label' in df.columns:
        output_cols += ['vader_label', 'vader_score']

    output_df = df[[c for c in output_cols if c in df.columns]]
    # Rename 'review' to 'review_text' for the output spec
    output_df = output_df.rename(columns={'review': 'review_text'})
    output_df.to_csv(output_path, index=False)
    print(f"  Saved {len(output_df)} rows, {len(output_df.columns)} columns")
    print(f"  Columns: {list(output_df.columns)}")

    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
