"""
scrape_reviews.py
-----------------
Quick-start script for collecting Google Play reviews from three
Ethiopian fintech apps: CBE, BOA, and Dashen Bank.

Methodology:
  - Uses google-play-scraper to fetch the 500 newest English reviews per app.
  - Deduplicates by reviewId, drops rows missing review text or rating.
  - Normalizes dates to YYYY-MM-DD and saves a clean 5-column CSV.
"""

import os
import sys

# Add project root to path so src/ modules can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_scraping import scrape_fintech_reviews, app_names
from src.preprocessing import normalize_schema, robust_clean


def main():
    print("=" * 50)
    print("  Fintech Review Scraper")
    print("=" * 50)

    # 1. Scrape reviews
    print("\n[1/5] Scraping reviews from Google Play Store...")
    df = scrape_fintech_reviews(app_names, count=500, retries=3)
    print(f"  Total raw reviews collected: {len(df)}")

    # 2. Deduplicate
    before_dedup = len(df)
    df = df.drop_duplicates(subset=['reviewId'], keep='first')
    dupes_removed = before_dedup - len(df)
    print(f"\n[2/5] Removed {dupes_removed} duplicate reviews ({len(df)} remaining)")

    # 3. Normalize schema (reviewText -> review)
    df = normalize_schema(df)

    # 4. Clean data
    print("\n[3/5] Cleaning data...")
    df = robust_clean(df)

    # 5. Select required columns only
    required_columns = ['review', 'rating', 'date', 'bank', 'source']
    df = df[required_columns]

    # 6. Report per-bank counts
    print(f"\n[4/5] Review count per bank:")
    for bank, count in df['bank'].value_counts().items():
        status = "✓" if count >= 400 else "⚠ Below 400 target"
        print(f"  {bank}: {count} {status}")
    print(f"  Total: {len(df)}")

    # 7. Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'reviews.csv')
    df.to_csv(output_path, index=False)
    print(f"\n[5/5] Saved cleaned dataset to {output_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")


if __name__ == '__main__':
    main()
