"""
insert_to_db.py
---------------
Inserts processed review data from data/sentiment_themes_results.csv into the
PostgreSQL database 'bank_reviews'. Runs validation queries to ensure data integrity.

Usage:
  python scripts/insert_to_db.py
"""

import os
import sys
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

# Define default database connection parameters
DB_NAME = os.environ.get("DB_NAME", "bank_reviews")
DB_USER = os.environ.get("DB_USER", "selam")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")
DB_HOST = os.environ.get("DB_HOST", "")
DB_PORT = os.environ.get("DB_PORT", "")

# Bank metadata to seed
BANKS_METADATA = [
    {"bank_name": "Commercial Bank of Ethiopia", "app_name": "com.combanketh.mobilebanking", "short": "CBE"},
    {"bank_name": "Bank of Abyssinia", "app_name": "com.boa.boaMobileBanking", "short": "BOA"},
    {"bank_name": "Dashen Bank", "app_name": "com.dashen.dashensuperapp", "short": "Dashen"}
]

def get_connection():
    """Establishes and returns a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        return conn
    except Exception as e:
        print(f"Error connecting to PostgreSQL database: {e}")
        sys.exit(1)

def seed_banks(conn):
    """Seeds the banks table and returns a mapping of short name -> bank_id."""
    cursor = conn.cursor()
    bank_mapping = {}
    
    print("\n[1/4] Seeding 'banks' metadata table...")
    
    insert_query = """
        INSERT INTO banks (bank_name, app_name) 
        VALUES (%s, %s)
        ON CONFLICT (bank_name) 
        DO UPDATE SET app_name = EXCLUDED.app_name
        RETURNING bank_id;
    """
    
    try:
        for bank in BANKS_METADATA:
            cursor.execute(insert_query, (bank["bank_name"], bank["app_name"]))
            bank_id = cursor.fetchone()[0]
            bank_mapping[bank["short"]] = bank_id
            print(f"  Bank '{bank['bank_name']}' seeded with ID {bank_id}")
        
        conn.commit()
        print("  Banks seeded successfully.")
        return bank_mapping
    except Exception as e:
        conn.rollback()
        print(f"  ERROR seeding banks: {e}")
        sys.exit(1)
    finally:
        cursor.close()

def insert_reviews(conn, bank_mapping, df_path):
    """Loads CSV review data, cleans it, and inserts it in batches into the database."""
    if not os.path.exists(df_path):
        print(f"  ERROR: Processed review CSV not found at {df_path}")
        print("  Please run the sentiment pipeline first: python scripts/run_sentiment_theme_pipeline.py --skip-transformer")
        sys.exit(1)

    print(f"\n[2/4] Loading processed reviews from {df_path}...")
    df = pd.read_csv(df_path)
    print(f"  Loaded {len(df)} rows from CSV.")

    # Data cleaning and normalization for database insertion
    # Replace NaN or pd.NA with None so psycopg2 inserts NULL
    df = df.where(pd.notnull(df), None)

    # Prepare insert tuples
    reviews_to_insert = []
    skipped_count = 0

    for idx, row in df.iterrows():
        short_bank_name = row.get("bank")
        bank_id = bank_mapping.get(short_bank_name)
        
        if not bank_id:
            print(f"  WARNING: Unknown bank name '{short_bank_name}' in row {idx}. Skipping.")
            skipped_count += 1
            continue
            
        review_text = row.get("review_text")
        rating = int(row.get("rating")) if row.get("rating") is not None else None
        review_date = row.get("date")
        sentiment_label = row.get("sentiment_label")
        sentiment_score = float(row.get("sentiment_score")) if row.get("sentiment_score") is not None else None
        identified_theme = row.get("identified_theme")
        source = row.get("source", "Google Play") # Fallback to Google Play if not present

        # Enforce that key columns are not null
        if not review_text or rating is None or not review_date:
            skipped_count += 1
            continue

        reviews_to_insert.append((
            bank_id,
            review_text,
            rating,
            review_date,
            sentiment_label,
            sentiment_score,
            identified_theme,
            source
        ))

    print(f"  Prepared {len(reviews_to_insert)} records for database insertion (Skipped {skipped_count} invalid records).")

    print("\n[3/4] Inserting reviews into database in batch...")
    cursor = conn.cursor()
    
    insert_query = """
        INSERT INTO reviews (
            bank_id, review_text, rating, review_date, 
            sentiment_label, sentiment_score, identified_theme, source
        )
        VALUES %s;
    """

    try:
        # Clear existing reviews first to avoid growing the table indefinitely on run
        cursor.execute("TRUNCATE TABLE reviews RESTART IDENTITY;")
        
        # Batch insert using execute_values for high performance
        execute_values(cursor, insert_query, reviews_to_insert)
        conn.commit()
        print(f"  Successfully inserted {len(reviews_to_insert)} reviews into database.")
    except Exception as e:
        conn.rollback()
        print(f"  ERROR inserting reviews: {e}")
        sys.exit(1)
    finally:
        cursor.close()

def run_integrity_queries(conn):
    """Executes data integrity verification queries and prints the results."""
    print("\n[4/4] Running SQL Verification Queries...")
    cursor = conn.cursor()

    queries = {
        "1. Count Reviews per Bank": """
            SELECT b.bank_name, COUNT(r.review_id) AS total_reviews
            FROM reviews r
            JOIN banks b ON r.bank_id = b.bank_id
            GROUP BY b.bank_name
            ORDER BY total_reviews DESC;
        """,
        "2. Average Rating per Bank": """
            SELECT b.bank_name, ROUND(AVG(r.rating), 2) AS avg_rating, COUNT(r.review_id) AS count
            FROM reviews r
            JOIN banks b ON r.bank_id = b.bank_id
            GROUP BY b.bank_name
            ORDER BY avg_rating DESC;
        """,
        "3. Nulls in Key Columns": """
            SELECT 
                COUNT(*) FILTER (WHERE review_text IS NULL) AS null_reviews,
                COUNT(*) FILTER (WHERE rating IS NULL) AS null_ratings,
                COUNT(*) FILTER (WHERE review_date IS NULL) AS null_dates,
                COUNT(*) FILTER (WHERE bank_id IS NULL) AS null_bank_ids
            FROM reviews;
        """,
        "4. Theme Distribution (Top 5)": """
            SELECT identified_theme, COUNT(*) AS count
            FROM reviews
            WHERE identified_theme IS NOT NULL
            GROUP BY identified_theme
            ORDER BY count DESC
            LIMIT 5;
        """
    }

    try:
        for title, sql in queries.items():
            print(f"\n  --- {title} ---")
            cursor.execute(sql)
            colnames = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            
            # Print column headers
            header = " | ".join(colnames)
            print(f"    {header}")
            print(f"    {'-' * len(header)}")
            
            # Print rows
            for row in rows:
                print(f"    {' | '.join(str(val) for val in row)}")
    except Exception as e:
        print(f"  ERROR executing integrity queries: {e}")
    finally:
        cursor.close()

def main():
    project_root = os.path.join(os.path.dirname(__file__), "..")
    df_path = os.path.join(project_root, "data", "sentiment_themes_results.csv")
    
    print("=" * 60)
    print("  Task 3: PostgreSQL Data Storage & Verification")
    print("=" * 60)
    
    # Establish connection
    conn = get_connection()
    
    try:
        # 1. Seed Banks Table
        bank_mapping = seed_banks(conn)
        
        # 2. Load & Insert Reviews Data
        insert_reviews(conn, bank_mapping, df_path)
        
        # 3. Run Validation and Integrity Checks
        run_integrity_queries(conn)
        
    finally:
        conn.close()
        print("\n" + "=" * 60)
        print("  Database Insertion Process Completed!")
        print("=" * 60)

if __name__ == "__main__":
    main()
