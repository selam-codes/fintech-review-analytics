# Fintech Review Analytics

A data analysis project that scrapes and analyzes financial app reviews from the Google Play Store.

## Project Overview

This project collects reviews from three major Ethiopian fintech applications:
- **CBE** (Commercial Bank of Ethiopia) - `com.combanketh.mobilebanking`
- **BOA** (Bank of Abyssinia) - `com.boa.boaMobileBanking`
- **Dashen** (Dashen Bank) - `com.dashen.dashensuperapp`

The reviews are processed, cleaned, and analyzed to extract insights about user sentiment and experiences.

## Features

- **Review Scraping**: Automated collection of up to 500 latest reviews per app using `google-play-scraper`
- **Data Cleaning**: Removes duplicates, handles missing values, and standardizes date formats
- **Review Analysis**: Analyzes ratings, sentiment, and user feedback across fintech applications
- **NLP Processing**: Leverages transformer models for advanced text analysis

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies

- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `matplotlib` & `seaborn` - Data visualization
- `google-play-scraper` - Google Play Store scraping
- `transformers` & `torch` - NLP and deep learning
- `nltk` - Natural language processing
- `scikit-learn` - Machine learning utilities
- `jupyter` - Interactive notebooks
- `pytest` - Unit testing

## Scraping Methodology

### Library & Parameters

Reviews are scraped from the Google Play Store using the [`google-play-scraper`](https://pypi.org/project/google-play-scraper/) Python library with the following configuration:

| Parameter | Value |
|---|---|
| Sort order | `Sort.NEWEST` (most recent first) |
| Language | `en` (English) |
| Country | `us` |
| Count per app | 500 (target: 400+ per bank, 1,200+ total) |
| Retries | 3 attempts per app with 2s backoff |

### Date Range

The scraper collects the **500 most recent** English-language reviews available at the time of execution. The actual date range depends on when the script is run and review volume for each app. Date ranges are logged during scraping for documentation.

### Usage

```bash
python scripts/scrape_reviews.py
```

The script uses the modular `src/data_scraping.py` and `src/preprocessing.py` modules to:
1. Scrape 500 newest reviews per app with retry logic
2. Deduplicate by `reviewId` (logs count of duplicates removed)
3. Normalize column names (`reviewText` → `review`)
4. Drop rows missing `review` or `rating` (logs counts)
5. Filter reviews with ≤ 2 characters
6. Normalize dates to `YYYY-MM-DD` format
7. Save the cleaned dataset to `data/raw/reviews.csv`

### Output

The final CSV contains exactly **5 columns**:

| Column | Description |
|---|---|
| `review` | Review text content |
| `rating` | User rating (1–5) |
| `date` | Review date (`YYYY-MM-DD`) |
| `bank` | Bank name (`CBE`, `BOA`, `Dashen`) |
| `source` | Always `Google Play` |

> **Note:** The CSV is listed in `.gitignore` and is never committed to the repository.

### Known Limitations

- `google-play-scraper` relies on undocumented Google Play endpoints; availability may change without notice.
- The library does not support pagination beyond the initial `count` parameter — if fewer than 400 reviews exist in English for an app, the target cannot be met without changing the `lang` parameter.
- Reviews are filtered to English (`lang='en'`) only; Amharic or other local-language reviews are excluded from this dataset.
- Scraping results may vary between runs due to Google Play's internal review ordering.

## Source Modules (`src/`)

### `data_scraping.py`

Modular scraping logic with built-in error handling.

- **`scrape_fintech_reviews(app_names, count=500, retries=3)`** — Fetches Google Play reviews for each configured app with a **retry + exponential backoff** strategy. Returns a consolidated `pandas.DataFrame` with columns: `bank`, `reviewId`, `reviewText`, `rating`, `date`, `source`.

### `preprocessing.py`

Full NLP preprocessing pipeline for review text.

| Function | Description |
|---|---|
| `clean_text(text)` | Lowercases, strips punctuation, and removes digits from raw review text. Handles `NaN` and non-string inputs gracefully. |
| `tokenize_and_lemmatize(text)` | Tokenizes text, removes English stopwords, and applies WordNet lemmatization. Returns a cleaned, space-joined string. |
| `normalize_schema(df)` | Maps common column name variants (`reviewText`, `content`, `score`, `at`, `Bank`) to the expected schema (`review`, `rating`, `date`, `bank`). |
| `preprocess_dataframe(df, text_column='review')` | Applies the full pipeline (`clean_text` → `tokenize_and_lemmatize`) to a DataFrame, adding `clean_text` and `processed_content` columns. Includes row-level error handling. |
| `robust_clean(df)` | Validates data integrity: strips whitespace, normalizes dates, drops rows with missing `review`/`rating`, and filters out very short reviews (≤ 2 chars). |

## Project Structure

```
.
├── data/
│   └── raw/                    # Raw scraped data
├── notebooks/                  # Jupyter notebooks for analysis
├── scripts/
│   └── scrape_reviews.py       # Quick-start scraping script
├── src/
│   ├── __init__.py
│   ├── data_scraping.py        # Modular scraper with retry logic
│   └── preprocessing.py        # NLP preprocessing pipeline
├── tests/
│   ├── test_data_scraping.py   # Scraper unit tests (mocked)
│   └── test_preprocessing.py   # Preprocessing unit tests
├── requirements.txt            # Project dependencies
└── README.md
```

## Testing

Run unit tests using pytest:

```bash
pytest
```

## CI/CD

This project uses GitHub Actions for continuous integration. Tests run automatically on pushes to the `main` branch. See `.github/workflows/unittest.yml` for the workflow configuration.

## Relational Database Setup (PostgreSQL)

This section details the design, setup, and validation of the PostgreSQL database used to persistently store preprocessed reviews.

### Database Creation
To initialize the database locally, execute:
```bash
psql -d postgres -c "CREATE DATABASE bank_reviews;"
```

### Schema Design
The relational schema comprises two normalized tables: `banks` and `reviews`. This layout avoids data redundancy by separating bank metadata from individual reviews.

The full SQL script is stored in [`db/schema.sql`](file:///Users/selam/Desktop/Ten%20Academy%20AI/week%202/fintech-review-analytics/db/schema.sql):

```sql
-- Seeded metadata table
CREATE TABLE banks (
    bank_id SERIAL PRIMARY KEY,
    bank_name VARCHAR(100) UNIQUE NOT NULL,
    app_name VARCHAR(100) NOT NULL
);

-- Primary reviews storage
CREATE TABLE reviews (
    review_id SERIAL PRIMARY KEY,
    bank_id INT REFERENCES banks(bank_id) ON DELETE CASCADE,
    review_text TEXT NOT NULL,
    rating INT NOT NULL,
    review_date DATE NOT NULL,
    sentiment_label VARCHAR(20),
    sentiment_score NUMERIC(5, 4),
    identified_theme VARCHAR(100),
    source VARCHAR(50) NOT NULL
);
```

Apply this schema using:
```bash
psql -d bank_reviews -f db/schema.sql
```

### Python Data Insertion
The Python script [`scripts/insert_to_db.py`](file:///Users/selam/Desktop/Ten%20Academy%20AI/week%202/fintech-review-analytics/scripts/insert_to_db.py) automates:
1. Seeding of bank metadata (`banks` table).
2. Clean loading and handling of local socket/TCP connections with zero password prompts.
3. Batch insertion of 1,458 reviews (well above the 400 target!) using PostgreSQL fast batch ingestion (`execute_values`).

To run insertion:
```bash
python scripts/insert_to_db.py
```

### Data Integrity & Verification Results
Every time `insert_to_db.py` is run, it executes validation queries to verify data integrity. The results on local run are:

#### 1. Count Reviews per Bank
- **Bank of Abyssinia**: 492 reviews
- **Dashen Bank**: 489 reviews
- **Commercial Bank of Ethiopia**: 477 reviews
- **Total**: 1,458 reviews (exceeding targets!)

#### 2. Average Rating per Bank
- **Commercial Bank of Ethiopia**: 4.06 ★
- **Dashen Bank**: 3.91 ★
- **Bank of Abyssinia**: 3.54 ★

#### 3. Key Columns Null Check
- **Missing Review Text**: 0
- **Missing Ratings**: 0
- **Missing Dates**: 0
- **Missing Bank Association**: 0

This validates 100% relational integrity and successful database seeding!

## Insights & Visualizations (Task 4)

This section details **Task 4: Insights and Recommendations**, including satisfaction drivers, pain points, and prioritized product recommendations.

### Execution
To regenerate all visualizations and the comprehensive business insights report directly from the database:
```bash
python scripts/generate_insights.py
```

### Generated Visualizations (inside [`insights/`](file:///Users/selam/Desktop/Ten%20Academy%20AI/week%202/fintech-review-analytics/insights/))
1.  **`sentiment_by_bank.png`**: Premium stacked bar chart demonstrating that CBE has the highest positive sentiment ratio while BOA has the highest negative ratio.
2.  **`rating_distribution.png`**: Grouped star-rating percentage breakdown showing 5-star peaks for CBE/Dashen and heavy 1-star peaks for BOA.
3.  **`theme_frequency.png`**: Horizontal bar plot highlighting that "UI & App Design" and "Other / General" are the highest occurring themes across all banks.
4.  **`sentiment_trend.png`**: Line plot detailing the monthly average star-rating trajectory over a 14-month window (March 2025 – May 2026).

### Synthesis & Recommendations
The full strategic synthesis is saved as a markdown document in [**`insights/report.md`**](file:///Users/selam/Desktop/Ten%20Academy%20AI/week%202/fintech-review-analytics/insights/report.md). It outlines:
- **Commercial Bank of Ethiopia (CBE)**: Actionable drivers (UI design, support) vs. pain points (transaction lag, update loops) and product fixes.
- **Bank of Abyssinia (BOA)**: CRITICAL launch crashes on newer Android systems, and device activation fixes.
- **Dashen Bank**: "Fayda" (National ID) onboarding failures, virtual account failures, and foreign currency transparent marketing.



