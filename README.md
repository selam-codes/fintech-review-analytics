# Fintech Review Analytics

A comprehensive analytics pipeline designed to ingest, clean, analyze, store, and visualize customer reviews for major Ethiopian fintech mobile applications: **Commercial Bank of Ethiopia (CBE)**, **Bank of Abyssinia (BOA)**, and **Dashen Bank**.

The system implements an end-to-end pipeline covering:
1. **Data Collection & Cleaning**: Automated Google Play Store scraping and robust data preprocessing.
2. **Sentiment & Thematic Classification**: DistilBERT-based transformer model and TF-IDF rule-based theme extraction.
3. **Relational Data Storage**: PostgreSQL database engineering with structured tables and integrity verification.
4. **Insights Generation & Visualizations**: Business analysis of drivers and pain points, with interactive plots and PDF reports.

---

## Project Structure

The project is organized in a modular structure to ensure clean separation of concerns and reproducibility:

```text
.
├── .github/
│   └── workflows/
│       └── unittest.yml         # GitHub Actions CI pipeline configuration
├── data/
│   ├── raw/
│   │   └── reviews.csv          # Scraped raw reviews dataset
│   └── sentiment_themes_results.csv # Output of sentiment & thematic analysis
├── db/
│   └── schema.sql               # PostgreSQL tables (banks, reviews) definitions
├── insights/
│   ├── rating_distribution.png  # Visualizing distribution of review star ratings
│   ├── sentiment_by_bank.png    # Sentiment ratio breakdown per bank
│   ├── sentiment_trend.png      # Monthly average rating trend line plot
│   ├── theme_frequency.png      # Bar chart of identified business themes
│   └── report.md                # Markdown business deep-dive report
├── notebooks/
│   └── task_2_sentiment_thematic_analysis.ipynb # Sentiment & theme exploration notebook
├── scripts/
│   ├── scrape_reviews.py        # Automated scraping and cleaning script
│   ├── run_sentiment_theme_pipeline.py # End-to-end sentiment classification pipeline
│   ├── insert_to_db.py          # PostgreSQL batch insertion & verification script
│   └── generate_insights.py     # Aggregation, plotting, and report generation script
├── src/
│   ├── __init__.py
│   ├── data_scraping.py         # Google Play Store review collection module
│   ├── preprocessing.py         # Tokenization, lemmatization, and schema normalization
│   ├── sentiment_analysis.py    # DistilBERT, VADER, and TextBlob analyzers
│   └── thematic_analysis.py     # TF-IDF keyword extraction and rule-based mapping
├── tests/
│   ├── test_data_scraping.py    # Unit tests for scraper module
│   ├── test_preprocessing.py    # Unit tests for text preprocessing
│   ├── test_sentiment_analysis.py # Unit tests for sentiment prediction
│   └── test_thematic_analysis.py # Unit tests for thematic categorization
├── LICENSE                      # Project license (MIT)
├── README.md                    # Main documentation file (this file)
└── requirements.txt             # Project Python dependencies
```

---

## Detailed Pipeline Phases

### 1. Data Collection & Preprocessing
* **Scraper (`src/data_scraping.py`)**: Uses the `google-play-scraper` package to gather reviews for CBE, BOA, and Dashen Bank. Implements a robust retry-with-backoff strategy to protect against transient API and network failures.
* **Cleaning (`src/preprocessing.py`)**: 
  - Standardizes column schemas.
  - Drops rows with missing reviews or rating values.
  - Normalizes dates to `YYYY-MM-DD` and strips whitespaces.
  - Cleans reviews by lowercase mapping, removing punctuation, and filtering out short reviews.
  - Tokenizes and lemmatizes text using `nltk` word tokenization, removing stop-words and applying `WordNetLemmatizer`.

### 2. Sentiment and Thematic Analysis
* **Sentiment Analysis (`src/sentiment_analysis.py`)**:
  - **Primary Model**: `distilbert-base-uncased-finetuned-sst-2-english` (HuggingFace Transformers pipeline) for fine-grained sentiment analysis.
  - **Neutral Label Heuristic**: Classifies a review as "neutral" if the highest softmax probability score is below a `neutral_threshold` (default `0.70`).
  - **Baselines**: Implements rule-based VADER and pattern-based TextBlob analyzers for baseline comparison and performance validation.
* **Thematic Analysis (`src/thematic_analysis.py`)**:
  - Extracts bank-specific terms and bigrams via `TF-IDF Vectorizer`.
  - Assigns reviews to one of 5 key business themes using keyword matching overlap:
    * *Account Access & Login*
    * *Transaction Performance*
    * *Customer Support*
    * *UI & App Design*
    * *Feature Requests & Bugs*
    * *Other / General* (default fallback)
  - Supports unsupervised theme discovery using Latent Dirichlet Allocation (LDA).

### 3. Database Engineering
* **Schema Design (`db/schema.sql`)**: 
  - Relational layout with a `banks` metadata table and a `reviews` table linked by a foreign key with cascade deletion.
  - Enforces integrity using constraints (`UNIQUE`, `NOT NULL`, primary/foreign keys).
* **Insertion & Verification (`scripts/insert_to_db.py`)**:
  - Establishes connection via `psycopg2`.
  - Seeds the `banks` metadata table.
  - Batch inserts cleaned and analyzed reviews using high-performance `execute_values`.
  - Automatically runs 4 data integrity checks (counts per bank, rating averages, null count audits, and top theme frequencies) upon successful load.

### 4. Insights, Visualizations & Recommendations
* **Insights Script (`scripts/generate_insights.py`)**:
  - Connects to the database and pulls consolidated datasets.
  - Generates matplotlib/seaborn plots: rating distribution, sentiment proportion, monthly trends, and theme frequencies.
  - Compiles `insights/report.md` documenting satisfaction drivers, customer pain points, and prioritized recommendations for each bank.

---

## Installation & Setup

### Prerequisites
- Python 3.10+
- PostgreSQL database instance

### 1. Clone the Repository & Configure Environment
```bash
git clone https://github.com/selam-codes/fintech-review-analytics.git
cd fintech-review-analytics
```

Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Configure Database Variables
Export environment variables for your PostgreSQL instance (used by the database insertion script):
```bash
export DB_NAME="bank_reviews"
export DB_USER="your_postgres_username"
export DB_PASSWORD="your_postgres_password"
export DB_HOST="localhost"
export DB_PORT="5432"
```

---

## Execution Guide

To run the complete data and analysis pipeline step-by-step:

### Step 1: Scrape Reviews
Scrapes raw Google Play Store reviews, cleans them, and saves a preliminary dataset.
```bash
python scripts/scrape_reviews.py
```
*Creates: `data/raw/reviews.csv`*

### Step 2: Run Sentiment & Thematic Pipeline
Processes text and generates sentiment classifications (DistilBERT + VADER comparison) and thematic flags.
```bash
# To run with full DistilBERT (requires PyTorch, takes longer on CPU):
python scripts/run_sentiment_theme_pipeline.py

# To run quickly using VADER sentiment analysis instead:
python scripts/run_sentiment_theme_pipeline.py --skip-transformer
```
*Creates: `data/sentiment_themes_results.csv`*

### Step 3: Insert Data into PostgreSQL
Build the database tables and populate them with the processed dataset.
```bash
# Ensure schema is initialized in your PostgreSQL database first:
psql -d bank_reviews -f db/schema.sql

# Seed tables and insert data:
python scripts/insert_to_db.py
```
*Populates the database and runs validation diagnostics on stdout.*

### Step 4: Generate Insights & Visualizations
Produce analytical charts and the markdown business report from the database content.
```bash
python scripts/generate_insights.py
```
*Saves charts and report under the `insights/` directory.*

---

## Testing & CI/CD

### Running Unit Tests
Unit tests are implemented using `pytest` inside the `tests/` directory. Run them with:
```bash
pytest tests/ -v
```

### GitHub Actions CI
The workflow in `.github/workflows/unittest.yml` triggers on pushes to the `main` branch to check that all tests pass, installing requirements and verifying script dependencies automatically.
