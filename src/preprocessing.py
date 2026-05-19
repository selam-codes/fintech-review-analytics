import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string

# Download necessary NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

def clean_text(text):
  
    if not isinstance(text, str):
        if pd.isna(text):
            return ""
        text = str(text) # Force string conversion for numeric data
    
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\d+", "", text)
    return text

def tokenize_and_lemmatize(text):
    """
    Tokenizes text, removes stopwords, and applies lemmatization.
    Matches the notebook's modular_nlp_pipeline logic.
    """
    # 1. Cleaning (Regex)
    text = re.sub(r'[^a-zA-Z\s]', ' ', str(text).lower())

    # 2. Tokenization
    tokens = nltk.word_tokenize(text)

    # 3. Stop-word removal & Lemmatization
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    processed = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2]
    return " ".join(processed)

def normalize_schema(df):
    """
    Rename common alternative column names to the expected schema:
    - reviewText, content, text -> review
    - Bank -> bank
    - score -> rating
    - at -> date
    """
    mapping = {}
    if 'reviewText' in df.columns and 'review' not in df.columns:
        mapping['reviewText'] = 'review'
    if 'content' in df.columns and 'review' not in df.columns:
        mapping['content'] = 'review'
    if 'text' in df.columns and 'review' not in df.columns:
        mapping['text'] = 'review'
    if 'Bank' in df.columns and 'bank' not in df.columns:
        mapping['Bank'] = 'bank'
    if 'score' in df.columns and 'rating' not in df.columns:
        mapping['score'] = 'rating'
    if 'at' in df.columns and 'date' not in df.columns:
        mapping['at'] = 'date'
    if mapping:
        df = df.rename(columns=mapping)
    return df

def preprocess_dataframe(df, text_column='review'):
    """
    Applies the full preprocessing pipeline to a dataframe.
    ERROR HANDLING: Row-level exception handling to prevent pipeline crashes.
    """
    try:
        df['clean_text'] = df[text_column].apply(clean_text)
        df['processed_content'] = df['clean_text'].apply(tokenize_and_lemmatize)
        return df
    except Exception as e:
        print(f" ERROR: Preprocessing failed on column '{text_column}': {e}")
        return df

def robust_clean(df):
    """
    Data integrity checks: schema validation, type enforcement,
    date normalization, and missing-value handling.

    Logs counts of dropped rows at each step for documentation.
    """
    try:
        initial_count = len(df)

        # 1. Strip Whitespaces & Trailing Zeros
        for col in df.select_dtypes(include=['object']):
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].str.replace(r'\.0$', '', regex=True)

        # 2. Date Normalisation
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.strftime('%Y-%m-%d')

        # 3. Handle Missing Values
        null_count = df.isnull().sum().sum()
        print(f"  Null values found: {null_count}")
        before_drop = len(df)
        df = df.dropna(subset=['review', 'rating'])
        dropped_nulls = before_drop - len(df)
        print(f"  Dropped {dropped_nulls} rows with missing review/rating")

        # 4. Filter empty/short reviews
        before_filter = len(df)
        df = df[df['review'].str.len() > 2]
        dropped_short = before_filter - len(df)
        print(f"  Dropped {dropped_short} rows with review length ≤ 2 chars")

        total_dropped = initial_count - len(df)
        missing_pct = (total_dropped / initial_count * 100) if initial_count > 0 else 0
        print(f"  Total rows dropped: {total_dropped}/{initial_count} ({missing_pct:.1f}%)")
        print(f"  Final dataset shape: {df.shape}")
        return df.reset_index(drop=True)
    except Exception as e:
        print(f"  ERROR: Robust cleaning failed: {e}")
        return df
