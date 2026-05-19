import pandas as pd
from google_play_scraper import reviews, Sort
import time


app_names = {
    "CBE": "com.combanketh.mobilebanking",
    "BOA": "com.boa.boaMobileBanking",
    "Dashen": "com.dashen.dashensuperapp"
}

def scrape_fintech_reviews(app_names, count=500, retries=3):
    """
    Scrape Google Play reviews for the given fintech apps.

    Uses a retry + backoff strategy for robustness against transient
    network or API failures.

    Args:
        app_names (dict): Mapping of bank name -> Google Play app ID.
        count (int): Number of reviews to request per app (default 500).
        retries (int): Max retry attempts per app on failure (default 3).

    Returns:
        pd.DataFrame: Consolidated reviews with columns:
            bank, reviewId, reviewText, rating, date, source.
    """
    all_reviews = []
    for bank, app_id in app_names.items():
        success = False
        for attempt in range(retries):
            try:
                res, _ = reviews(app_id, lang='en', country='us', sort=Sort.NEWEST, count=count)
                if not res:
                    raise ValueError(f'Empty response for {bank} (App ID: {app_id})')
                
                for r in res:
                    all_reviews.append({
                        'bank': bank,
                        'reviewId': r['reviewId'],
                        'reviewText': r['content'],
                        'rating': r['score'],
                        'date': r['at'],
                        'source': 'Google Play'
                    })

                # Report date range for documentation purposes
                dates = [r['at'] for r in res]
                min_date = min(dates).strftime('%Y-%m-%d')
                max_date = max(dates).strftime('%Y-%m-%d')
                print(f'  Scraped {len(res)} reviews for {bank} (date range: {min_date} to {max_date})')
                success = True
                break
            except Exception as e:
                print(f'  [Attempt {attempt+1}/{retries}] Failed to scrape {bank}: {e}')
                time.sleep(2) # Backoff
        
        if not success:
            print(f'  CRITICAL: Could not collect data for {bank} after {retries} attempts.')
            
    return pd.DataFrame(all_reviews)
