import os
import pandas as pd

RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))

def load_raw_data():
    """
    Loads the raw CSV files using pandas with appropriate encoding.
    Returns: (books_df, ratings_df, users_df)
    """
    books_path = os.path.join(RAW_DIR, "BX-Books.csv")
    ratings_path = os.path.join(RAW_DIR, "BX-Book-Ratings.csv")
    users_path = os.path.join(RAW_DIR, "BX-Users.csv")

    if not all(os.path.exists(p) for p in [books_path, ratings_path, users_path]):
        raise FileNotFoundError("Raw dataset files not found. Please run scripts/download_dataset.py first.")

    # Load Books
    # The dataset uses latin-1 and has some malformed lines
    books_df = pd.read_csv(
        books_path, 
        sep=';', 
        encoding="latin-1", 
        on_bad_lines='skip',
        low_memory=False
    )
    
    # Load Ratings
    ratings_df = pd.read_csv(
        ratings_path, 
        sep=';', 
        encoding="latin-1"
    )

    # Load Users
    users_df = pd.read_csv(
        users_path, 
        sep=';', 
        encoding="latin-1",
        on_bad_lines='skip'
    )

    return books_df, ratings_df, users_df
