import pandas as pd

def clean_data(books_df, ratings_df, min_user_ratings=100, min_book_ratings=50):
    """
    Cleans the datasets:
    - Removes missing values
    - Handles invalid ratings
    - Filters out rare users and books based on configurable thresholds
    """
    print(f"Initial ratings count: {len(ratings_df)}")
    
    # Clean Books: keep essential columns, drop NaNs in Title or Author
    books_clean = books_df[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher']].copy()
    books_clean.dropna(subset=['Book-Title', 'Book-Author'], inplace=True)
    
    # Clean Ratings
    ratings_clean = ratings_df.copy()
    
    # Keep only ratings for books that actually exist in the books dataset
    ratings_clean = ratings_clean[ratings_clean['ISBN'].isin(books_clean['ISBN'])]
    
    # Remove duplicates if any user rated the same book twice
    ratings_clean = ratings_clean.drop_duplicates(subset=['User-ID', 'ISBN'])
    
    # Filter rare users
    user_counts = ratings_clean['User-ID'].value_counts()
    active_users = user_counts[user_counts >= min_user_ratings].index
    ratings_clean = ratings_clean[ratings_clean['User-ID'].isin(active_users)]
    
    # Filter rare books
    book_counts = ratings_clean['ISBN'].value_counts()
    popular_books = book_counts[book_counts >= min_book_ratings].index
    ratings_clean = ratings_clean[ratings_clean['ISBN'].isin(popular_books)]
    
    # Keep only books that survived the rating filter
    books_clean = books_clean[books_clean['ISBN'].isin(ratings_clean['ISBN'])]
    
    print(f"Cleaned ratings count: {len(ratings_clean)}")
    print(f"Cleaned books count: {len(books_clean)}")
    
    return books_clean, ratings_clean
