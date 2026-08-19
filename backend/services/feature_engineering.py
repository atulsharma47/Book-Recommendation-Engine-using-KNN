import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

def create_interaction_matrix(ratings_clean):
    """
    Creates a sparse user-book interaction matrix using SciPy.
    Returns the sparse matrix, book indices (rows), and user indices (columns).
    """
    # Create pivot table
    pivot_df = ratings_clean.pivot(index='ISBN', columns='User-ID', values='Book-Rating').fillna(0)
    
    # Convert to sparse matrix
    sparse_matrix = csr_matrix(pivot_df.values)
    
    return sparse_matrix, pivot_df.index, pivot_df.columns

def extract_text_features(books_clean):
    """
    Extracts text features (TF-IDF) from book metadata (Title, Author, Publisher).
    Returns the TF-IDF sparse matrix and the vectorizer.
    """
    # Combine metadata into a single string per book
    metadata = books_clean['Book-Title'].astype(str) + " " + \
               books_clean['Book-Author'].astype(str) + " " + \
               books_clean['Publisher'].astype(str)
                              
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(metadata)
    
    return tfidf_matrix, vectorizer
