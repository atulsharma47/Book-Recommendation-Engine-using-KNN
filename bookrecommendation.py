import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Load the dataset
books = pd.read_csv('BX-Books.csv', sep=';', error_bad_lines=False, encoding="latin-1")
ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
# Remove users with less than 200 ratings and books with less than 100 ratings
user_counts = ratings['User-ID'].value_counts()
rating_counts = ratings['ISBN'].value_counts()

ratings = ratings[ratings['User-ID'].isin(user_counts[user_counts >= 200].index)]
ratings = ratings[ratings['ISBN'].isin(rating_counts[rating_counts >= 100].index)]

# Merge books and ratings based on ISBN
merged_df = pd.merge(ratings, books, on='ISBN')

# Create a pivot table of users and their ratings for books
pivot_df = merged_df.pivot(index='Book-Title', columns='User-ID', values='Book-Rating').fillna(0)
# Convert pivot table to sparse matrix
pivot_matrix = csr_matrix(pivot_df.values)

# Initialize the NearestNeighbors model using cosine similarity
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=6, n_jobs=-1)
model_knn.fit(pivot_matrix)
def get_recommends(book_title):
    # Find the index of the book title in the pivot table
    book_idx = list(pivot_df.index).index(book_title)
    
    # Find the distances and indices of the nearest neighbors
    distances, indices = model_knn.kneighbors(pivot_df.iloc[book_idx, :].values.reshape(1, -1), n_neighbors=6)
    
    recommended_books = []
    for i in range(1, len(distances.flatten())):
        recommended_books.append([pivot_df.index[indices.flatten()[i]], distances.flatten()[i]])
    
    return [book_title, recommended_books]
# Test the function with a specific book title
get_recommends("The Queen of the Damned (Vampire Chronicles (Paperback))")
