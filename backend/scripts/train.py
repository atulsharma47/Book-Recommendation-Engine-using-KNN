import os
import sys
import json

# Add backend directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.data_loader import load_raw_data
from services.preprocessing import clean_data
from services.feature_engineering import create_interaction_matrix, extract_text_features
from services.collaborative_filtering import train_collaborative_model
from services.content_based import train_content_model

def run_training_pipeline():
    print("Loading raw data...")
    books, ratings, users = load_raw_data()
    
    print("Cleaning data...")
    # Using lower thresholds so the synthetic fallback dataset doesn't get completely filtered out
    books_clean, ratings_clean = clean_data(books, ratings, min_user_ratings=10, min_book_ratings=5)
    
    print("Feature Engineering: Interaction Matrix...")
    sparse_matrix, book_indices, user_indices = create_interaction_matrix(ratings_clean)
    
    print("Feature Engineering: Text Features...")
    tfidf_matrix, vectorizer = extract_text_features(books_clean)
    
    print("Training Collaborative Filtering model...")
    train_collaborative_model(sparse_matrix, book_indices, n_neighbors=20)
    
    print("Training Content-Based model...")
    train_content_model(tfidf_matrix, book_indices)
    
    # Save basic analytics
    metrics = {
        "total_books_processed": len(books_clean),
        "total_users_processed": len(user_indices),
        "total_ratings_processed": len(ratings_clean),
        "sparsity": 1.0 - (len(ratings_clean) / float(len(user_indices) * len(book_indices)))
    }
    
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f)
        
    # Also save the cleaned books lookup for the API
    books_clean.to_pickle(os.path.join(models_dir, 'books_clean.pkl'))
        
    print("Training pipeline complete.")

if __name__ == "__main__":
    run_training_pipeline()
