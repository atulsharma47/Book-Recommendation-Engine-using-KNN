import pickle
import os

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

def train_content_model(tfidf_matrix, book_indices):
    """
    Saves the TF-IDF matrix for content-based similarity lookups.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, 'tfidf_matrix.pkl'), 'wb') as f:
        pickle.dump(tfidf_matrix, f)
        
    return tfidf_matrix
