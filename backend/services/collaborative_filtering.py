from sklearn.neighbors import NearestNeighbors
import pickle
import os

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

def train_collaborative_model(sparse_matrix, book_indices, n_neighbors=6):
    """
    Trains a NearestNeighbors model for collaborative filtering.
    """
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=n_neighbors, n_jobs=-1)
    model_knn.fit(sparse_matrix)
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, 'knn_model.pkl'), 'wb') as f:
        pickle.dump(model_knn, f)
    
    with open(os.path.join(MODELS_DIR, 'book_indices.pkl'), 'wb') as f:
        pickle.dump(list(book_indices), f)
        
    return model_knn
