from fastapi import APIRouter, HTTPException
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
from services.explainability import explain_recommendation
from services.hybrid_recommender import generate_hybrid_recommendations

router = APIRouter()
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

model_knn = None
book_indices = None
tfidf_matrix = None
books_df = None

def load_models():
    global model_knn, book_indices, tfidf_matrix, books_df
    try:
        if model_knn is None:
            with open(os.path.join(MODELS_DIR, "knn_model.pkl"), "rb") as f:
                model_knn = pickle.load(f)
            with open(os.path.join(MODELS_DIR, "book_indices.pkl"), "rb") as f:
                book_indices = pickle.load(f)
            with open(os.path.join(MODELS_DIR, "tfidf_matrix.pkl"), "rb") as f:
                tfidf_matrix = pickle.load(f)
            with open(os.path.join(MODELS_DIR, "books_clean.pkl"), "rb") as f:
                books_df = pickle.load(f)
    except FileNotFoundError:
        pass

@router.get("/recommend/hybrid/{book_id}")
def recommend_hybrid(book_id: str):
    load_models()
    if model_knn is None or book_indices is None or tfidf_matrix is None or books_df is None:
        raise HTTPException(status_code=503, detail="Models not trained")
        
    try:
        book_idx = book_indices.index(book_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Book not found in training set")
        
    distances, indices = model_knn.kneighbors(model_knn._fit_X[book_idx].reshape(1, -1), n_neighbors=11)
    
    collab_scores = {}
    for d, idx in zip(distances.flatten()[1:], indices.flatten()[1:]):
        isbn = book_indices[idx]
        collab_scores[isbn] = 1.0 - d
        
    cosine_sim = cosine_similarity(tfidf_matrix[book_idx], tfidf_matrix).flatten()
    content_indices = cosine_sim.argsort()[-11:-1][::-1]
    
    content_scores = {}
    for idx in content_indices:
        isbn = book_indices[idx]
        content_scores[isbn] = cosine_sim[idx]
        
    hybrid_recs = generate_hybrid_recommendations(collab_scores, content_scores)
    
    response = []
    for rec_isbn, score in hybrid_recs[:5]:
        rec_book_info = books_df[books_df["ISBN"] == rec_isbn].iloc[0].to_dict()
        c_score = collab_scores.get(rec_isbn, 0.0)
        t_score = content_scores.get(rec_isbn, 0.0)
        explanations = explain_recommendation(book_id, rec_isbn, c_score, t_score, books_df)
        
        response.append({
            "book": rec_book_info,
            "similarity_score": score,
            "collaborative_score": c_score,
            "content_score": t_score,
            "reasons": explanations
        })
        
    return {
        "source_book": books_df[books_df["ISBN"] == book_id].iloc[0].to_dict(),
        "recommendations": response
    }
