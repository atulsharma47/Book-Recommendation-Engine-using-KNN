import pandas as pd
import numpy as np

def generate_hybrid_recommendations(collaborative_scores, content_scores, collab_weight=0.7, content_weight=0.3):
    """
    Combines collaborative and content-based recommendation scores using a weighted sum.
    Both input dictionaries should map book ISBNs to similarity scores (0.0 to 1.0).
    Returns a sorted list of tuples (ISBN, combined_score).
    """
    all_books = set(collaborative_scores.keys()).union(set(content_scores.keys()))
    
    hybrid_results = []
    for book in all_books:
        c_score = collaborative_scores.get(book, 0.0)
        t_score = content_scores.get(book, 0.0)
        
        final_score = (c_score * collab_weight) + (t_score * content_weight)
        hybrid_results.append((book, final_score))
        
    hybrid_results.sort(key=lambda x: x[1], reverse=True)
    return hybrid_results
