def explain_recommendation(book_id, recommended_book_id, collaborative_score, content_score, books_df):
    """
    Generates data-driven explanations for why a book was recommended.
    """
    explanations = []
    
    if collaborative_score > 0.6:
        explanations.append(f"High similarity in reader rating patterns (Score: {collaborative_score:.2f}).")
    elif collaborative_score > 0.0:
        explanations.append("Similar readers also rated this book.")
        
    if content_score > 0.5:
        explanations.append(f"Strong metadata/content similarity (Score: {content_score:.2f}).")
        
    # Check if books have same author/publisher
    source_book_matches = books_df[books_df['ISBN'] == book_id]
    rec_book_matches = books_df[books_df['ISBN'] == recommended_book_id]
    
    if not source_book_matches.empty and not rec_book_matches.empty:
        source_book = source_book_matches.iloc[0]
        rec_book = rec_book_matches.iloc[0]
        
        if source_book['Book-Author'] == rec_book['Book-Author']:
            explanations.append("Written by the same author.")
        if source_book['Publisher'] == rec_book['Publisher']:
            explanations.append("Published by the same publisher.")
            
    if not explanations:
        explanations.append("Recommended by the hybrid engine.")
        
    return explanations
