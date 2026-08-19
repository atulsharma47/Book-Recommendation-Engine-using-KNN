from fastapi import APIRouter, HTTPException
import pickle
import os
import json

router = APIRouter()
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

books_df = None

def load_books():
    global books_df
    try:
        if books_df is None:
            with open(os.path.join(MODELS_DIR, "books_clean.pkl"), "rb") as f:
                books_df = pickle.load(f)
    except FileNotFoundError:
        pass

@router.get("/books")
def get_books(limit: int = 50):
    load_books()
    if books_df is None:
        raise HTTPException(status_code=503, detail="Models not trained yet")
    return books_df.head(limit).to_dict(orient="records")

@router.get("/books/{book_id}")
def get_book(book_id: str):
    load_books()
    if books_df is None:
        raise HTTPException(status_code=503, detail="Models not trained yet")
    book = books_df[books_df["ISBN"] == book_id]
    if book.empty:
        raise HTTPException(status_code=404, detail="Book not found")
    return book.iloc[0].to_dict()

@router.get("/search")
def search_books(q: str):
    load_books()
    if books_df is None:
        raise HTTPException(status_code=503, detail="Models not trained yet")
    results = books_df[books_df["Book-Title"].str.contains(q, case=False, na=False)]
    return results.head(20).to_dict(orient="records")
