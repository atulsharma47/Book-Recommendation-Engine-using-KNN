import os
import zipfile
import requests
from io import BytesIO
import pandas as pd
import numpy as np

DATA_URL = "http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip"
RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))

def generate_synthetic_dataset():
    print("\n--- WARNING ---")
    print("Original dataset URL failed. Generating synthetic Book-Crossing dataset")
    print("for demonstration purposes so the ML pipeline can be verified.")
    print("Replace files in data/raw/ with the actual dataset when available.\n")
    
    # 1. Generate Books
    real_books = [
        ("The Hobbit", "J.R.R. Tolkien", 1937, "Allen & Unwin"),
        ("Harry Potter and the Sorcerer's Stone", "J.K. Rowling", 1997, "Scholastic"),
        ("1984", "George Orwell", 1949, "Secker and Warburg"),
        ("To Kill a Mockingbird", "Harper Lee", 1960, "J. B. Lippincott & Co."),
        ("The Great Gatsby", "F. Scott Fitzgerald", 1925, "Scribner"),
        ("Pride and Prejudice", "Jane Austen", 1813, "T. Egerton"),
        ("The Catcher in the Rye", "J.D. Salinger", 1951, "Little, Brown"),
        ("The Lord of the Rings", "J.R.R. Tolkien", 1954, "Allen & Unwin"),
        ("Dune", "Frank Herbert", 1965, "Chilton Books"),
        ("Fahrenheit 451", "Ray Bradbury", 1953, "Ballantine Books"),
        ("The Alchemist", "Paulo Coelho", 1988, "HarperTorch"),
        ("Brave New World", "Aldous Huxley", 1932, "Chatto & Windus"),
        ("The Hunger Games", "Suzanne Collins", 2008, "Scholastic"),
        ("Catch-22", "Joseph Heller", 1961, "Simon & Schuster"),
        ("The Chronicles of Narnia", "C.S. Lewis", 1950, "Geoffrey Bles"),
        ("The Hitchhiker's Guide to the Galaxy", "Douglas Adams", 1979, "Pan Books"),
        ("Ender's Game", "Orson Scott Card", 1985, "Tor Books"),
        ("A Game of Thrones", "George R.R. Martin", 1996, "Bantam Spectra"),
        ("Foundation", "Isaac Asimov", 1951, "Gnome Press"),
        ("The Martian", "Andy Weir", 2011, "Crown Publishing"),
        ("Project Hail Mary", "Andy Weir", 2021, "Ballantine Books"),
        ("The Da Vinci Code", "Dan Brown", 2003, "Doubleday"),
        ("The Silent Patient", "Alex Michaelides", 2019, "Celadon Books")
    ]
    
    num_books = 5000
    isbns = [f"ISBN{str(i).zfill(5)}" for i in range(num_books)]
    
    # Repeat the real books to fill up the initial slots, then use synthetic for the rest
    titles, authors, years, publishers = [], [], [], []
    for i in range(num_books):
        if i < len(real_books):
            b = real_books[i]
        else:
            # Fallback to slightly better synthetic names
            b = (f"Volume {i}: The Mystery of the Data", f"Author {i%50}", 2000 + (i%24), f"Publisher {i%10}")
        titles.append(b[0])
        authors.append(b[1])
        years.append(b[2])
        publishers.append(b[3])
        
    books_df = pd.DataFrame({
        "ISBN": isbns,
        "Book-Title": titles,
        "Book-Author": authors,
        "Year-Of-Publication": years,
        "Publisher": publishers
    })
    books_df.to_csv(os.path.join(RAW_DIR, "BX-Books.csv"), sep=';', index=False)
    
    # 2. Generate Users
    num_users = 1000
    user_ids = np.arange(1, num_users + 1)
    users_df = pd.DataFrame({
        "User-ID": user_ids,
        "Location": ["Location"] * num_users,
        "Age": np.random.randint(15, 80, size=num_users)
    })
    users_df.to_csv(os.path.join(RAW_DIR, "BX-Users.csv"), sep=';', index=False)
    
    # 3. Generate Ratings
    num_ratings = 50000
    ratings_df = pd.DataFrame({
        "User-ID": np.random.choice(user_ids, size=num_ratings),
        "ISBN": np.random.choice(isbns, size=num_ratings),
        "Book-Rating": np.random.randint(0, 11, size=num_ratings)
    })
    ratings_df.drop_duplicates(subset=['User-ID', 'ISBN'], inplace=True)
    ratings_df.to_csv(os.path.join(RAW_DIR, "BX-Book-Ratings.csv"), sep=';', index=False)
    
    print("Synthetic dataset generated successfully.")

def download_and_extract():
    os.makedirs(RAW_DIR, exist_ok=True)
    expected_files = ["BX-Books.csv", "BX-Book-Ratings.csv", "BX-Users.csv"]
    
    if all(os.path.exists(os.path.join(RAW_DIR, f)) for f in expected_files):
        print("Dataset already exists locally. Skipping download.")
        return

    try:
        print(f"Attempting to download dataset from {DATA_URL}...")
        response = requests.get(DATA_URL, timeout=10)
        response.raise_for_status()
        
        print("Extracting zip file...")
        with zipfile.ZipFile(BytesIO(response.content)) as z:
            z.extractall(RAW_DIR)
        print(f"Download and extraction complete. Data saved to {RAW_DIR}")
    except Exception as e:
        print(f"Download failed: {e}")
        generate_synthetic_dataset()

if __name__ == "__main__":
    download_and_extract()
