from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import routes_books, routes_recommendations, routes_analytics

app = FastAPI(title="BookWise AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "healthy"}

app.include_router(routes_books.router, prefix="/api/v1")
app.include_router(routes_recommendations.router, prefix="/api/v1")
app.include_router(routes_analytics.router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
