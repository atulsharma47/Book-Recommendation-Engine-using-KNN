from fastapi import APIRouter, HTTPException
import json
import os

router = APIRouter()
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

@router.get("/analytics")
def get_analytics():
    try:
        with open(os.path.join(MODELS_DIR, "metrics.json"), "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Metrics not found. Train model first.")
