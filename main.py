from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from transformers import pipeline
from pathlib import Path
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from db import log_to_db, get_logs_from_db

app = FastAPI(title="RoBERTa Sentiment Analysis API")

MODEL_PATH = "vojmahdal/roberta-sentiment-3labels" 

# Simple frontend (static files)
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


#
# --- Simple SQLite "database" for logging/anonymized storage ---
#

try:
    print("Loading model, please wait...")
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=MODEL_PATH,
        tokenizer=MODEL_PATH
    )
    print("Model successfully loaded!")
except Exception as e:
    print(f"Error loading model: {e}")
    sentiment_pipeline = None

# Definition of the request structure
class TextRequest(BaseModel):
    text: str

@app.get("/")
def home():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "Sentiment API is running. Use POST on /predict."}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": sentiment_pipeline is not None}


@app.get("/logs")
def get_logs(limit: int = 100):
    """
    Return last `limit` anonymized inputs from SQLite.
    Only anonymized_text + metadata, nikdy ne původní text.
    """
    return get_logs_from_db(limit=limit)

@app.post("/predict")
async def predict_sentiment(payload: TextRequest, request: Request):
    if sentiment_pipeline is None:
        raise HTTPException(status_code=500, detail="Model is not available.")
    
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    # Prediction
    result = sentiment_pipeline(payload.text)[0]

    # Try to log anonymized version into SQLite "database"
    try:
        client_ip = request.client.host if request.client else None
        log_to_db(
            original_text=payload.text,
            label=result["label"],
            score=float(result["score"]),
            client_ip=client_ip,
        )
    except Exception as e:
        # Logging failure must not break the API
        print(f"Logging to DB failed: {e}")
    
    return {
        "text": payload.text,
        "label": result["label"],
        "score": round(result["score"], 4)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)