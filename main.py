from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import os
from pathlib import Path
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="RoBERTa Sentiment Analysis API")

MODEL_PATH = "vojmahdal/roberta-sentiment-3labels" 

# Jednoduchý frontend (statické soubory)
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


try:
    print("Loading model, please wait...")
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=MODEL_PATH,
        tokenizer=MODEL_PATH
    )
    print("Model úspěšně načten!")
except Exception as e:
    print(f"Chyba při načítání modelu: {e}")
    sentiment_pipeline = None

# Definice struktury požadavku
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

@app.post("/predict")
async def predict_sentiment(request: TextRequest):
    if sentiment_pipeline is None:
        raise HTTPException(status_code=500, detail="Model is not available.")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    # Samotná predikce
    result = sentiment_pipeline(request.text)[0]
    
    return {
        "text": request.text,
        "label": result["label"],
        "score": round(result["score"], 4)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)