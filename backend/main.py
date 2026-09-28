from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from inference import predict_sentiment

app = FastAPI(title="SentimentAI API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class PredictRequest(BaseModel):
    text: str

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_ready": True,
        "mode": "trained"
    }

@app.get("/model-info")
def model_info():
    return {
        "name": "Simple RNN Sentiment Classifier",
        "framework": "PyTorch",
        "classes": ["negative", "neutral", "positive"],
        "embedding": "GloVe 50D",
        "mode": "trained"
    }

@app.post("/predict")
def predict(req: PredictRequest):
    text = req.text.strip()

    if len(text) < 2:
        raise HTTPException(
            status_code=400,
            detail="Please enter a longer text."
        )

    label, scores = predict_sentiment(text)

    confidence = max(scores.values()) * 100

    return {
        "text": text,
        "sentiment": label,
        "confidence": round(confidence, 1),
        "scores": scores,
        "model_mode": "trained"
    }