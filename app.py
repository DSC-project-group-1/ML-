import torch
import librosa
import numpy as np
from transformers import HubertForSequenceClassification
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Tone Classification API",
    description="API for analyzing emotional tone in audio files using HuBERT model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load the model at startup
model_name = "Venkatesh4342/hubert-base-ls960-tone-classification"
model = HubertForSequenceClassification.from_pretrained(model_name)
model.eval()  # Set the model to evaluation mode

# Emotion mapping
EMOTIONS = {
    "0": "anger",
    "1": "fear",
    "2": "joy",
    "3": "love",
    "4": "sadness",
    "5": "surprise"
}

def preprocess_audio(audio_file_path):
    """Preprocess audio file for model input."""
    audio, sr = librosa.load(audio_file_path, sr=16000)  # Resample to 16kHz
    input_values = torch.tensor(audio).unsqueeze(0)  # Shape: (1, num_samples)
    return input_values

def predict_tone(audio_file_path):
    """Predict emotional tone from audio file."""
    input_values = preprocess_audio(audio_file_path)
    
    with torch.no_grad():
        logits = model(input_values).logits
    
    # Convert logits to probabilities using softmax
    probabilities = torch.nn.functional.softmax(logits, dim=-1).numpy()
    return probabilities.flatten()

@app.get("/predict")
async def predict_audio_tone():
    """
    Endpoint to predict emotional tone from an uploaded audio file.
    
    Parameters:
    - file: Audio file (WAV format recommended)
    
    Returns:
    - JSON with emotion probabilities
    """
    try:
        file = '../backend/uploads/record_out.wav'

        # Process the audio file
        predicted_probabilities = predict_tone(file)
        
        # Create emotion probability dictionary
        emotion_probabilities = {
            EMOTIONS[str(i)]: float(predicted_probabilities[i])
            for i in range(len(predicted_probabilities))
        }

        return JSONResponse(content={
            "status": "success",
            "predictions": emotion_probabilities
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Tone Classification API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Upload audio file for tone prediction",
            "/": "GET - This information"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)