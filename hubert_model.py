import torch
import librosa
from transformers import HubertForSequenceClassification
import numpy as np

# Load the pre-trained model
model_name = "Venkatesh4342/hubert-base-ls960-tone-classification"
model = HubertForSequenceClassification.from_pretrained(model_name)

# Function to preprocess audio file
def preprocess_audio(audio_file):
    audio, sr = librosa.load(audio_file, sr=16000)  # Resample to 16kHz
    input_values = torch.tensor(audio).unsqueeze(0)  # Shape: (1, num_samples)
    return input_values

# Function to predict tone from audio file and return probabilities
def predict_tone(audio_file):
    input_values = preprocess_audio(audio_file)
    
    with torch.no_grad():
        logits = model(input_values).logits
    
    # Convert logits to probabilities using softmax
    probabilities = torch.nn.functional.softmax(logits, dim=-1).numpy()
    
    return probabilities.flatten()  # Return as a flat array

# Mapping of indices to emotions
EMOTIONS = {
    "0": "anger",
    "1": "fear",
    "2": "joy",
    "3": "love",
    "4": "sadness",
    "5": "surprise"
}

# Example usage
audio_file = 'record_out.wav'
predicted_probabilities = predict_tone(audio_file)

# Create a dictionary to map emotions to their predicted probabilities
emotion_probabilities = {EMOTIONS[str(i)]: predicted_probabilities[i] for i in range(len(predicted_probabilities))}

print("Emotion Probabilities:")
for emotion, probability in emotion_probabilities.items():
    print(f"{emotion}: {probability:.3f}")
