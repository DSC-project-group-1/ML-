import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Function to extract MFCC features from an audio file
def extract_mfcc_features(audio_file):
    y, sr = librosa.load(audio_file, sr=None)  # Load the audio file
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Extract MFCCs
    mfcc_mean = np.mean(mfcc, axis=1)  # Take the mean of each MFCC coefficient
    return mfcc_mean

# Function to predict emotion from an audio file
def predict_emotion(audio_file, svm_model, encoder):
    # Extract features from the audio file
    features = extract_mfcc_features(audio_file)

    # Reshape features to match the input shape of the SVM model (1, -1)
    features = features.reshape(1, -1)

    # Make prediction using the trained SVM model
    predicted_label = svm_model.predict(features)

    # Decode the predicted label to get the emotion
    emotion = encoder.inverse_transform(predicted_label)

    return emotion[0]


# Replace with your own path to the audio file
audio_file = '/content/converted_audio.wav'

# Call the predict_emotion function
emotion = predict_emotion(audio_file, svm_model, encoder)

print(f"Predicted Emotion: {emotion}")
