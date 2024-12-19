import librosa
import numpy as np

# Function to extract features using VGGish model (replacing VGGish feature extraction here)
def extract_mfcc_features(audio_file):
    # Load the audio file
    y, sr = librosa.load(audio_file, sr=None)

    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Extract 13 MFCCs

    # Take the mean of each MFCC coefficient over the time axis
    mfcc_mean = np.mean(mfcc, axis=1)

    return mfcc_mean

# Load the RAVDESS dataset and extract features
def load_data(dataset_path):
    features = []
    labels = []

    # Emotion mapping based on RAVDESS emotion codes
    emotion_dict = {
        '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
        '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
    }

    for actor_folder in os.listdir(dataset_path):
        actor_path = os.path.join(dataset_path, actor_folder)
        if os.path.isdir(actor_path):
            for filename in os.listdir(actor_path):
                if filename.endswith('.wav'):
                    emotion_code = filename.split('-')[2]  # Extract emotion code from filename
                    file_path = os.path.join(actor_path, filename)
                    feature = extract_mfcc_features(file_path)
                    features.append(feature)
                    labels.append(emotion_dict.get(emotion_code, 'unknown'))

    features = np.array(features)
    labels = np.array(labels)

    return features, labels

# Load the dataset from the extracted folder
dataset_path = '/content/ravdess'  # Path to the extracted dataset
features, labels = load_data(dataset_path)

# Encode the labels (using a LabelEncoder)
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
labels_encoded = encoder.fit_transform(labels)

# Split the dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42)

# Train a Support Vector Machine (SVM) classifier
from sklearn.svm import SVC
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Evaluate the classifier
from sklearn.metrics import accuracy_score
y_pred = svm_model.predict(X_test)

# Print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")





