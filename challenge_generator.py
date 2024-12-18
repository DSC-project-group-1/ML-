import random
import pandas as pd

# Predefined emotions
EMOTIONS = {
    "0": "anger",
    "1": "fear",
    "2": "joy",
    "3": "love",
    "4": "sadness",
    "5": "surprise"
}

def load_dataset(file_path = 'test.csv'):
    """
    Load the dataset from a CSV file and return it as a pandas DataFrame.
    """
    df = pd.read_csv('test.csv')
    if 'text' not in df.columns or 'emotion' not in df.columns:                       # this will ensure the dataset contains the necessary columns
        raise ValueError("Dataset must contain 'text' and 'emotion' columns.")
    return df

def get_random_sentence_with_emotion(df):
    """
    Select a random sentence from the dataset, return it along with its core emotion and a random emotion.
    """
    # dataset_path = 'test.csv'  # Update this path
    # df = load_dataset(dataset_path)
    # Select a random row from the DataFrame
    selected = df.sample(n=1).iloc[0]
    sentence = selected['text']
    core_emotion_key = str(selected['emotion'])
    core_emotion = EMOTIONS.get(core_emotion_key, "unknown")

    # Select a random emotion that is not the same as the core emotion
    possible_emotions = [key for key in EMOTIONS.keys() if key != core_emotion_key]
    random_emotion_key = random.choice(possible_emotions)
    random_emotion = EMOTIONS[random_emotion_key]

    return sentence, core_emotion, random_emotion
