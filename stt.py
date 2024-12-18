import os
import asyncio
import string
from dotenv import load_dotenv
from aiohttp import ClientSession
from difflib import SequenceMatcher

from challenge_generator import get_random_sentence_with_emotion, load_dataset

# this will load environment variables
load_dotenv()
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

# this handles transcription of audio files using Deepgram API
class DeepgramTranscriber:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.deepgram.com/v1/listen"
    
    async def transcribe_audio_file(self, audio_file):
        try:
            with open(audio_file, 'rb') as f:
                audio_data = f.read()
            
            headers = {
                "Authorization": f"Token {self.api_key}",
                "Content-Type": "audio/wav",
            }
            params = {
                "model": "nova",
                "language": "en-US",
                "punctuate": "true",
                "utterances": "true",
            }
            
            async with ClientSession() as session:
                async with session.post(self.base_url, headers=headers, params=params, data=audio_data) as response:
                    if response.status == 200:
                        result = await response.json()
                        transcript = result['results']['channels'][0]['alternatives'][0]['transcript']
                        print(f"Transcription: {transcript}")
                        return transcript
                    else:
                        print(f"Error: {await response.text()}")
                        return None
        except Exception as e:
            print(f"Transcription error: {e}")
            return None


def check_challenge_statement(transcribed_text, challenge_statement, similarity_threshold=0.9):
    """Checks if the transcribed text matches the challenge statement."""
    def normalize_and_remove_punctuation(text):
        return ''.join(char.lower() for char in text if char not in string.punctuation).strip()
    
    normalized_transcribed = normalize_and_remove_punctuation(transcribed_text)
    normalized_challenge = normalize_and_remove_punctuation(challenge_statement)
    
    similarity_ratio = SequenceMatcher(None, normalized_transcribed, normalized_challenge).ratio()
    return similarity_ratio >= similarity_threshold


async def process_audio_and_match(audio_file_path, sentence):
    """Processes the provided audio file and checks transcription against a challenge statement."""
    if not DEEPGRAM_API_KEY:
        raise ValueError("Deepgram API key not found. Check your .env file.")
    

    # All of this is commented coz these processes will be carried out in audio_recorder.py until this is not connected to the web component.
    # Load the dataset
    # dataset_path = 'test.csv'  # Update this path
    # df = load_dataset(dataset_path)

    # # Get a random sentence with its core and random emotions
    # sentence, core_emotion, random_emotion = get_random_sentence_with_emotion(df)
    # print("Challenge sentence:", sentence)
    
    transcriber = DeepgramTranscriber(DEEPGRAM_API_KEY)
    transcript = await transcriber.transcribe_audio_file(audio_file_path)

    if transcript:
        return check_challenge_statement(transcript, sentence)
    else:
        return False
