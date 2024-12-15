import os
import asyncio
import wave
import pyaudio
from dotenv import load_dotenv
from aiohttp import ClientSession
import string

import string
from difflib import SequenceMatcher

# Load environment variables
load_dotenv()
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

# This class deals with recording the audio from the user and define it's properties.
class AudioRecorder:
    def __init__(self, 
                filename="recording.wav", 
                channels=1, 
                rate=16000, 
                chunk=1024, 
                record_seconds=5):
        self.filename = filename
        self.channels = channels
        self.rate = rate
        self.chunk = chunk
        self.record_seconds = record_seconds
        
        self.pyaudio = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        
    def start_recording(self):
        print("Recording started. Speak now...")
        
        # Open stream
        self.stream = self.pyaudio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        
        # Initialize frames list
        self.frames = []
        
        # Record audio
        for _ in range(0, int(self.rate / self.chunk * self.record_seconds)):
            data = self.stream.read(self.chunk)
            self.frames.append(data)
        
        # Stop and close the stream
        self.stream.stop_stream()
        self.stream.close()
        self.pyaudio.terminate()
        
        # Save the recorded audio to a WAV file
        wf = wave.open(self.filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.pyaudio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        
        print(f"Recording saved to {self.filename}")
        
        return self.filename


# This class takes the audio of .wav format from the record Class and transcripts it.
class DeepgramTranscriber:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.deepgram.com/v1/listen"
    
    async def transcribe_audio_file(self, audio_file):
        try:
            # Open the audio file in binary mode
            with open(audio_file, 'rb') as f:
                audio_data = f.read()
            
            # Configure headers and parameters
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
            
            # Send the audio to Deepgram for transcription
            async with ClientSession() as session:
                async with session.post(self.base_url, headers=headers, params=params, data=audio_data) as response:
                    if response.status == 200:
                        result = await response.json()
                        transcript = result['results']['channels'][0]['alternatives'][0]['transcript']
                        confidence = result['results']['channels'][0]['alternatives'][0]['confidence']
                        
                        print("\n--- Transcription Results ---")
                        print(f"Transcript: {transcript}")
                        print(f"Confidence: {confidence}")
                        
                        # Write the transcript to a text file
                        self.write_transcript_to_file(transcript)
                        
                        return transcript, confidence
                    else:
                        error_message = await response.text()
                        print(f"Error: {error_message}")
                        return None, None
        
        except Exception as e:
            print(f"Transcription error: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def write_transcript_to_file(self, transcript):
        try:
            # Define the file path (can be modified)
            file_path = "transcription.txt"
            
            # open the file in append mode, or create it if it doesn't exist
            with open(file_path, 'w') as file:
                file.write(f"{transcript}")
            
            print(f"Transcript saved to {file_path}")
        
        except Exception as e:
            print(f"Error saving transcript to file: {e}")
            import traceback
            traceback.print_exc()
# compares the transcribed text with the challenge statement based on a similarity threshold.
def check_challenge_statement(transcribed_text, challenge_statement="my kids died", similarity_threshold=0.9):
    # normalize and remove punctuation
    def normalize_and_remove_punctuation(text):
        return ''.join(char.lower() for char in text if char not in string.punctuation).strip()
    
    normalized_transcribed = normalize_and_remove_punctuation(transcribed_text)
    normalized_challenge = normalize_and_remove_punctuation(challenge_statement)
    
    # calculate similarity
    similarity_ratio = SequenceMatcher(None, normalized_transcribed, normalized_challenge).ratio()
    
    print(f"Similarity ratio: {similarity_ratio:.2f}")
    
    if similarity_ratio >= similarity_threshold:
        print("The audio matches the challenge statement based on similarity threshold.")
        return True
    else:
        print("The audio does not match the challenge statement based on similarity threshold.")
        print(f"Expected: '{normalized_challenge}'")
        print(f"Received: '{normalized_transcribed}'")
        return False

def delete_transcription_file(file_path="transcription.txt"):
    # this will delete the transcription file if it exists.
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted file: {file_path}")
        else:
            print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error deleting file: {e}")
        import traceback
        traceback.print_exc()

async def main():
    if not DEEPGRAM_API_KEY:
        raise ValueError("Deepgram API key not found. Check your .env file.")
    
    try:
        recorder = AudioRecorder(record_seconds=5)
        audio_file = recorder.start_recording()
        
        transcriber = DeepgramTranscriber(DEEPGRAM_API_KEY)
        transcript, confidence = await transcriber.transcribe_audio_file(audio_file)
        
        if transcript:
            is_match = check_challenge_statement(transcript)
            delete_transcription_file()
            return is_match
        else:
            print("No transcription available to check.")
            delete_transcription_file()
            return False

    except Exception as e:
        print(f"Main process error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = asyncio.run(main())
    if result:
        print("Challenge completed successfully.")
    else:
        print("Challenge failed.")
