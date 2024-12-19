!pip install pydub
from pydub import AudioSegment

# Convert audio to wav format
input_file ='/content/Recording (2).m4a' # Replace with your audio file's path
output_file = '/content/converted_audio.wav'

# Load and export as wav
audio = AudioSegment.from_file(input_file)
audio.export(output_file, format='wav')
print(f"Converted {input_file} to {output_file}")
