import wave
import contextlib
import os
import whisper
import json

# Parameters
file_path = "output.wav"
segment_length = 10  # seconds
model = whisper.load_model("medium")  # Load the Whisper model

# Get duration of the audio file
with contextlib.closing(wave.open(file_path, 'r')) as f:
    frames = f.getnframes()
    rate = f.getframerate()
    duration = frames / float(rate)

# Create folder for segments
os.makedirs("segments", exist_ok=True)

# Split into chunks using ffmpeg
num_segments = int(duration // segment_length + 1)
for i in range(num_segments):
    start = i * segment_length
    output_name = f"segments/segment_{i}.wav"
    os.system(f"ffmpeg -y -i {file_path} -ss {start} -t {segment_length} {output_name}")

# Transcribe each segment and simulate speaker turns
dialogue = []
for i in range(num_segments):
    segment_file = f"segments/segment_{i}.wav"
    result = model.transcribe(segment_file)
    speaker = f"Speaker {i % 2 + 1}"
    dialogue.append({
        "speaker": speaker,
        "text": result["text"].strip()
    })

# Save to JSON
with open("transcription.json", "w") as f:
    json.dump(dialogue, f, indent=2)

print("Transcription saved to transcription.json")
