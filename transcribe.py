import whisper
import os
from datetime import datetime
import glob

print("Loading Whisper model...")
print("(This might take a minute on first run - downloading model)")

# Load Whisper model
# Options: tiny, base, small, medium, large
# tiny = fastest but less accurate
# base = good balance (recommended to start)
# large = most accurate but slowest
model = whisper.load_model("base")

print("✅ Model loaded!\n")

# Find all audio files in recordings folder
audio_files = glob.glob("recordings/*.wav")

if not audio_files:
    print("❌ No audio files found in recordings/ folder")
    exit()

print(f"Found {len(audio_files)} audio file(s) to transcribe\n")

# Create/open transcription log file
transcript_file = "transcriptions.txt"

with open(transcript_file, "a", encoding="utf-8") as f:
    for audio_path in sorted(audio_files):
        filename = os.path.basename(audio_path)
        print(f"🎧 Transcribing: {filename}...")
        
        try:
            # Transcribe with Whisper
            result = model.transcribe(audio_path, fp16=False)
            
            # Extract info
            text = result["text"].strip()
            language = result["language"]
            
            # Get file timestamp from filename (speech_YYYYMMDD_HHMMSS.wav)
            try:
                timestamp_str = filename.replace("speech_", "").replace(".wav", "")
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                formatted_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            except:
                formatted_time = "Unknown time"
            
            # Calculate duration
            duration = len(whisper.load_audio(audio_path)) / 16000
            
            # Write to file
            f.write(f"\n{'='*60}\n")
            f.write(f"[{formatted_time}] Duration: {duration:.1f}s | Language: {language}\n")
            f.write(f'"{text}"\n')
            f.write(f"File: {filename}\n")
            
            # Print to console
            print(f"  ✅ Language: {language}")
            print(f'  💬 "{text}"')
            print()
            
        except Exception as e:
            print(f"  ❌ Error: {e}\n")
            f.write(f"\n[ERROR] {filename}: {e}\n")

print(f"\n✅ All transcriptions saved to: {transcript_file}")

# Optional: Ask if user wants to delete audio files
print("\nDo you want to delete the audio files to save space? (y/n): ", end="")
response = input().strip().lower()

if response == 'y':
    for audio_path in audio_files:
        os.remove(audio_path)
        print(f"🗑️  Deleted: {os.path.basename(audio_path)}")
    print("\n✅ Audio files deleted!")
else:
    print("\n📁 Audio files kept in recordings/ folder")
