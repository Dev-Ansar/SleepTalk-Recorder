import sounddevice as sd
import numpy as np
import torch
import torchaudio
from scipy.io import wavfile
from datetime import datetime
import os
import time

# Create recordings folder
os.makedirs('recordings', exist_ok=True)

# Load Silero VAD model
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=False,
                              onnx=False,
                              trust_repo=True)

(get_speech_timestamps, _, read_audio, *_) = utils

# Audio settings
SAMPLE_RATE = 16000  # Silero VAD works best with 16kHz
CHUNK_SIZE = 512  # Silero VAD expects exactly 512 samples for 16kHz
CHUNK_DURATION = CHUNK_SIZE / SAMPLE_RATE  # ~0.032 seconds

# Recording settings
SPEECH_THRESHOLD = 0.5  # Confidence threshold for speech detection
SILENCE_DURATION = 2.0  # Seconds of silence before stopping recording
PRE_SPEECH_BUFFER = 1.0  # Seconds to include before speech starts

print("Sleep Talk Recorder Starting...")
print(f"Listening for speech... (threshold: {SPEECH_THRESHOLD})")
print("Press Ctrl+C to stop\n")

# Buffer to store recent audio (for pre-speech context)
pre_buffer = []
pre_buffer_size = int(PRE_SPEECH_BUFFER / CHUNK_DURATION)

# Recording state
is_recording = False
recording_buffer = []
silence_chunks = 0
silence_threshold = int(SILENCE_DURATION / CHUNK_DURATION)

def save_recording(audio_data, sample_rate):
    """Save recorded audio with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"recordings/speech_{timestamp}.wav"
    
    # Convert list of chunks to single array
    audio_array = np.concatenate(audio_data)
    
    # Save as WAV file
    wavfile.write(filename, sample_rate, audio_array)
    print(f"💾 Saved: {filename} ({len(audio_array)/sample_rate:.1f}s)")

def audio_callback(indata, frames, time_info, status):
    """Called for each audio chunk"""
    global is_recording, recording_buffer, silence_chunks, pre_buffer
    
    if status:
        print(f"Status: {status}")
    
    # Convert to the format Silero expects
    audio_chunk = indata[:, 0].copy()  # Get mono channel
    audio_tensor = torch.from_numpy(audio_chunk).float()
    
    # Get speech probability from VAD
    speech_prob = model(audio_tensor, SAMPLE_RATE).item()
    
    # Check if speech is detected
    is_speech = speech_prob > SPEECH_THRESHOLD
    
    if is_speech:
        if not is_recording:
            # Start recording - include pre-buffer
            print(f"\n🎤 Speech detected! (confidence: {speech_prob:.2f})")
            is_recording = True
            recording_buffer = list(pre_buffer)  # Include buffered audio
            
        recording_buffer.append(audio_chunk)
        silence_chunks = 0
        
    elif is_recording:
        # Still recording but no speech in this chunk
        recording_buffer.append(audio_chunk)
        silence_chunks += 1
        
        if silence_chunks >= silence_threshold:
            # Enough silence, stop and save
            print(f"🔇 Silence detected, stopping recording...")
            save_recording(recording_buffer, SAMPLE_RATE)
            is_recording = False
            recording_buffer = []
            silence_chunks = 0
    
    # Maintain pre-speech buffer
    pre_buffer.append(audio_chunk)
    if len(pre_buffer) > pre_buffer_size:
        pre_buffer.pop(0)

# Start the audio stream
try:
    with sd.InputStream(callback=audio_callback,
                       channels=1,
                       samplerate=SAMPLE_RATE,
                       blocksize=CHUNK_SIZE):
        print("🎧 Recording... waiting for speech")
        while True:
            time.sleep(0.1)
            
except KeyboardInterrupt:
    print("\n\n👋 Stopping recorder...")
    if is_recording and recording_buffer:
        print("Saving final recording...")
        save_recording(recording_buffer, SAMPLE_RATE)
    print("Done!")
