# SleepTalk Recorder 🎤💤

A Python script that automatically detects and records talking, then transcribes it with automatic language detection.

## The Problem

Ever had a roommate who rambles in their sleep but never believes you when you tell them what they said? Yeah, me too. Catching the exact moment they start talking at 3 AM is basically impossible.

This script solves that by listening continuously and only recording when it detects actual speech (not snoring, fan noise, or other sounds).

## How It Works
- **Continuous listening** using your microphone
- **Speech detection** using Silero-VAD (distinguishes speech from other sounds like snoring)
- **Automatic recording** starts when speech is detected, stops after 2 seconds of silence
- **Timestamped audio clips** saved locally (no cloud, completely private)
- **Transcription** using OpenAI's Whisper with automatic language detection (supports 99+ languages)

Everything runs locally on your machine. No APIs, no servers.

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/sleeptalk-recorder.git
cd sleeptalk-recorder
```

### 2. Create a virtual environment

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install torch torchaudio sounddevice scipy numpy openai-whisper packaging
```

This will take a few minutes as PyTorch is fairly large (~700MB).

## Usage

### Recording Sleep Talk

1. Make sure your microphone is working
2. Run the recorder:
```bash
python sleep_recorder.py
```

3. You'll see:
```
Sleep Talk Recorder Starting...
Listening for speech... (threshold: 0.5)
Press Ctrl+C to stop

🎧 Recording... waiting for speech
```

4. When it detects speech, you'll see:
```
🎤 Speech detected! (confidence: 0.87)
🔇 Silence detected, stopping recording...
💾 Saved: recordings/speech_20240206_032345.wav (4.2s)
```

5. Press `Ctrl+C` to stop.

### Transcribing Recordings

After you've collected some recordings:
```bash
python transcribe.py
```

This will:
- Find all `.wav` files in the `recordings/` folder
- Transcribe each one (auto-detecting the language)
- Save everything to `transcriptions.txt`
- Optionally delete audio files to save space

**Note:** First run will download the Whisper model (~140MB).


## Configuration

You can adjust settings in `sleep_recorder.py`:
```python
SPEECH_THRESHOLD = 0.5      # Lower = more sensitive (0.3-0.7 recommended)
SILENCE_DURATION = 2.0      # Seconds of silence before stopping
PRE_SPEECH_BUFFER = 1.0     # Seconds to include before speech starts
```

## How It Works (Technical)

1. **Audio Capture**: Uses `sounddevice` to capture audio in 512-sample chunks (~32ms at 16kHz)
2. **Speech Detection**: Silero-VAD model analyzes each chunk and outputs speech probability (0-1)
3. **Recording Logic**: 
   - When probability > threshold, start recording
   - Include 1 second of pre-buffered audio
   - Stop after 2 seconds of continuous silence
4. **Transcription**: Whisper model converts audio to text with language auto-detection

## Privacy

- Everything runs **locally** on your machine
- No data is sent to any servers or APIs
- Audio files stay in your `recordings/` folder
- You have full control over what gets saved and deleted



## Tech Stack

- **Python 3.8+**
- **PyTorch** - Deep learning framework
- **Silero-VAD** - Voice activity detection model
- **OpenAI Whisper** - Speech-to-text transcription
- **sounddevice** - Audio capture
- **scipy** - Audio file I/O

## Contributing

Feel free to open issues or submit PRs! Some ideas for improvements:
- Web interface for viewing recordings
- Real-time transcription
- Better noise filtering
- Mobile app version

## License

MIT License - feel free to use and modify as you like.

## Disclaimer

Make sure you have consent from anyone being recorded. This tool is intended for fun personal use with roommates/family who are aware and okay with it.

---

Built during a late night coding session because my roommate wouldn't believe he talks in his sleep. 😅
