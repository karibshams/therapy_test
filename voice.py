import os
import io
import wave
import threading
import time
from typing import Optional
import sounddevice as sd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class VoiceInput:
    def __init__(self):
        """Initialize the voice chatbot with OpenAI client and audio settings."""
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.stt_model = os.getenv('STT_MODEL', 'whisper-1')
        
        
        # Audio settings
        self.sample_rate = 44100
        self.channels = 1
        self.chunk_duration = 1.0  # seconds
        self.min_recording_duration = 2.0  # minimum seconds to record
        
        # Recording state
        self.is_recording = False
        self.audio_buffer = []
       
        
        print("🎤 Voice Chatbot initialized!")
        print(f"Using STT model: {self.stt_model}")
        print("\nInstructions:")
        print("- Press ENTER to start recording")
        print("- Press ENTER again to stop recording and get response")
        print("- Type 'quit' to exit")
        print("-" * 50)

    def audio_callback(self, indata, frames, time, status):
        """Callback function for audio input."""
        if self.is_recording:
            self.audio_buffer.extend(indata[:, 0])

    def start_recording(self):
        """Start recording audio from microphone."""
        self.is_recording = True
        self.audio_buffer = []
        print("🔴 Recording... Press ENTER to stop")
        
        with sd.InputStream(callback=self.audio_callback, 
                          samplerate=self.sample_rate, 
                          channels=self.channels):
            input()  
        
        self.is_recording = False
        print("⏹️ Recording stopped")

    def save_audio_to_bytes(self, audio_data: np.ndarray) -> bytes:
        """Convert numpy audio data to WAV bytes."""
        # Convert float32 to int16
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(2)  # 2 bytes for int16
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        wav_buffer.seek(0)
        return wav_buffer.read()

    def speech_to_text(self, audio_bytes: bytes) -> Optional[str]:
        """Convert speech to text using OpenAI Whisper."""
        try:
            # Create a file-like object from bytes
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = "recording.wav"  # Whisper needs a filename
            
            print("🔄 Converting speech to text...")
            response = self.client.audio.transcriptions.create(
                model=self.stt_model,
                file=audio_file,
                response_format="text"
            )
            
            return response.strip() if response else None
            
        except Exception as e:
            print(f"❌ Speech-to-text error: {e}")
            return None

    def get_voice_input(self) -> Optional[str]:
        """Record voice input and convert to text."""
        if not self.audio_buffer:
            print("❌ No audio recorded")
            return None
        
        # Check if recording is long enough
        recording_duration = len(self.audio_buffer) / self.sample_rate
        if recording_duration < self.min_recording_duration:
            print(f"❌ Recording too short ({recording_duration:.1f}s). Minimum is {self.min_recording_duration}s")
            return None
        
        print(f"🔊 Processing {recording_duration:.1f} seconds of audio...")
        
        # Convert audio to bytes
        audio_array = np.array(self.audio_buffer, dtype=np.float32)
        audio_bytes = self.save_audio_to_bytes(audio_array)
        
        # Speech to text
        transcribed_text = self.speech_to_text(audio_bytes)
        if not transcribed_text:
            print("❌ Could not transcribe audio")
            return None
        
        print(f"📋 You said: \"{transcribed_text}\"")
        return transcribed_text

    def record_and_transcribe(self) -> Optional[str]:
        """Complete workflow: record audio and return transcribed text."""
        self.start_recording()
        return self.get_voice_input()
    # In voice.py, add noise reduction and better audio validation
    def validate_audio_quality(self, audio_data: np.ndarray) -> bool:
        """Check if audio has sufficient signal."""
        rms = np.sqrt(np.mean(audio_data ** 2))
        return rms > 0.01  # Adjust threshold as needed