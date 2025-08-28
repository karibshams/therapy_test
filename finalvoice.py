import os
import io
import wave
from typing import Optional
import sounddevice as sd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class VoiceInput:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.stt_model = os.getenv('STT_MODEL', 'whisper-1')
        
        # Audio settings
        self.sample_rate = 44100
        self.channels = 1
        self.min_recording_duration = 2.0
        
        # Recording state
        self.is_recording = False
        self.audio_buffer = []

    def audio_callback(self, indata, frames, time, status):
        if self.is_recording:
            self.audio_buffer.extend(indata[:, 0])

    def start_recording(self):
        self.is_recording = True
        self.audio_buffer = []
        
        with sd.InputStream(callback=self.audio_callback, 
                          samplerate=self.sample_rate, 
                          channels=self.channels):
            input()  # Wait for user input to stop
        
        self.is_recording = False

    def save_audio_to_bytes(self, audio_data: np.ndarray) -> bytes:
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        wav_buffer.seek(0)
        return wav_buffer.read()

    def speech_to_text(self, audio_bytes: bytes) -> Optional[str]:
        try:
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = "recording.wav"
            
            response = self.client.audio.transcriptions.create(
                model=self.stt_model,
                file=audio_file,
                response_format="text"
            )
            
            return response.strip() if response else None
            
        except Exception as e:
            print(f"Speech-to-text error: {e}")
            return None

    def get_voice_input(self) -> Optional[str]:
        if not self.audio_buffer:
            return None
        
        recording_duration = len(self.audio_buffer) / self.sample_rate
        if recording_duration < self.min_recording_duration:
            return None
        
        audio_array = np.array(self.audio_buffer, dtype=np.float32)
        audio_bytes = self.save_audio_to_bytes(audio_array)
        
        return self.speech_to_text(audio_bytes)

    def record_and_transcribe(self) -> Optional[str]:
        self.start_recording()
        return self.get_voice_input()
    def validate_audio_quality(self, audio_data: np.ndarray) -> bool:
        """Check if audio has sufficient signal."""
        rms = np.sqrt(np.mean(audio_data ** 2))
        return rms > 0.01  