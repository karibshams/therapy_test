import os
import io
import wave
import queue
import numpy as np
import sounddevice as sd
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv
import time

load_dotenv()

class VoiceInput:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.stt_model = os.getenv("STT_MODEL", "whisper-1")

        # Audio settings
        self.sample_rate = 16000
        self.channels = 1

        # Silence detection settings
        self.silence_threshold = 0.01   # volume level for silence
        self.silence_duration = 2.0     # stop after 2s of silence

        self.audio_buffer = []
        self.is_recording = False

    def save_audio_to_bytes(self, audio_data: np.ndarray) -> bytes:
        """Convert numpy audio array to WAV bytes."""
        audio_int16 = (audio_data * 32767).astype(np.int16)
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        wav_buffer.seek(0)
        return wav_buffer.read()

    def speech_to_text(self, audio_bytes: bytes) -> Optional[str]:
        """Send audio to Whisper and get transcription."""
        try:
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = "recording.wav"
            response = self.client.audio.transcriptions.create(
                model=self.stt_model,
                file=audio_file,
                response_format="text",
                language="en"
            )
            return response.strip() if response else None
        except Exception as e:
            print(f"Speech-to-text error: {e}")
            return None

    def record_and_transcribe(self) -> Optional[str]:
        """Record voice until silence is detected, then transcribe."""
        print("🎤 Listening... Start speaking. (Auto-stop after silence)")

        self.audio_buffer = []
        self.is_recording = True
        silence_start = None

        def callback(indata, frames, time_info, status):
            nonlocal silence_start
            if status:
                print(f"⚠️ Recording error: {status}", flush=True)

            # Flatten input to 1D
            samples = indata[:, 0]
            self.audio_buffer.extend(samples)

            # Check volume (RMS)
            rms = np.sqrt(np.mean(samples ** 2))

            if rms < self.silence_threshold:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > self.silence_duration:
                    raise sd.CallbackStop()  # stop stream
            else:
                silence_start = None  # reset if speaking again

        try:
            with sd.InputStream(callback=callback,
                                samplerate=self.sample_rate,
                                channels=self.channels):
                sd.sleep(60000)  # allow up to 1 minute per turn
        except sd.CallbackStop:
            pass

        self.is_recording = False

        if not self.audio_buffer:
            print("⚠️ No audio captured.")
            return None

        print("⏹️ Recording stopped. Processing...")

        audio_array = np.array(self.audio_buffer, dtype=np.float32)
        audio_bytes = self.save_audio_to_bytes(audio_array)

        return self.speech_to_text(audio_bytes)
