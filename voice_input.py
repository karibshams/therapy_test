import openai
import os
import pyaudio
import wave
import threading
import io
import numpy as np
import time
from typing import Optional, Callable
from dotenv import load_dotenv
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class RealTimeVoiceInput:
    """Real-time voice input with silence detection (no webrtcvad)."""

    def __init__(self, sample_rate: int = 16000, chunk_size: int = 1024, silence_threshold: float = 500):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.silence_threshold = silence_threshold  # lower = more sensitive

        # OpenAI client
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Audio
        self.pyaudio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False

        # Buffers
        self.audio_frames = []
        self.silence_counter = 0

        # Callbacks
        self.on_transcript_update: Optional[Callable[[str]]] = None
        self.on_final_transcript: Optional[Callable[[str]]] = None
        self.on_recording_start: Optional[Callable] = None
        self.on_recording_stop: Optional[Callable] = None

        self.record_thread = None

    def set_callbacks(self, on_transcript_update=None, on_final_transcript=None, on_recording_start=None, on_recording_stop=None):
        self.on_transcript_update = on_transcript_update
        self.on_final_transcript = on_final_transcript
        self.on_recording_start = on_recording_start
        self.on_recording_stop = on_recording_stop

    def _energy(self, audio_chunk: bytes) -> float:
        """Calculate RMS energy of audio chunk."""
        data = np.frombuffer(audio_chunk, dtype=np.int16)
        return np.sqrt(np.mean(data**2))

    def _record_audio(self):
        try:
            self.stream = self.pyaudio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )

            if self.on_recording_start:
                self.on_recording_start()

            while self.is_recording:
                audio_chunk = self.stream.read(self.chunk_size, exception_on_overflow=False)
                energy = self._energy(audio_chunk)

                if energy > self.silence_threshold:
                    self.audio_frames.append(audio_chunk)
                    self.silence_counter = 0
                else:
                    self.silence_counter += 1

                # Stop after ~1 sec of silence
                if self.audio_frames and self.silence_counter > int(self.sample_rate / self.chunk_size):
                    self._process_audio()
                    self.audio_frames = []
                    self.silence_counter = 0

        except Exception as e:
            print(f"Error recording: {e}")
        finally:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            if self.on_recording_stop:
                self.on_recording_stop()

    def _process_audio(self):
        """Send recorded audio to Whisper for transcription."""
        if not self.audio_frames:
            return
        try:
            audio_data = b"".join(self.audio_frames)
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(self.pyaudio.get_sample_size(pyaudio.paInt16))
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_data)
            wav_buffer.seek(0)
            wav_buffer.name = "audio.wav"

            transcript = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=wav_buffer,
                response_format="text"
            )

            if transcript.strip():
                if self.on_transcript_update:
                    self.on_transcript_update(transcript.strip())
                if self.on_final_transcript:
                    self.on_final_transcript(transcript.strip())
        except Exception as e:
            print(f"Error processing audio: {e}")

    def start_recording(self):
        if self.is_recording:
            return
        self.is_recording = True
        self.record_thread = threading.Thread(target=self._record_audio, daemon=True)
        self.record_thread.start()
        print("🎤 Recording started")

    def stop_recording(self):
        self.is_recording = False
        if self.record_thread and self.record_thread.is_alive():
            self.record_thread.join(timeout=2.0)
        print("🎤 Recording stopped")

    def cleanup(self):
        self.stop_recording()
        if self.pyaudio:
            self.pyaudio.terminate()
