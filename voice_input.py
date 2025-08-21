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

class RealTimeVoiceInput:
    """Simplified real-time voice input for Streamlit compatibility."""

    def __init__(self, 
                 sample_rate: int = 16000, 
                 chunk_size: int = 1024, 
                 silence_threshold: float = 500,
                 silence_duration: float = 2.0):
        
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.silence_threshold = silence_threshold
        self.silence_frames = int((silence_duration * sample_rate) / chunk_size)

        # OpenAI client
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Audio
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False

        # Data
        self.frames = []
        self.silence_count = 0
        
        # Simple callbacks (no Streamlit state access)
        self.transcript_callback = None
        self.start_callback = None
        self.stop_callback = None

        logger.info("Voice input initialized")

    def set_callbacks(self, on_transcript_update=None, on_final_transcript=None, 
                     on_recording_start=None, on_recording_stop=None):
        """Set callbacks - only final transcript is used."""
        self.transcript_callback = on_final_transcript
        self.start_callback = on_recording_start
        self.stop_callback = on_recording_stop

    def _get_volume(self, data):
        """Calculate volume level."""
        try:
            audio_data = np.frombuffer(data, dtype=np.int16)
            return np.sqrt(np.mean(audio_data**2))
        except:
            return 0

    def _record_loop(self):
        """Main recording loop."""
        try:
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )

            logger.info("Recording started")
            if self.start_callback:
                try:
                    self.start_callback()
                except:
                    pass

            self.frames = []
            self.silence_count = 0
            speech_started = False

            while self.is_recording:
                try:
                    data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                    volume = self._get_volume(data)
                    
                    if volume > self.silence_threshold:
                        # Speech detected
                        if not speech_started:
                            speech_started = True
                            logger.info("Speech started")
                        
                        self.frames.append(data)
                        self.silence_count = 0
                        
                    else:
                        # Silence
                        if speech_started:
                            self.silence_count += 1
                            if self.silence_count < 5:  # Add a few silence frames
                                self.frames.append(data)
                            
                            # Process if enough silence
                            if self.silence_count >= self.silence_frames and len(self.frames) > 10:
                                logger.info(f"Processing {len(self.frames)} frames")
                                self._process_audio()
                                
                                # Reset
                                self.frames = []
                                self.silence_count = 0
                                speech_started = False

                except Exception as e:
                    logger.error(f"Recording error: {e}")
                    break

        except Exception as e:
            logger.error(f"Stream error: {e}")
        finally:
            self._cleanup()

    def _process_audio(self):
        """Process recorded audio."""
        if not self.frames:
            return
            
        try:
            # Combine frames
            audio_data = b"".join(self.frames)
            
            # Create WAV
            wav_io = io.BytesIO()
            with wave.open(wav_io, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_data)
            
            wav_io.seek(0)
            wav_io.name = "audio.wav"

            # Transcribe
            response = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=wav_io,
                response_format="text"
            )
            
            text = response.strip()
            if text and len(text) > 1:
                logger.info(f"Transcribed: {text}")
                if self.transcript_callback:
                    # Call callback in main thread to avoid Streamlit issues
                    self.transcript_callback(text)
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")

    def _cleanup(self):
        """Clean up audio stream."""
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            logger.info("Stream closed")
        except:
            pass
        
        if self.stop_callback:
            try:
                self.stop_callback()
            except:
                pass

    def start_recording(self):
        """Start recording."""
        if self.is_recording:
            return False
            
        self.is_recording = True
        self.record_thread = threading.Thread(target=self._record_loop, daemon=True)
        self.record_thread.start()
        return True

    def stop_recording(self):
        """Stop recording."""
        if not self.is_recording:
            return False
            
        self.is_recording = False
        if hasattr(self, 'record_thread'):
            self.record_thread.join(timeout=2)
        return True

    def is_recording_active(self):
        """Check if recording."""
        return self.is_recording

    def cleanup(self):
        """Cleanup everything."""
        self.stop_recording()
        try:
            self.audio.terminate()
        except:
            pass
        logger.info("Voice input cleaned up")

    def get_status(self):
        """Get status."""
        return {
            "is_recording": self.is_recording,
            "frames_count": len(self.frames) if hasattr(self, 'frames') else 0
        }