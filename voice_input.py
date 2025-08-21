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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class RealTimeVoiceInput:
    """Simplified real-time voice input with reliable STT processing."""

    def __init__(self, 
                 sample_rate: int = 16000, 
                 chunk_size: int = 1024, 
                 silence_threshold: float = 500,
                 silence_duration: float = 2.0,
                 min_audio_length: float = 1.0):
        """
        Initialize voice input system with simplified parameters.
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.min_audio_length = min_audio_length
        
        # Calculate frames needed
        self.silence_frames_needed = int((silence_duration * sample_rate) / chunk_size)
        self.min_audio_frames = int((min_audio_length * sample_rate) / chunk_size)

        # OpenAI client
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Audio components
        self.pyaudio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False

        # Audio processing
        self.audio_buffer = []
        self.silence_counter = 0
        self.speech_detected = False

        # Callbacks
        self.on_transcript_update: Optional[Callable[[str]]] = None
        self.on_final_transcript: Optional[Callable[[str]]] = None
        self.on_recording_start: Optional[Callable] = None
        self.on_recording_stop: Optional[Callable] = None

        # Threading
        self.record_thread = None
        self.is_processing = False
        
        logger.info("Voice input initialized successfully")

    def set_callbacks(self, on_transcript_update=None, on_final_transcript=None, 
                     on_recording_start=None, on_recording_stop=None):
        """Set callback functions for voice events."""
        self.on_transcript_update = on_transcript_update
        self.on_final_transcript = on_final_transcript
        self.on_recording_start = on_recording_start
        self.on_recording_stop = on_recording_stop

    def _calculate_volume(self, audio_chunk: bytes) -> float:
        """Calculate volume level of audio chunk."""
        try:
            data = np.frombuffer(audio_chunk, dtype=np.int16)
            if len(data) == 0:
                return 0.0
            return float(np.sqrt(np.mean(data.astype(np.float64)**2)))
        except Exception as e:
            logger.error(f"Error calculating volume: {e}")
            return 0.0

    def _is_speech(self, volume: float) -> bool:
        """Simple speech detection based on volume threshold."""
        return volume > self.silence_threshold

    def _record_audio_loop(self):
        """Main recording loop with simplified logic."""
        try:
            # Open audio stream
            self.stream = self.pyaudio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )

            logger.info("Audio recording started")
            
            # Notify recording started
            if self.on_recording_start:
                try:
                    self.on_recording_start()
                except Exception as e:
                    logger.error(f"Error in recording start callback: {e}")

            # Reset state
            self.audio_buffer = []
            self.silence_counter = 0
            self.speech_detected = False

            while self.is_recording:
                try:
                    # Read audio chunk
                    audio_chunk = self.stream.read(self.chunk_size, exception_on_overflow=False)
                    
                    # Calculate volume
                    volume = self._calculate_volume(audio_chunk)
                    
                    # Check for speech
                    if self._is_speech(volume):
                        # Speech detected
                        if not self.speech_detected:
                            self.speech_detected = True
                            logger.info("Speech started...")
                            self._update_transcript_live("Listening...")
                        
                        self.audio_buffer.append(audio_chunk)
                        self.silence_counter = 0
                        
                    else:
                        # Silence detected
                        if self.speech_detected:
                            self.silence_counter += 1
                            
                            # Add some silence frames for natural speech
                            if self.silence_counter <= 3:
                                self.audio_buffer.append(audio_chunk)
                            
                            # Check if enough silence to process
                            if self.silence_counter >= self.silence_frames_needed:
                                if len(self.audio_buffer) >= self.min_audio_frames:
                                    logger.info(f"Processing speech segment: {len(self.audio_buffer)} frames")
                                    self._process_speech_segment()
                                
                                # Reset for next segment
                                self._reset_audio_state()

                except Exception as e:
                    logger.error(f"Error reading audio: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error in recording loop: {e}")
        finally:
            self._cleanup_audio_stream()

    def _update_transcript_live(self, text: str):
        """Update live transcript safely."""
        if self.on_transcript_update:
            try:
                self.on_transcript_update(text)
            except Exception as e:
                logger.error(f"Error in transcript update callback: {e}")

    def _process_speech_segment(self):
        """Process collected audio segment."""
        if self.is_processing or not self.audio_buffer:
            return
            
        self.is_processing = True
        audio_data = self.audio_buffer.copy()
        
        # Process in separate thread to avoid blocking
        process_thread = threading.Thread(
            target=self._transcribe_audio,
            args=(audio_data,),
            daemon=True
        )
        process_thread.start()

    def _transcribe_audio(self, audio_frames):
        """Transcribe audio using OpenAI Whisper."""
        try:
            self._update_transcript_live("Processing...")
            
            # Combine audio frames
            audio_data = b"".join(audio_frames)
            
            # Create WAV file in memory
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(self.pyaudio.get_sample_size(pyaudio.paInt16))
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_data)
            
            wav_buffer.seek(0)
            wav_buffer.name = "speech.wav"

            # Call Whisper API
            logger.info("Sending to Whisper API...")
            response = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=wav_buffer,
                response_format="text",
                language="en"
            )
            
            transcript = response.strip() if response else ""
            
            if transcript and len(transcript) > 2:
                logger.info(f"Transcription: '{transcript}'")
                
                # Update transcript
                self._update_transcript_live(transcript)
                
                # Final transcript callback
                if self.on_final_transcript:
                    try:
                        self.on_final_transcript(transcript)
                    except Exception as e:
                        logger.error(f"Error in final transcript callback: {e}")
            else:
                logger.warning("No valid transcription received")
                self._update_transcript_live("(No speech detected)")
                
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            self._update_transcript_live("(Error processing speech)")
        finally:
            self.is_processing = False

    def _reset_audio_state(self):
        """Reset audio processing state."""
        self.audio_buffer = []
        self.silence_counter = 0
        self.speech_detected = False

    def _cleanup_audio_stream(self):
        """Clean up audio stream safely."""
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
                logger.info("Audio stream closed")
        except Exception as e:
            logger.error(f"Error closing stream: {e}")
        
        # Notify recording stopped
        if self.on_recording_stop:
            try:
                self.on_recording_stop()
            except Exception as e:
                logger.error(f"Error in recording stop callback: {e}")

    def start_recording(self) -> bool:
        """Start voice recording."""
        if self.is_recording:
            logger.warning("Already recording")
            return False

        if not os.getenv("OPENAI_API_KEY"):
            logger.error("OpenAI API key not found")
            return False

        logger.info("Starting recording...")
        self.is_recording = True
        
        # Start recording thread
        self.record_thread = threading.Thread(target=self._record_audio_loop, daemon=True)
        self.record_thread.start()
        
        return True

    def stop_recording(self) -> bool:
        """Stop voice recording."""
        if not self.is_recording:
            logger.warning("Not currently recording")
            return False

        logger.info("Stopping recording...")
        self.is_recording = False

        # Wait for thread to complete
        if self.record_thread and self.record_thread.is_alive():
            self.record_thread.join(timeout=2.0)
        
        return True

    def is_recording_active(self) -> bool:
        """Check if recording is active."""
        return self.is_recording

    def cleanup(self):
        """Clean up all resources."""
        logger.info("Cleaning up voice input...")
        self.stop_recording()
        
        try:
            if self.pyaudio:
                self.pyaudio.terminate()
        except Exception as e:
            logger.error(f"Error terminating PyAudio: {e}")

    def get_status(self) -> dict:
        """Get current status."""
        return {
            "is_recording": self.is_recording,
            "is_processing": self.is_processing,
            "speech_detected": self.speech_detected,
            "buffer_size": len(self.audio_buffer),
            "silence_counter": self.silence_counter
        }