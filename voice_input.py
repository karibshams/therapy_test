# voice_input.py - Improved real-time voice input with better STT handling

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
openai.api_key = os.getenv("OPENAI_API_KEY")

class RealTimeVoiceInput:
    """Improved real-time voice input with better silence detection and STT processing."""

    def __init__(self, 
                 sample_rate: int = 16000, 
                 chunk_size: int = 1024, 
                 silence_threshold: float = 300,
                 silence_duration: float = 1.5,
                 min_audio_length: float = 0.5):
        """
        Initialize voice input system.
        
        Args:
            sample_rate: Audio sample rate (Hz)
            chunk_size: Audio chunk size for processing
            silence_threshold: RMS threshold below which audio is considered silence
            silence_duration: Duration of silence (seconds) before processing audio
            min_audio_length: Minimum audio length (seconds) before processing
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.min_audio_length = min_audio_length
        
        # Calculate silence frames needed
        self.silence_frames_needed = int((silence_duration * sample_rate) / chunk_size)
        self.min_audio_frames = int((min_audio_length * sample_rate) / chunk_size)

        # OpenAI client
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Audio components
        self.pyaudio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False

        # Audio processing
        self.audio_frames = []
        self.silence_counter = 0
        self.total_frames = 0
        self.last_speech_time = 0

        # Callbacks
        self.on_transcript_update: Optional[Callable[[str]]] = None
        self.on_final_transcript: Optional[Callable[[str]]] = None
        self.on_recording_start: Optional[Callable] = None
        self.on_recording_stop: Optional[Callable] = None

        # Threading
        self.record_thread = None
        self.processing_lock = threading.Lock()
        
        logger.info(f"Voice input initialized: threshold={silence_threshold}, silence_duration={silence_duration}s")

    def set_callbacks(self, on_transcript_update=None, on_final_transcript=None, 
                     on_recording_start=None, on_recording_stop=None):
        """Set callback functions for voice events."""
        self.on_transcript_update = on_transcript_update
        self.on_final_transcript = on_final_transcript
        self.on_recording_start = on_recording_start
        self.on_recording_stop = on_recording_stop
        logger.info("Voice callbacks configured")

    def _calculate_energy(self, audio_chunk: bytes) -> float:
        """Calculate RMS energy of audio chunk."""
        try:
            data = np.frombuffer(audio_chunk, dtype=np.int16)
            if len(data) == 0:
                return 0.0
            return float(np.sqrt(np.mean(data.astype(np.float64)**2)))
        except Exception as e:
            logger.error(f"Error calculating energy: {e}")
            return 0.0

    def _is_speech(self, energy: float) -> bool:
        """Determine if audio chunk contains speech."""
        return energy > self.silence_threshold

    def _record_audio(self):
        """Main recording loop with improved speech detection."""
        try:
            # Initialize audio stream
            self.stream = self.pyaudio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                input_device_index=None  # Use default microphone
            )

            logger.info("Audio stream opened successfully")

            # Notify recording started
            if self.on_recording_start:
                self.on_recording_start()

            # Initialize counters
            self.audio_frames = []
            self.silence_counter = 0
            self.total_frames = 0
            speech_detected = False

            while self.is_recording:
                try:
                    # Read audio chunk
                    audio_chunk = self.stream.read(
                        self.chunk_size, 
                        exception_on_overflow=False
                    )
                    
                    # Calculate energy
                    energy = self._calculate_energy(audio_chunk)
                    
                    # Check if this chunk contains speech
                    is_speech = self._is_speech(energy)
                    
                    if is_speech:
                        # Speech detected
                        self.audio_frames.append(audio_chunk)
                        self.silence_counter = 0
                        self.total_frames += 1
                        self.last_speech_time = time.time()
                        
                        if not speech_detected:
                            speech_detected = True
                            logger.info("Speech detected, starting capture...")
                            
                        # Provide live feedback for longer speech
                        if len(self.audio_frames) % 20 == 0 and len(self.audio_frames) > 20:
                            self._provide_live_transcript()
                            
                    else:
                        # Silence detected
                        if speech_detected and len(self.audio_frames) > 0:
                            # Add a few silence frames for natural speech boundaries
                            if self.silence_counter < 5:  # Add up to 5 silence frames
                                self.audio_frames.append(audio_chunk)
                            
                            self.silence_counter += 1
                            
                            # Check if we have enough silence to process
                            if (self.silence_counter >= self.silence_frames_needed and 
                                len(self.audio_frames) >= self.min_audio_frames):
                                
                                logger.info(f"Processing speech: {len(self.audio_frames)} frames, "
                                          f"{self.silence_counter} silence frames")
                                
                                # Process the collected audio
                                self._process_audio_async()
                                
                                # Reset for next speech segment
                                self.audio_frames = []
                                self.silence_counter = 0
                                self.total_frames = 0
                                speech_detected = False

                except Exception as e:
                    logger.error(f"Error in recording loop: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error in audio recording: {e}")
        finally:
            self._cleanup_stream()

    def _provide_live_transcript(self):
        """Provide live transcript updates during long speech."""
        if self.on_transcript_update and len(self.audio_frames) > 30:
            # For live updates, we can show a processing indicator
            if self.on_transcript_update:
                self.on_transcript_update("Speaking... (processing)")

    def _process_audio_async(self):
        """Process audio in a separate thread to avoid blocking recording."""
        audio_frames_copy = self.audio_frames.copy()
        processing_thread = threading.Thread(
            target=self._process_audio_data,
            args=(audio_frames_copy,),
            daemon=True
        )
        processing_thread.start()

    def _process_audio_data(self, audio_frames):
        """Send recorded audio to Whisper for transcription."""
        if not audio_frames or len(audio_frames) < self.min_audio_frames:
            logger.warning(f"Audio too short: {len(audio_frames)} frames")
            return

        with self.processing_lock:
            try:
                logger.info(f"Processing {len(audio_frames)} audio frames...")
                
                # Combine all audio frames
                audio_data = b"".join(audio_frames)
                
                # Create WAV file in memory
                wav_buffer = io.BytesIO()
                with wave.open(wav_buffer, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(self.pyaudio.get_sample_size(pyaudio.paInt16))
                    wf.setframerate(self.sample_rate)
                    wf.writeframes(audio_data)
                
                wav_buffer.seek(0)
                wav_buffer.name = "audio.wav"

                # Send to Whisper API
                logger.info("Sending audio to Whisper API...")
                
                response = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=wav_buffer,
                    response_format="text",
                    language="en" 
                )
                
                transcript = response.strip() if response else ""
                
                if transcript and len(transcript) > 0:
                    logger.info(f"Transcription received: '{transcript}'")
                    
                    # Call update callback
                    if self.on_transcript_update:
                        self.on_transcript_update(transcript)
                    
                    # Call final transcript callback
                    if self.on_final_transcript:
                        self.on_final_transcript(transcript)
                else:
                    logger.warning("Empty transcription received")
                    
            except Exception as e:
                logger.error(f"Error processing audio: {e}")
                # Notify about the error
                if self.on_transcript_update:
                    self.on_transcript_update("(Speech not recognized)")

    def _cleanup_stream(self):
        """Clean up audio stream."""
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
            self.on_recording_stop()

    def start_recording(self):
        """Start voice recording."""
        if self.is_recording:
            logger.warning("Recording already active")
            return False

        logger.info("Starting voice recording...")
        self.is_recording = True
        
        # Start recording thread
        self.record_thread = threading.Thread(target=self._record_audio, daemon=True)
        self.record_thread.start()
        
        return True

    def stop_recording(self):
        """Stop voice recording."""
        if not self.is_recording:
            logger.warning("Recording not active")
            return False

        logger.info("Stopping voice recording...")
        self.is_recording = False

        if self.record_thread and self.record_thread.is_alive():
            self.record_thread.join(timeout=3.0)
            if self.record_thread.is_alive():
                logger.warning("Recording thread did not terminate cleanly")
        
        return True

    def is_recording_active(self) -> bool:
        """Check if recording is currently active."""
        return self.is_recording

    def cleanup(self):
        """Clean up all resources."""
        logger.info("Cleaning up voice input resources...")
        self.stop_recording()
 
        try:
            if self.pyaudio:
                self.pyaudio.terminate()
                self.pyaudio = None
        except Exception as e:
            logger.error(f"Error terminating PyAudio: {e}")
        
        logger.info("Voice input cleanup complete")

    def get_stats(self) -> dict:
        """Get current statistics."""
        return {
            "is_recording": self.is_recording,
            "audio_frames_count": len(self.audio_frames) if self.audio_frames else 0,
            "silence_counter": self.silence_counter,
            "total_frames": self.total_frames,
            "silence_threshold": self.silence_threshold,
            "sample_rate": self.sample_rate
        }