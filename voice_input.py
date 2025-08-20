# voice_input.py - Minimal Real-time Voice Integration

import openai
import os
import pyaudio
import wave
import threading
import time
import queue
import numpy as np
from typing import Optional, Callable
from dotenv import load_dotenv
import webrtcvad
import collections
import io

# Load environment variables
load_dotenv()

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

class RealTimeVoiceInput:
    """Real-time voice input with live transcription - minimal integration"""
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 chunk_size: int = 1024,
                 silence_threshold: int = 1000):  # ms of silence before stopping
        
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.silence_threshold = silence_threshold
        
        # OpenAI client
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Audio processing
        self.pyaudio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        
        # Voice Activity Detection
        self.vad = webrtcvad.Vad(2)  # Aggressiveness level 0-3
        self.ring_buffer = collections.deque(maxlen=30)
        self.triggered = False
        self.voiced_frames = []
        
        # Callbacks for UI updates
        self.on_transcript_update: Optional[Callable[[str]]] = None
        self.on_final_transcript: Optional[Callable[[str]]] = None
        self.on_recording_start: Optional[Callable] = None
        self.on_recording_stop: Optional[Callable] = None
        
        # Threading
        self.record_thread = None
        
    def set_callbacks(self, 
                     on_transcript_update: Optional[Callable[[str]]] = None,
                     on_final_transcript: Optional[Callable[[str]]] = None,
                     on_recording_start: Optional[Callable] = None,
                     on_recording_stop: Optional[Callable] = None):
        """Set callback functions for UI updates"""
        self.on_transcript_update = on_transcript_update
        self.on_final_transcript = on_final_transcript
        self.on_recording_start = on_recording_start
        self.on_recording_stop = on_recording_stop
    
    def _is_speech(self, audio_chunk: bytes) -> bool:
        """Use VAD to detect if audio chunk contains speech"""
        try:
            return self.vad.is_speech(audio_chunk, self.sample_rate)
        except:
            return False
    
    def _record_audio(self):
        """Continuous audio recording with voice activity detection"""
        try:
            self.stream = self.pyaudio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            frames_per_buffer = int(self.sample_rate * 0.03)  # 30ms frames for VAD
            
            while self.is_recording:
                try:
                    # Read audio data
                    audio_chunk = self.stream.read(self.chunk_size, exception_on_overflow=False)
                    
                    # Resize for VAD if needed
                    if len(audio_chunk) != frames_per_buffer * 2:  # 2 bytes per sample
                        continue
                    
                    is_speech = self._is_speech(audio_chunk)
                    
                    if not self.triggered:
                        self.ring_buffer.append((audio_chunk, is_speech))
                        num_voiced = len([f for f, speech in self.ring_buffer if speech])
                        
                        if num_voiced > 0.7 * self.ring_buffer.maxlen:
                            self.triggered = True
                            if self.on_recording_start:
                                self.on_recording_start()
                            
                            # Add buffered audio
                            for frame, _ in self.ring_buffer:
                                self.voiced_frames.append(frame)
                            self.ring_buffer.clear()
                    else:
                        self.voiced_frames.append(audio_chunk)
                        self.ring_buffer.append((audio_chunk, is_speech))
                        num_unvoiced = len([f for f, speech in self.ring_buffer if not speech])
                        
                        # If we have enough silence, process the audio
                        if num_unvoiced > 0.8 * self.ring_buffer.maxlen:
                            if len(self.voiced_frames) > 10:  # Minimum audio length
                                self._process_audio_chunk()
                            
                            self.triggered = False
                            self.voiced_frames = []
                            if self.on_recording_stop:
                                self.on_recording_stop()
                
                except Exception as e:
                    print(f"Error in recording: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error opening audio stream: {e}")
        finally:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
    
    def _process_audio_chunk(self):
        """Process collected audio frames for transcription"""
        if not self.voiced_frames:
            return
        
        try:
            # Combine all frames
            audio_data = b''.join(self.voiced_frames)
            
            # Create WAV file in memory
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(self.pyaudio.get_sample_size(pyaudio.paInt16))
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_data)
            
            wav_buffer.seek(0)
            
            # Send to OpenAI Whisper
            self._transcribe_audio(wav_buffer)
            
        except Exception as e:
            print(f"Error processing audio chunk: {e}")
    
    def _transcribe_audio(self, audio_buffer):
        """Transcribe audio using OpenAI Whisper"""
        try:
            audio_buffer.name = "audio.wav"  # Required for OpenAI API
            
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_buffer,
                response_format="text"
            )
            
            if transcript and len(transcript.strip()) > 0:
                # Call callbacks for UI updates
                if self.on_transcript_update:
                    self.on_transcript_update(transcript.strip())
                if self.on_final_transcript:
                    self.on_final_transcript(transcript.strip())
                    
        except Exception as e:
            print(f"Error transcribing audio: {e}")
    
    def start_recording(self):
        """Start real-time voice recording"""
        if self.is_recording:
            return
        
        self.is_recording = True
        self.triggered = False
        self.voiced_frames = []
        self.ring_buffer.clear()
        
        # Start recording thread
        self.record_thread = threading.Thread(target=self._record_audio, daemon=True)
        self.record_thread.start()
        
        print("🎤 Voice recording started...")
    
    def stop_recording(self):
        """Stop voice recording"""
        self.is_recording = False
        
        # Process any remaining audio
        if self.voiced_frames:
            self._process_audio_chunk()
        
        if self.record_thread and self.record_thread.is_alive():
            self.record_thread.join(timeout=2.0)
        
        print("🎤 Voice recording stopped.")
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_recording()
        if self.pyaudio:
            self.pyaudio.terminate()


# Note: Old VoiceInput class removed since you only need real-time voice input