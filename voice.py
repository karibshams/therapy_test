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

class VoiceChatbot:
    def __init__(self):
        """Initialize the voice chatbot with OpenAI client and audio settings."""
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.stt_model = os.getenv('STT_MODEL', 'whisper-1')
        self.chat_model = os.getenv('CHAT_MODEL', 'gpt-4o-mini')
        
        # Audio settings
        self.sample_rate = 44100
        self.channels = 1
        self.chunk_duration = 1.0  # seconds
        self.min_recording_duration = 2.0  # minimum seconds to record
        
        # Recording state
        self.is_recording = False
        self.audio_buffer = []
        self.conversation_history = []
        
        print("🎤 Voice Chatbot initialized!")
        print(f"Using STT model: {self.stt_model}")
        print(f"Using Chat model: {self.chat_model}")
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
            input()  # Wait for user to press ENTER
        
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

    def get_chat_response(self, text: str) -> Optional[str]:
        """Get chatbot response using GPT."""
        try:
            # Add user message to conversation history
            self.conversation_history.append({"role": "user", "content": text})
            
            print("🤖 Generating response...")
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=self.conversation_history,
                max_tokens=500,
                temperature=0.7
            )
            
            assistant_message = response.choices[0].message.content
            
            # Add assistant response to conversation history
            self.conversation_history.append({"role": "assistant", "content": assistant_message})
            
            return assistant_message
            
        except Exception as e:
            print(f"❌ Chat response error: {e}")
            return None

    def process_voice_input(self):
        """Process recorded voice input and get chatbot response."""
        if not self.audio_buffer:
            print("❌ No audio recorded")
            return
        
        # Check if recording is long enough
        recording_duration = len(self.audio_buffer) / self.sample_rate
        if recording_duration < self.min_recording_duration:
            print(f"❌ Recording too short ({recording_duration:.1f}s). Minimum is {self.min_recording_duration}s")
            return
        
        print(f"📊 Processing {recording_duration:.1f} seconds of audio...")
        
        # Convert audio to bytes
        audio_array = np.array(self.audio_buffer, dtype=np.float32)
        audio_bytes = self.save_audio_to_bytes(audio_array)
        
        # Speech to text
        transcribed_text = self.speech_to_text(audio_bytes)
        if not transcribed_text:
            print("❌ Could not transcribe audio")
            return
        
        print(f"📝 You said: \"{transcribed_text}\"")
        
        # Get chatbot response
        response = self.get_chat_response(transcribed_text)
        if response:
            print(f"💬 Bot response: {response}")
        else:
            print("❌ Could not generate response")
        
        print("-" * 50)

    def run(self):
        """Main loop for the voice chatbot."""
        try:
            while True:
                user_input = input("\nPress ENTER to record (or type 'quit' to exit): ").strip().lower()
                
                if user_input == 'quit':
                    print("👋 Goodbye!")
                    break
                
                if user_input == '':
                    # Start recording
                    self.start_recording()
                    # Process the recorded audio
                    self.process_voice_input()
                else:
                    print("Invalid input. Press ENTER to record or type 'quit' to exit.")
                    
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

def main():
    """Main function to run the voice chatbot."""
    # Check if required environment variables are set
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key in the .env file")
        return
    
    # Create and run the chatbot
    chatbot = VoiceChatbot()
    chatbot.run()

if __name__ == "__main__":
    main()