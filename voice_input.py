import openai
import os
import pyaudio
import wave
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

class VoiceInput:
    def __init__(self, audio_filename="input.wav", duration=5):
        self.audio_filename = audio_filename  # Audio file name
        self.duration = duration  # Duration of the recording in seconds

    def record_audio(self):
        """Record audio from the microphone and save it to a file."""
        try:
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paInt16,
                            channels=1,
                            rate=16000,
                            input=True,
                            frames_per_buffer=1024)
            
            print("Recording...")
            frames = []
            for _ in range(0, int(16000 / 1024 * self.duration)):
                data = stream.read(1024)
                frames.append(data)
            
            print("Recording finished.")
            stream.stop_stream()
            stream.close()
            p.terminate()

            # Save the recorded audio to the specified file
            with wave.open(self.audio_filename, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
                wf.setframerate(16000)
                wf.writeframes(b''.join(frames))

            return self.audio_filename  # Return the path of the saved audio file
        except Exception as e:
            print(f"Error recording audio: {e}")
            return None

    def transcribe_audio(self):
        """Use Whisper-1 to transcribe the recorded audio into text."""
        try:
            if not os.path.exists(self.audio_filename):
                print("Error: Audio file not found.")
                return None

            # Open the recorded audio file and send it to OpenAI's Whisper API (Updated Method)
            with open(self.audio_filename, "rb") as audio_file:
                transcript = openai.Audio.transcriptions.create(
                    model="whisper-1",  # Updated method for transcription
                    file=audio_file
                )
                return transcript['text']
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            return None
