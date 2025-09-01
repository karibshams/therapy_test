import os
import logging
import asyncio
from typing import Optional
from enum import Enum
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
import pygame
import tempfile

load_dotenv()
logger = logging.getLogger(__name__)

class VoiceProfile(Enum):
    WARM_FEMALE = "en-US-JennyNeural"
    WARM_MALE = "en-US-GuyNeural"

class SpeechStyle(Enum):
    EMPATHETIC = "empathetic"
    FRIENDLY = "friendly"
    GENTLE = "gentle"
    CHEERFUL = "cheerful"
    SUPPORTIVE = "supportive"
    HOPEFUL = "hopeful"
    SORRY = "sorry"

class VoiceOutput:
    def __init__(
        self,
        azure_key: Optional[str] = None,
        azure_region: Optional[str] = None,
        voice_profile: VoiceProfile = VoiceProfile.WARM_FEMALE,
        speech_style: SpeechStyle = SpeechStyle.EMPATHETIC,
        rate: float = 0.95,
        pitch: float = 0.0
    ):
        self.azure_key = azure_key or os.getenv('AZURE_TTS_KEY')
        self.azure_region = azure_region or os.getenv('AZURE_TTS_REGION', 'eastus')
        
        if not self.azure_key:
            raise ValueError("Azure TTS API key required in .env file")
        
        self.voice_profile = voice_profile
        self.speech_style = speech_style
        self.rate = rate
        self.pitch = pitch
        self.speech_config = self._init_config()
        pygame.mixer.init()
        
    def _init_config(self):
        config = speechsdk.SpeechConfig(self.azure_key, self.azure_region)
        config.speech_synthesis_voice_name = self.voice_profile.value
        config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Audio24Khz160KBitRateMonoMp3
        )
        return config
    
    def _create_ssml(self, text: str, style: Optional[str] = None) -> str:
        emotion = style or self.speech_style.value
        return f'''
        <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"
               xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="en-US">
            <voice name="{self.voice_profile.value}">
                <mstts:express-as style="{emotion}">
                    <prosody rate="{self.rate}" pitch="{self.pitch:+.0f}%">
                        {text}
                    </prosody>
                </mstts:express-as>
            </voice>
        </speak>
        '''
    
    async def speak(self, text: str, emotion: Optional[str] = None) -> bool:
        try:
            # Only use emotion if it's one of the supported styles
            valid_emotions = ["empathetic", "friendly", "gentle", "cheerful", "supportive", "hopeful", "sorry"]
            style = emotion if emotion in valid_emotions else None
            ssml = self._create_ssml(text, style)
            
            synthesizer = speechsdk.SpeechSynthesizer(self.speech_config, None)
            result = synthesizer.speak_ssml_async(ssml).get()
            
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                await self._play_audio(result.audio_data)
                return True
            return False
            
        except Exception as e:
            logger.error(f"Speech synthesis error: {e}")
            return False
    
    async def _play_audio(self, audio_data: bytes):
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name
        
        pygame.mixer.music.load(temp_file_path)
        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy():
            await asyncio.sleep(0.1)
        
        os.unlink(temp_file_path)
    
    def set_voice_profile(self, voice_profile: VoiceProfile):
        self.voice_profile = voice_profile
        self.speech_config.speech_synthesis_voice_name = voice_profile.value
    
    def set_speech_style(self, speech_style: SpeechStyle):
        self.speech_style = speech_style
    
    def adjust_speech_parameters(self, rate: Optional[float] = None, pitch: Optional[float] = None):
        if rate is not None:
            self.rate = max(0.5, min(2.0, rate))
        if pitch is not None:
            self.pitch = max(-50, min(50, pitch))
    
    def cleanup(self):
        pygame.mixer.quit()


class TherapeuticVoiceManager:
    def __init__(self, voice_output: VoiceOutput):
        self.voice_output = voice_output
        
    async def respond_with_voice(self, text: str, user_mood: Optional[str] = None):
        emotion = self._get_emotion_for_mood(user_mood)
        chunks = self._split_text(text)
        
        for chunk in chunks:
            await self.voice_output.speak(chunk, emotion)
            await asyncio.sleep(0.3)
    
    def _get_emotion_for_mood(self, mood: Optional[str]) -> str:
        mood_map = {
            'sad': 'empathetic',
            'anxious': 'gentle',
            'happy': 'friendly',
            'angry': 'supportive',
            'neutral': 'empathetic' 
            
        }
        return mood_map.get(mood, 'empathetic')
    
    def _split_text(self, text: str, max_length: int = 150) -> list:
        if len(text) <= max_length:
            return [text]
        
        sentences = text.split('. ')
        chunks = []
        current = ""
        
        for sentence in sentences:
            if len(current + sentence) < max_length:
                current += sentence + ". "
            else:
                if current:
                    chunks.append(current.strip())
                current = sentence + ". "
        
        if current:
            chunks.append(current.strip())
        
        return chunks


# Integration helper function for main application
async def create_voice_output_handler() -> VoiceOutput:
    """Factory function to create VoiceOutput instance"""
    return VoiceOutput(
        voice_profile=VoiceProfile.WARM_FEMALE,
        speech_style=SpeechStyle.EMPATHETIC,
        rate=0.95
    )