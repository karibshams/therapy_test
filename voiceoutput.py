import os
import io
import logging
import asyncio
from typing import Optional, Dict, Any
from enum import Enum
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
import pygame
import tempfile

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VoiceProfile(Enum):
    """Available voice profiles for Azure TTS"""
    WARM_FEMALE = "en-US-JennyNeural"  # Warm, friendly female voice
    WARM_MALE = "en-US-GuyNeural"      # Warm, friendly male voice
    EMPATHETIC_FEMALE = "en-US-AriaNeural"  # Empathetic female voice
    GENTLE_FEMALE = "en-US-SaraNeural"  # Gentle, soft female voice
    PROFESSIONAL_MALE = "en-US-DavisNeural"  # Professional male voice


class SpeechStyle(Enum):
    """Speech styles for emotional expression"""
    EMPATHETIC = "empathetic"
    FRIENDLY = "friendly"
    CHEERFUL = "cheerful"
    GENTLE = "gentle"
    HOPEFUL = "hopeful"
    SORRY = "sorry"
    UNFRIENDLY = "unfriendly"
    WHISPERING = "whispering"


class VoiceOutput:
    """
    Azure Text-to-Speech handler for EmothriveAI
    Provides warm, therapeutic voice output with emotional expression
    """
    
    def __init__(
        self,
        azure_key: Optional[str] = None,
        azure_region: Optional[str] = None,
        voice_profile: VoiceProfile = VoiceProfile.WARM_FEMALE,
        speech_style: SpeechStyle = SpeechStyle.EMPATHETIC,
        rate: float = 0.95,  # Slightly slower for therapeutic context
        pitch: float = 0.0   # Normal pitch
    ):
        """
        Initialize Azure TTS Voice Output
        
        Args:
            azure_key: Azure Speech Service API key
            azure_region: Azure region (e.g., 'eastus')
            voice_profile: Voice profile to use
            speech_style: Speech style for emotional expression
            rate: Speech rate (0.5-2.0, 1.0 is normal)
            pitch: Voice pitch adjustment (-50% to +50%)
        """
        self.azure_key = azure_key or os.getenv('AZURE_TTS_KEY')
        self.azure_region = azure_region or os.getenv('AZURE_TTS_REGION', 'eastus')
        
        if not self.azure_key:
            raise ValueError("Azure TTS API key not provided. Set AZURE_TTS_KEY in .env file")
        
        self.voice_profile = voice_profile
        self.speech_style = speech_style
        self.rate = rate
        self.pitch = pitch
        
        # Initialize Azure Speech SDK
        self.speech_config = self._initialize_speech_config()
        
        # Initialize pygame for audio playback
        pygame.mixer.init()
        
        # Cache for frequently used phrases
        self.audio_cache: Dict[str, bytes] = {}
        self.max_cache_size = 50
        
        logger.info(f"VoiceOutput initialized with voice: {voice_profile.value}")
    
    def _initialize_speech_config(self) -> speechsdk.SpeechConfig:
        """Initialize Azure Speech configuration"""
        speech_config = speechsdk.SpeechConfig(
            subscription=self.azure_key,
            region=self.azure_region
        )
        
        # Set voice profile
        speech_config.speech_synthesis_voice_name = self.voice_profile.value
        
        # Set output format to high quality
        speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Audio24Khz160KBitRateMonoMp3
        )
        
        return speech_config
    
    def _create_ssml(self, text: str, emotion_context: Optional[str] = None) -> str:
        """
        Create SSML (Speech Synthesis Markup Language) for emotional expression
        
        Args:
            text: Text to synthesize
            emotion_context: Optional emotional context for adaptive styling
        
        Returns:
            SSML formatted string
        """
        # Detect emotional context if not provided
        if not emotion_context:
            emotion_context = self._detect_emotion_context(text)
        
        # Map emotion to speech style
        style = self._map_emotion_to_style(emotion_context)
        
        # Build SSML with prosody and expression
        ssml = f'''
        <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"
               xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="en-US">
            <voice name="{self.voice_profile.value}">
                <mstts:express-as style="{style}" styledegree="1.5">
                    <prosody rate="{self.rate}" pitch="{self.pitch:+.0f}%">
                        {self._add_pauses_and_emphasis(text)}
                    </prosody>
                </mstts:express-as>
            </voice>
        </speak>
        '''
        
        return ssml.strip()
    
    def _detect_emotion_context(self, text: str) -> str:
        """Detect emotional context from text for appropriate voice styling"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['sorry', 'apologize', 'difficult']):
            return 'sorry'
        elif any(word in text_lower for word in ['great', 'wonderful', 'excellent', 'proud']):
            return 'cheerful'
        elif any(word in text_lower for word in ['understand', 'feel', 'hear you']):
            return 'empathetic'
        elif any(word in text_lower for word in ['hope', 'better', 'improve', 'progress']):
            return 'hopeful'
        elif any(word in text_lower for word in ['calm', 'relax', 'peaceful']):
            return 'gentle'
        else:
            return 'friendly'
    
    def _map_emotion_to_style(self, emotion: str) -> str:
        """Map detected emotion to Azure speech style"""
        emotion_style_map = {
            'sorry': 'sorry',
            'cheerful': 'cheerful',
            'empathetic': 'empathetic',
            'hopeful': 'hopeful',
            'gentle': 'gentle',
            'friendly': 'friendly'
        }
        
        return emotion_style_map.get(emotion, self.speech_style.value)
    
    def _add_pauses_and_emphasis(self, text: str) -> str:
        """Add natural pauses and emphasis to text for more natural speech"""
        # Add pauses after periods and commas
        text = text.replace('. ', '. <break time="500ms"/> ')
        text = text.replace(', ', ', <break time="200ms"/> ')
        text = text.replace('?', '? <break time="300ms"/> ')
        
        # Add emphasis to certain therapeutic keywords
        emphasis_words = ['important', 'remember', 'notice', 'feel', 'understand']
        for word in emphasis_words:
            text = text.replace(f' {word} ', f' <emphasis level="moderate">{word}</emphasis> ')
        
        return text
    
    async def synthesize_speech(self, text: str, play_audio: bool = True) -> Optional[bytes]:
        """
        Synthesize speech from text using Azure TTS
        
        Args:
            text: Text to convert to speech
            play_audio: Whether to play the audio immediately
        
        Returns:
            Audio data as bytes, or None if synthesis failed
        """
        try:
            # Check cache first
            cache_key = f"{text[:100]}_{self.voice_profile.value}_{self.speech_style.value}"
            if cache_key in self.audio_cache:
                logger.info("Using cached audio")
                audio_data = self.audio_cache[cache_key]
                if play_audio:
                    await self.play_audio(audio_data)
                return audio_data
            
            # Create SSML
            ssml = self._create_ssml(text)
            
            # Create synthesizer
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=self.speech_config,
                audio_config=None  # We'll get audio data directly
            )
            
            # Synthesize speech
            result = synthesizer.speak_ssml_async(ssml).get()
            
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                audio_data = result.audio_data
                
                # Cache the audio
                self._update_cache(cache_key, audio_data)
                
                # Play audio if requested
                if play_audio:
                    await self.play_audio(audio_data)
                
                logger.info("Speech synthesis successful")
                return audio_data
            
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation = result.cancellation_details
                logger.error(f"Speech synthesis canceled: {cancellation.reason}")
                if cancellation.reason == speechsdk.CancellationReason.Error:
                    logger.error(f"Error details: {cancellation.error_details}")
                return None
            
        except Exception as e:
            logger.error(f"Error in speech synthesis: {e}")
            return None
    
    def _update_cache(self, key: str, audio_data: bytes):
        """Update audio cache with size management"""
        if len(self.audio_cache) >= self.max_cache_size:
            # Remove oldest item (FIFO)
            oldest_key = next(iter(self.audio_cache))
            del self.audio_cache[oldest_key]
        
        self.audio_cache[key] = audio_data
    
    async def play_audio(self, audio_data: bytes):
        """
        Play audio data using pygame
        
        Args:
            audio_data: Audio data in bytes format
        """
        try:
            # Create temporary file for audio playback
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            # Load and play audio
            pygame.mixer.music.load(temp_file_path)
            pygame.mixer.music.play()
            
            # Wait for playback to complete
            while pygame.mixer.music.get_busy():
                await asyncio.sleep(0.1)
            
            # Clean up temporary file
            os.unlink(temp_file_path)
            
        except Exception as e:
            logger.error(f"Error playing audio: {e}")
    
    def set_voice_profile(self, voice_profile: VoiceProfile):
        """Change voice profile dynamically"""
        self.voice_profile = voice_profile
        self.speech_config.speech_synthesis_voice_name = voice_profile.value
        logger.info(f"Voice profile changed to: {voice_profile.value}")
    
    def set_speech_style(self, speech_style: SpeechStyle):
        """Change speech style dynamically"""
        self.speech_style = speech_style
        logger.info(f"Speech style changed to: {speech_style.value}")
    
    def adjust_speech_parameters(self, rate: Optional[float] = None, pitch: Optional[float] = None):
        """
        Adjust speech parameters
        
        Args:
            rate: Speech rate (0.5-2.0)
            pitch: Voice pitch (-50 to +50)
        """
        if rate is not None:
            self.rate = max(0.5, min(2.0, rate))
        if pitch is not None:
            self.pitch = max(-50, min(50, pitch))
        
        logger.info(f"Speech parameters updated - Rate: {self.rate}, Pitch: {self.pitch}")
    
    async def speak_with_emotion(self, text: str, emotion: Optional[str] = None):
        """
        Convenience method to speak with specific emotion
        
        Args:
            text: Text to speak
            emotion: Optional emotion override
        """
        if emotion:
            original_style = self.speech_style
            self.speech_style = SpeechStyle(emotion) if emotion in [s.value for s in SpeechStyle] else original_style
            
        await self.synthesize_speech(text, play_audio=True)
        
        # Restore original style if changed
        if emotion:
            self.speech_style = original_style
    
    def cleanup(self):
        """Cleanup resources"""
        pygame.mixer.quit()
        self.audio_cache.clear()
        logger.info("VoiceOutput cleaned up")


class TherapeuticVoiceManager:
    """
    Manager class for therapeutic voice interactions
    Coordinates between voice input, AI processing, and voice output
    """
    
    def __init__(self, voice_output: VoiceOutput):
        self.voice_output = voice_output
        self.session_mood = "neutral"
        
    async def respond_with_voice(self, ai_response: str, user_mood: Optional[str] = None):
        """
        Process AI response with appropriate voice styling
        
        Args:
            ai_response: Text response from AI
            user_mood: Optional detected user mood
        """
        # Adjust voice based on user mood
        if user_mood:
            self._adjust_voice_for_mood(user_mood)
        
        # Split long responses into chunks for natural pauses
        chunks = self._split_response(ai_response)
        
        for chunk in chunks:
            await self.voice_output.synthesize_speech(chunk, play_audio=True)
            await asyncio.sleep(0.3)  # Natural pause between chunks
    
    def _adjust_voice_for_mood(self, mood: str):
        """Adjust voice parameters based on user mood"""
        mood_adjustments = {
            'sad': {'rate': 0.9, 'style': SpeechStyle.EMPATHETIC},
            'anxious': {'rate': 0.85, 'style': SpeechStyle.GENTLE},
            'happy': {'rate': 1.0, 'style': SpeechStyle.CHEERFUL},
            'angry': {'rate': 0.9, 'style': SpeechStyle.EMPATHETIC},
            'neutral': {'rate': 0.95, 'style': SpeechStyle.FRIENDLY}
        }
        
        if mood in mood_adjustments:
            settings = mood_adjustments[mood]
            self.voice_output.adjust_speech_parameters(rate=settings.get('rate'))
            self.voice_output.set_speech_style(settings.get('style'))
    
    def _split_response(self, text: str, max_chunk_size: int = 150) -> list:
        """Split text into natural chunks for speech"""
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < max_chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks


# Integration helper function for main application
async def create_voice_output_handler() -> VoiceOutput:
    """Factory function to create VoiceOutput instance"""
    return VoiceOutput(
        voice_profile=VoiceProfile.WARM_FEMALE,
        speech_style=SpeechStyle.EMPATHETIC,
        rate=0.95
    )