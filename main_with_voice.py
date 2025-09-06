import os
import asyncio
import logging
from typing import Dict, Optional
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

# Import your existing modules
from finalmain import EmothriveAI
from prompt import TherapyType, ConversationStyle
from finalvoice import VoiceInput
from voiceoutput import VoiceOutput, TherapeuticVoiceManager, SpeechStyle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmothriveAIWithVoice(EmothriveAI):
    """
    Extended EmothriveAI class with voice input/output capabilities
    Inherits from your existing EmothriveAI class
    """
    
    def __init__(
        self,
        openai_api_key: str,
        pdf_folder: str = './pdf/',
        default_therapy_type: TherapyType = TherapyType.GENERAL,
        model: str = None,
        enable_voice_input: bool = True,
        enable_voice_output: bool = True,
        azure_tts_key: Optional[str] = None,
        azure_region: Optional[str] = None
    ):
        # Initialize parent class
        super().__init__(
            openai_api_key=openai_api_key,
            pdf_folder=pdf_folder,
            default_therapy_type=default_therapy_type,
            model=model,
            enable_voice=enable_voice_input
        )
        
        # Initialize voice output if enabled
        self.voice_output = None
        self.therapeutic_voice_manager = None
        
        if enable_voice_output:
            try:
                self.voice_output = VoiceOutput(
                    azure_key=azure_tts_key,
                    azure_region=azure_region,
                    speech_style=SpeechStyle.EMPATHETIC,
                    rate=0.95
                )
                self.therapeutic_voice_manager = TherapeuticVoiceManager(self.voice_output)
                logger.info("Voice output initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize voice output: {e}")
                logger.warning("Continuing without voice output")
        
        self.enable_voice_output = enable_voice_output and self.voice_output is not None
    
    async def process_voice_conversation(self) -> Dict:
        """
        Handle a complete voice conversation cycle:
        1. Get voice input (STT)
        2. Process with AI
        3. Speak response (TTS)
        """
        try:
            # Step 1: Get voice input
            if self.voice_input:
                print("Listening... Press Enter when done speaking.")
                user_text = self.voice_input.record_and_transcribe()
                
                if not user_text:
                    return {
                        "success": False, 
                        "error": "No speech detected or transcription failed"
                    }
                
                print(f"You said: {user_text}")
            else:
                return {
                    "success": False,
                    "error": "Voice input not enabled"
                }
            
            # Step 2: Process with AI
            response_data = await self.process_message({"message": user_text})
            
            if not response_data.get("success"):
                return response_data
            
            ai_response = response_data["response"]["text"]
            print(f"AI Response: {ai_response}")
            
            # Step 3: Speak response if voice output is enabled
            if self.enable_voice_output and self.therapeutic_voice_manager:
                print("Speaking response...")
                
                # Detect user mood from input (simplified - you could enhance this)
                user_mood = self._detect_user_mood(user_text)
                
                # Speak with appropriate emotion
                await self.therapeutic_voice_manager.respond_with_voice(
                    ai_response, 
                    user_mood=user_mood
                )
            
            return {
                "success": True,
                "user_input": user_text,
                "ai_response": ai_response,
                "voice_output_played": self.enable_voice_output
            }
            
        except Exception as e:
            logger.error(f"Error in voice conversation: {e}")
            return {"success": False, "error": str(e)}
    
    def _detect_user_mood(self, text: str) -> str:
        """Simple mood detection from user text"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['sad', 'depressed', 'down', 'unhappy']):
            return 'sad'
        elif any(word in text_lower for word in ['anxious', 'worried', 'nervous', 'scared']):
            return 'anxious'
        elif any(word in text_lower for word in ['happy', 'good', 'great', 'wonderful']):
            return 'happy'
        elif any(word in text_lower for word in ['angry', 'frustrated', 'upset', 'mad']):
            return 'angry'
        else:
            return 'neutral'
    
    async def process_message_with_voice(self, request_data: Dict) -> Dict:
        """
        Process text message with optional voice output
        """
        # Process message normally
        response = await self.process_message(request_data)
        
        # If successful and voice output is enabled, speak the response
        if response.get("success") and self.enable_voice_output:
            ai_response = response["response"]["text"]
            
            if self.therapeutic_voice_manager:
                # Speak response in background (non-blocking)
                asyncio.create_task(
                    self.therapeutic_voice_manager.respond_with_voice(ai_response)
                )
        
        return response
    
    def cleanup(self):
        """Cleanup resources"""
        if self.voice_output:
            self.voice_output.cleanup()


async def main():
    """
    Main function demonstrating voice-enabled therapeutic AI
    """
    # Initialize the voice-enabled AI
    ai = EmothriveAIWithVoice(
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        pdf_folder='./pdf/',
        default_therapy_type=TherapyType.GENERAL,
        model=os.getenv('OPENAI_MODEL', 'gpt-4-mini'),
        enable_voice_input=True,
        enable_voice_output=True,
        azure_tts_key=os.getenv('AZURE_TTS_KEY'),
        azure_region=os.getenv('AZURE_TTS_REGION', 'eastus')
    )
    
    print("=" * 50)
    print("EmothriveAI - Voice-Enabled Therapeutic Assistant")
    print("=" * 50)
    print("\nOptions:")
    print("1. Voice conversation (speak and listen)")
    print("2. Text conversation with voice output")
    print("3. Text-only conversation")
    print("4. Exit")
    print("-" * 50)
    
    while True:
        try:
            choice = input("\nSelect mode (1-4): ").strip()
            
            if choice == '1':
                # Full voice conversation
                print("\n🎤 Voice Conversation Mode")
                result = await ai.process_voice_conversation()
                
                if not result.get("success"):
                    print(f"Error: {result.get('error')}")
            
            elif choice == '2':
                # Text input with voice output
                print("\n💬 Text Input with Voice Output Mode")
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    break
                
                result = await ai.process_message_with_voice({"message": user_input})
                
                if result.get("success"):
                    print(f"AI: {result['response']['text']}")
                else:
                    print(f"Error: {result.get('error')}")
            
            elif choice == '3':
                # Text-only conversation
                print("\n📝 Text-Only Mode")
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    break
                
                result = await ai.process_message({"message": user_input})
                
                if result.get("success"):
                    print(f"AI: {result['response']['text']}")
                else:
                    print(f"Error: {result.get('error')}")
            
            elif choice == '4':
                print("\nGoodbye! Take care of yourself. 💚")
                break
            
            else:
                print("Invalid choice. Please select 1-4.")
        
        except KeyboardInterrupt:
            print("\n\nSession interrupted. Goodbye!")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            print(f"An error occurred: {e}")
    
    # Cleanup
    ai.cleanup()


if __name__ == "__main__":
    asyncio.run(main())