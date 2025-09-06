import os
import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime
from openai import OpenAI

from pdf_processor import PDFVectorStore
from prompt import TherapyType, PromptManager, ConversationStyle
from finalvoice import VoiceInput
from voiceoutput import VoiceOutput, SpeechStyle, TherapeuticVoiceManager

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmothriveAI:
    def __init__(
        self,
        openai_api_key: str,
        pdf_folder: str = './pdf/',
        default_therapy_type: TherapyType = TherapyType.GENERAL,
        model: str = "gpt-4.1-mini",
        enable_crisis_detection: bool = True,
        enable_voice: bool = True
    ):
        self.client = OpenAI(api_key=openai_api_key)
        
        self.pdf_store = PDFVectorStore(folder_path=pdf_folder)
        self.prompt_manager = PromptManager(
            default_therapy_type=default_therapy_type,
            conversation_style=ConversationStyle.EMPATHETIC
        )
        
        self.model = model
        self.enable_crisis_detection = enable_crisis_detection
        self.enable_voice = enable_voice
        
        # Voice components
        if self.enable_voice:
            self.voice_input = VoiceInput()
            self.voice_output = VoiceOutput(
                speech_style=SpeechStyle.EMPATHETIC,
                rate=0.95
            )
            self.voice_manager = TherapeuticVoiceManager(self.voice_output)
            self.detected_gender = None
        
        self.conversation_history: List[Dict] = []
        self.session_data = {
            'session_id': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'start_time': datetime.now(),
            'messages_count': 0,
            'therapy_types_used': set()
        }
        
        self._initialize_knowledge_base()
        logger.info(f"EmothriveAI initialized with model: {self.model}")

    def _initialize_knowledge_base(self):
        try:
            if not self.pdf_store.load_vector_store(allow_dangerous_deserialization=True):
                logger.info("Building vector store from PDFs...")
                self.pdf_store.build_vector_store()
            
            if hasattr(self.pdf_store, "get_stats"):
                stats = self.pdf_store.get_stats()
                logger.info(f"Knowledge base ready: {stats['total_pdfs']} PDFs, "
                            f"{stats.get('total_chunks', 0)} chunks indexed")
            else:
                logger.info("Knowledge base ready: PDF vector store loaded (stats unavailable)")
        except Exception as e:
            logger.error(f"Error initializing knowledge base: {e}")
            logger.warning("Continuing without PDF knowledge base")

   

    async def process_voice_input(self) -> Optional[str]:
        """Process voice input and return transcribed text"""
        if not self.enable_voice:
            return None
            
        try:
            logger.info("Recording voice input...")
            transcribed_text = self.voice_input.record_and_transcribe()
            if transcribed_text:
                logger.info(f"Voice input transcribed: {transcribed_text[:50]}...")
                return transcribed_text
            return None
        except Exception as e:
            logger.error(f"Error processing voice input: {e}")
            return None

    async def process_message(self, request_data: Dict) -> Dict:
        user_message = request_data.get("message", "")
        is_voice_input = request_data.get("is_voice_input", False)
        
        # Handle voice input if requested
        if is_voice_input and self.enable_voice:
            voice_text = await self.process_voice_input()
            if voice_text:
                user_message = voice_text
            else:
                return {"success": False, "error": "Failed to process voice input"}
        
        # Use prompt manager for gender detection and mood detection
        if self.enable_voice and not self.detected_gender:
            detected_gender = self.prompt_manager.detect_gender_context(user_message)
            if detected_gender:
                self.detected_gender = detected_gender
                self._update_voice_profile(detected_gender)
        
        # Detect user mood for voice adaptation
        user_mood = None
        if self.enable_voice:
            user_mood = self.prompt_manager.detect_user_mood(user_message)
        
        simple_responses = {
            "how are you?": "I'm here and ready to help. How are you feeling today?",
            "please find me a girlfriend": "Building connections takes time, but I'm here to guide you. How do you feel about trying new social activities?",
            "what kind of therapy do you suggest?": "I recommend Cognitive Behavioral Therapy (CBT) for building confidence. Would you like to learn more?",
            "hi": "Hello! How can I support you today?"
        }
        
        if user_message.lower() in simple_responses:
            response_text = simple_responses[user_message.lower()]
            result = {"success": True, "response": {"text": response_text}}
            
            # Add voice output if enabled
            if self.enable_voice and request_data.get("enable_voice_output", False):
                try:
                    await self.voice_manager.respond_with_voice(response_text, user_mood)
                    result["response"]["has_voice"] = True
                except Exception as e:
                    logger.error(f"Voice output error: {e}")
                    result["response"]["has_voice"] = False
            
            return result
        
        if self.session_data['messages_count'] > 0 and user_message:
            if len(user_message.split()) < 10:  
                response_text = (
                    "It sounds like you're going through something important. Could you share more about how you're feeling or what challenges you're facing? I'm here to help."
                )
                result = {"success": True, "response": {"text": response_text}}
                
                if self.enable_voice and request_data.get("enable_voice_output", False):
                    try:
                        await self.voice_manager.respond_with_voice(response_text, user_mood)
                        result["response"]["has_voice"] = True
                    except Exception as e:
                        logger.error(f"Voice output error: {e}")
                        result["response"]["has_voice"] = False
                
                return result

        # Process with AI
        pdf_context = ""
        if self.pdf_store and self.pdf_store.vector_store:
            pdf_context = self.pdf_store.retrieve_pdf_context(user_message)
        
        conversation_history = self.conversation_history or []

        messages = self.prompt_manager.create_conversation_messages(
            user_input=user_message,
            pdf_context=pdf_context,
            conversation_history=conversation_history,
            is_voice_input=is_voice_input
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=350
            )
            response_text = response.choices[0].message.content

            response_text = self._make_warm_and_supportive(response_text)

            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": response_text})
            self.session_data['messages_count'] += 1

            result = {"success": True, "response": {"text": response_text}}
            
            # Add voice output if enabled
            if self.enable_voice and request_data.get("enable_voice_output", False):
                try:
                    # Use prompt manager to determine appropriate emotion for response
                    response_emotion = self.prompt_manager.get_voice_emotion_for_response(response_text)
                    # Map prompt emotion to voice manager emotion
                    voice_mood = user_mood if user_mood in ['sad', 'anxious', 'happy'] else None
                    await self.voice_manager.respond_with_voice(response_text, voice_mood)
                    result["response"]["has_voice"] = True
                except Exception as e:
                    logger.error(f"Voice output error: {e}")
                    result["response"]["has_voice"] = False

            return result
            
        except Exception as e:
            logger.error(f"Error during OpenAI API call: {e}")
            return {"success": False, "error": str(e)}

    def _make_warm_and_supportive(self, response: str) -> str:
        response = response.replace("*", "") 
        response = response.replace("I suggest", "It might be helpful to try")
        response = response.replace("I recommend", "Perhaps exploring this could be a great step for you")
        response = response.replace("You should", "It might feel good to")

        if "therapy" in response.lower():
            response += "\nI'm here to guide you through this process, and you're not alone in it."

        return response

    def cleanup(self):
        """Cleanup resources"""
        if self.enable_voice:
            self.voice_output.cleanup()
        logger.info("EmothriveAI cleanup completed")


class EmothriveBackendInterface:
    def __init__(self, ai_engine: EmothriveAI):
        self.ai_engine = ai_engine
    
    async def process_message(self, request_data: Dict) -> Dict:
        return await self.ai_engine.process_message(request_data)
    
    async def process_voice_input(self) -> Dict:
        """Handle voice input processing"""
        try:
            transcribed_text = await self.ai_engine.process_voice_input()
            if transcribed_text:
                return {"success": True, "transcribed_text": transcribed_text}
            else:
                return {"success": False, "error": "No voice input detected"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def cleanup(self):
        """Cleanup backend resources"""
        self.ai_engine.cleanup()