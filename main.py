import os
import asyncio
import logging
from typing import Dict, List
from datetime import datetime
from openai import OpenAI

from pdf_processor import PDFVectorStore
from prompt import TherapyType, PromptManager, ConversationStyle, InputMode
from voice_input import RealTimeVoiceInput  # Import the RealTimeVoiceInput class

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
        enable_crisis_detection: bool = True
    ):
        self.client = OpenAI(api_key=openai_api_key)
        
        self.pdf_store = PDFVectorStore(folder_path=pdf_folder)
        self.prompt_manager = PromptManager(
            default_therapy_type=default_therapy_type,
            conversation_style=ConversationStyle.EMPATHETIC
        )
        
        self.model = model
        self.enable_crisis_detection = enable_crisis_detection
        
        self.conversation_history: List[Dict] = []
        self.session_data = {
            'session_id': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'start_time': datetime.now(),
            'messages_count': 0,
            'therapy_types_used': set()
        }
        
        self._initialize_knowledge_base()
        logger.info(f"EmothriveAI initialized with model: {self.model}")
        
        # Initialize RealTimeVoiceInput for continuous voice processing
        self.real_time_voice = RealTimeVoiceInput()
        self.current_transcript = ""
        self.is_voice_active = False
        
        # Set up real-time voice callbacks
        self._setup_voice_callbacks()

    def _setup_voice_callbacks(self):
        """Setup callbacks for real-time voice input"""
        def on_transcript_update(transcript: str):
            self.current_transcript = transcript
            logger.info(f"Live transcript: {transcript}")
        
        def on_final_transcript(transcript: str):
            self.current_transcript = transcript
            logger.info(f"Final transcript: {transcript}")
            # Auto-process the transcript when speech ends
            asyncio.create_task(self._process_voice_transcript(transcript))
        
        def on_recording_start():
            self.is_voice_active = True
            logger.info("🎤 Voice recording started...")
        
        def on_recording_stop():
            self.is_voice_active = False
            logger.info("🎤 Voice recording stopped.")
        
        self.real_time_voice.set_callbacks(
            on_transcript_update=on_transcript_update,
            on_final_transcript=on_final_transcript,
            on_recording_start=on_recording_start,
            on_recording_stop=on_recording_stop
        )

    async def _process_voice_transcript(self, transcript: str):
        """Process voice transcript automatically"""
        if transcript and len(transcript.strip()) > 0:
            # Use voice acknowledgment for very short inputs
            if len(transcript.split()) < 3:
                acknowledgment = self.prompt_manager.create_voice_acknowledgment(transcript)
                logger.info(f"Voice acknowledgment: {acknowledgment}")
                return {"success": True, "response": {"text": acknowledgment}}
            
            request_data = {"message": transcript.strip(), "source": "voice"}
            result = await self.process_message(request_data)
            logger.info(f"Voice response: {result.get('response', {}).get('text', 'No response')}")
            return result
        return None

    def start_real_time_voice(self):
        """Start real-time voice input"""
        self.real_time_voice.start_recording()
        logger.info("Real-time voice input started")

    def stop_real_time_voice(self):
        """Stop real-time voice input"""
        self.real_time_voice.stop_recording()
        logger.info("Real-time voice input stopped")

    def get_current_transcript(self) -> str:
        """Get the current live transcript"""
        return self.current_transcript

    def is_voice_recording(self) -> bool:
        """Check if voice is currently being recorded"""
        return self.is_voice_active

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

    async def process_message(self, request_data: Dict) -> Dict:
        user_message = request_data.get("message", "")

        # Handle voice help command
        if user_message.lower() in ["voice help", "help voice", "voice commands"]:
            help_text = self.prompt_manager.get_voice_help_text()
            return {"success": True, "response": {"text": help_text}}

        # Handle voice input commands
        if user_message.lower() == "start voice":
            self.start_real_time_voice()
            return {"success": True, "response": {"text": "Real-time voice input started. Speak naturally, and I'll respond when you pause."}}
        
        if user_message.lower() == "stop voice":
            self.stop_real_time_voice()
            return {"success": True, "response": {"text": "Real-time voice input stopped."}}
        
        if user_message.lower() == "voice status":
            status = "active" if self.is_voice_recording() else "inactive"
            current_text = self.get_current_transcript()
            response_text = f"Voice input is {status}."
            if current_text:
                response_text += f" Current transcript: '{current_text}'"
            return {"success": True, "response": {"text": response_text}}

        # Process the message, whether typed or transcribed from voice
        simple_responses = {
            "how are you?": "I'm here and ready to help. How are you feeling today?",
            "please find me a girlfriend": "Building connections takes time, but I'm here to guide you. How do you feel about trying new social activities?",
            "what kind of therapy do you suggest?": "I recommend Cognitive Behavioral Therapy (CBT) for building confidence. Would you like to learn more?",
            "hi": "Hello! How can I support you today?"
        }

        if user_message.lower() in simple_responses:
            return {"success": True, "response": {"text": simple_responses[user_message.lower()]}}
        
        if self.session_data['messages_count'] > 0 and user_message:
            if len(user_message.split()) < 10:  
                response_text = (
                    "It sounds like you're going through something important. Could you share more about how you're feeling or what challenges you're facing? I'm here to help."
                )
                return {"success": True, "response": {"text": response_text}}

        pdf_context = ""
        if self.pdf_store and self.pdf_store.vector_store:
            pdf_context = self.pdf_store.retrieve_pdf_context(user_message)
        
        conversation_history = self.conversation_history or []

        # Determine input mode for voice-aware prompting
        input_mode = InputMode.REAL_TIME_VOICE if hasattr(request_data, 'source') and request_data.get('source') == 'voice' else None
        
        messages = self.prompt_manager.create_conversation_messages(
            user_input=user_message,
            pdf_context=pdf_context,
            conversation_history=conversation_history,
            input_mode=input_mode
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=400
            )
            response_text = response.choices[0].message.content
            
            # Format response for voice output if needed
            if input_mode and (input_mode.value == 'voice' or input_mode.value == 'real_time_voice'):
                response_text = self.prompt_manager.format_response_for_voice(response_text, input_mode)
            
            response_text = self._make_warm_and_supportive(response_text)

            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": response_text})

            return {"success": True, "response": {"text": response_text}}
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
        """Clean up voice resources when shutting down"""
        self.real_time_voice.cleanup()
        logger.info("EmothriveAI resources cleaned up")

class EmothriveBackendInterface:
    def __init__(self, ai_engine: EmothriveAI):
        self.ai_engine = ai_engine
    
    async def process_message(self, request_data: Dict) -> Dict:
        return await self.ai_engine.process_message(request_data)
    
    def start_voice_mode(self):
        """Start real-time voice input mode"""
        self.ai_engine.start_real_time_voice()
    
    def stop_voice_mode(self):
        """Stop real-time voice input mode"""
        self.ai_engine.stop_real_time_voice()
    
    def get_voice_status(self) -> Dict:
        """Get current voice input status"""
        return {
            "is_recording": self.ai_engine.is_voice_recording(),
            "current_transcript": self.ai_engine.get_current_transcript()
        }
    
    def cleanup(self):
        """Clean up resources"""
        self.ai_engine.cleanup()