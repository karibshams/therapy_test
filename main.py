import os
import asyncio
import logging
from typing import Dict, List
from datetime import datetime
from openai import OpenAI

from pdf_processor import PDFVectorStore
from prompt import TherapyType, PromptManager, ConversationStyle
from voice_input import RealTimeVoiceInput

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
        model: str = "gpt-4o-mini",
        enable_crisis_detection: bool = True,
        enable_voice_input: bool = True
    ):
        self.client = OpenAI(api_key=openai_api_key)
        
        # Initialize PDF store safely
        try:
            self.pdf_store = PDFVectorStore(folder_path=pdf_folder)
        except Exception as e:
            logger.warning(f"PDF store initialization failed: {e}")
            self.pdf_store = None
            
        self.prompt_manager = PromptManager(
            default_therapy_type=default_therapy_type,
            conversation_style=ConversationStyle.EMPATHETIC
        )
        
        self.model = model
        self.enable_crisis_detection = enable_crisis_detection
        
        # Initialize voice input system
        self.voice_system = None
        if enable_voice_input:
            try:
                self.voice_system = RealTimeVoiceInput()
                logger.info("Voice input system initialized")
            except Exception as e:
                logger.warning(f"Voice input initialization failed: {e}")
                self.voice_system = None
        
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
        """Initialize PDF knowledge base safely."""
        if not self.pdf_store:
            logger.warning("No PDF store available")
            return
            
        try:
            if not self.pdf_store.load_vector_store(allow_dangerous_deserialization=True):
                logger.info("Building vector store from PDFs...")
                self.pdf_store.build_vector_store()
            
            if hasattr(self.pdf_store, "get_stats"):
                stats = self.pdf_store.get_stats()
                logger.info(f"Knowledge base ready: {stats['total_pdfs']} PDFs, "
                            f"{stats.get('total_chunks', 0)} chunks indexed")
            else:
                logger.info("Knowledge base ready: PDF vector store loaded")
        except Exception as e:
            logger.error(f"Error initializing knowledge base: {e}")
            logger.warning("Continuing without PDF knowledge base")
            self.pdf_store = None

    async def process_message(self, request_data: Dict) -> Dict:
        user_message = request_data.get("message", "")
        
        # Simple responses mapping (keeping your original logic)
        simple_responses = {
            "how are you?": "I'm here and ready to help. How are you feeling today?",
            "please find me a girlfriend": "Building connections takes time, but I'm here to guide you. How do you feel about trying new social activities?",
            "what kind of therapy do you suggest?": "I recommend Cognitive Behavioral Therapy (CBT) for building confidence. Would you like to learn more?",
            "hi": "Hello! How can I support you today?"
        }
        
        # Check for simple responses first
        if user_message.lower() in simple_responses:
            return {"success": True, "response": {"text": simple_responses[user_message.lower()]}}
        
        # Handle short messages (keeping your original logic)
        if self.session_data['messages_count'] > 0 and user_message:
            if len(user_message.split()) < 10:  
                response_text = (
                    "It sounds like you're going through something important. Could you share more about how you're feeling or what challenges you're facing? I'm here to help."
                )
                return {"success": True, "response": {"text": response_text}}

        # Get PDF context
        pdf_context = ""
        if self.pdf_store and self.pdf_store.vector_store:
            pdf_context = self.pdf_store.retrieve_pdf_context(user_message)
        
        conversation_history = self.conversation_history or []

        messages = self.prompt_manager.create_conversation_messages(
            user_input=user_message,
            pdf_context=pdf_context,
            conversation_history=conversation_history
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=300
            )
            response_text = response.choices[0].message.content

            # Apply your original response enhancement
            response_text = self._make_warm_and_supportive(response_text)

            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": response_text})

            return {"success": True, "response": {"text": response_text}}
        except Exception as e:
            logger.error(f"Error during OpenAI API call: {e}")
            return {"success": False, "error": str(e)}

    def _make_warm_and_supportive(self, response: str) -> str:
        # Clean up formatting (keeping your original logic)
        response = response.replace("*", "") 
        response = response.replace("I suggest", "It might be helpful to try")
        response = response.replace("I recommend", "Perhaps exploring this could be a great step for you")
        response = response.replace("You should", "It might feel good to")

        # Add supportive message for therapy mentions
        if "therapy" in response.lower():
            response += "\nI'm here to guide you through this process, and you're not alone in it."

        return response

    # Voice input methods
    def get_voice_system(self) -> RealTimeVoiceInput:
        """Get the voice input system."""
        return self.voice_system

    def is_voice_available(self) -> bool:
        """Check if voice input is available."""
        return self.voice_system is not None

    def start_voice_recording(self) -> bool:
        """Start voice recording if available."""
        if self.voice_system:
            return self.voice_system.start_recording()
        return False

    def stop_voice_recording(self) -> bool:
        """Stop voice recording if available."""
        if self.voice_system:
            return self.voice_system.stop_recording()
        return False

    def is_voice_recording(self) -> bool:
        """Check if voice recording is active."""
        if self.voice_system:
            return self.voice_system.is_recording_active()
        return False

    def cleanup(self):
        """Clean up resources including voice system."""
        if self.voice_system:
            try:
                self.voice_system.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up voice system: {e}")
        
        if self.pdf_store:
            try:
                self.pdf_store.cleanup()
            except:
                pass
        logger.info("EmothriveAI cleanup completed")

class EmothriveBackendInterface:
    def __init__(self, ai_engine: EmothriveAI):
        self.ai_engine = ai_engine
    
    async def process_message(self, request_data: Dict) -> Dict:
        return await self.ai_engine.process_message(request_data)
    
    # Voice input interface methods
    def get_voice_system(self):
        """Get the voice input system from AI engine."""
        return self.ai_engine.get_voice_system()
    
    def is_voice_available(self) -> bool:
        """Check if voice input is available."""
        return self.ai_engine.is_voice_available()
    
    def start_voice_recording(self) -> bool:
        """Start voice recording."""
        return self.ai_engine.start_voice_recording()
    
    def stop_voice_recording(self) -> bool:
        """Stop voice recording."""
        return self.ai_engine.stop_voice_recording()
    
    def is_voice_recording(self) -> bool:
        """Check if voice recording is active."""
        return self.ai_engine.is_voice_recording()

# Example usage
if __name__ == "__main__":
    async def test_main():
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            print("Please set OPENAI_API_KEY environment variable")
            return
        
        # Initialize AI engine
        ai_engine = EmothriveAI(openai_api_key=openai_key)
        backend = EmothriveBackendInterface(ai_engine)
        
        # Test message
        test_request = {
            "message": "I've been feeling anxious lately",
            "source": "text"
        }
        
        response = await backend.process_message(test_request)
        print("Response:", response)
        
        # Show stats
        print("Stats:", backend.get_stats())

    # Run test
    asyncio.run(test_main())