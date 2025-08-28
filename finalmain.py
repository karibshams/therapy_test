import os
import asyncio
import logging
from typing import Dict, List
from datetime import datetime
from openai import OpenAI

from pdf_processor import PDFVectorStore
from prompt import TherapyType, PromptManager, ConversationStyle
from voice import VoiceInput

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
        model: str = None,
        enable_voice: bool = False
    ):
        self.client = OpenAI(api_key=openai_api_key)
        
        self.pdf_store = PDFVectorStore(folder_path=pdf_folder)
        self.prompt_manager = PromptManager(
            default_therapy_type=default_therapy_type,
            conversation_style=ConversationStyle.EMPATHETIC
        )
        
        self.model = model or os.getenv('OPENAI_MODEL', 'gpt-4.1-mini')
        self.voice_input = VoiceInput() if enable_voice else None
        self.conversation_history: List[Dict] = []
        
        # Session tracking for analytics and user experience
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
            logger.info("Knowledge base ready")
        except Exception as e:
            logger.error(f"Error initializing knowledge base: {e}")
            logger.warning("Continuing without PDF knowledge base")

    async def process_message(self, request_data: Dict) -> Dict:
        user_message = request_data.get("message", "")
        
        if not user_message:
            return {"success": False, "error": "No message provided"}

        # Quick responses for common queries - improves response time
        simple_responses = {
            "how are you?": "I'm here and ready to help. How are you feeling today?",
            "please find me a girlfriend": "Building connections takes time, but I'm here to guide you. How do you feel about trying new social activities?",
            "what kind of therapy do you suggest?": "I recommend Cognitive Behavioral Therapy (CBT) for building confidence. Would you like to learn more?",
            "hi": "Hello! How can I support you today?"
        }
        
        if user_message.lower() in simple_responses:
            self.session_data['messages_count'] += 1
            return {"success": True, "response": {"text": simple_responses[user_message.lower()]}}
        
        # Handle brief messages that need more context
        if self.session_data['messages_count'] > 0 and user_message:
            if len(user_message.split()) < 10:  
                response_text = (
                    "It sounds like you're going through something important. Could you share more about how you're feeling or what challenges you're facing? I'm here to help."
                )
                self.session_data['messages_count'] += 1
                return {"success": True, "response": {"text": response_text}}

        # Get PDF context
        pdf_context = ""
        if self.pdf_store and self.pdf_store.vector_store:
            pdf_context = self.pdf_store.retrieve_pdf_context(user_message)
        
        # Create messages
        messages = self.prompt_manager.create_conversation_messages(
            user_input=user_message,
            pdf_context=pdf_context,
            conversation_history=self.conversation_history
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=400
            )
            response_text = response.choices[0].message.content
            response_text = self._make_warm_and_supportive(response_text)

            # Update conversation history
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": response_text})
            
            # Update session data
            self.session_data['messages_count'] += 1

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

class EmothriveBackendInterface:
    def __init__(self, ai_engine: EmothriveAI):
        self.ai_engine = ai_engine
    
    async def process_message(self, request_data: Dict) -> Dict:
        return await self.ai_engine.process_message(request_data)