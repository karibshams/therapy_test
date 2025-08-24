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
        enable_crisis_detection: bool = True,
        enable_voice: bool = False
    ):
        self.client = OpenAI(api_key=openai_api_key)
        
        self.pdf_store = PDFVectorStore(folder_path=pdf_folder)
        self.prompt_manager = PromptManager(
            default_therapy_type=default_therapy_type,
            conversation_style=ConversationStyle.EMPATHETIC
        )
        
        # Use environment variable for model or default
        self.model = model or os.getenv('OPENAI_MODEL', 'gpt-4.1-mini')
        self.enable_crisis_detection = enable_crisis_detection
        
        # Initialize voice input if enabled
        self.voice_input = VoiceInput() if enable_voice else None
        
        self.conversation_history: List[Dict] = []
        self.session_data = {
            'session_id': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'start_time': datetime.now(),
            'messages_count': 0,
            'therapy_types_used': set()
        }
        
        self._initialize_knowledge_base()
        logger.info(f"EmothriveAI initialized with model: {self.model}")
        if enable_voice:
            logger.info("Voice input enabled")

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

            response_text = self._make_warm_and_supportive(response_text)

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

    async def run_interactive_session(self):
        """Run an interactive session with voice and text input options."""
        print("\n🌟 Welcome to EmothriveAI Interactive Session!")
        print("Choose your input method:")
        if self.voice_input:
            print("- Press ENTER for voice input")
        print("- Type your message for text input")
        print("- Type 'quit' to exit")
        print("-" * 50)
        
        try:
            while True:
                if self.voice_input:
                    user_input = input("\nPress ENTER for voice or type message ('quit' to exit): ").strip()
                else:
                    user_input = input("\nType your message ('quit' to exit): ").strip()
                
                if user_input.lower() == 'quit':
                    print("👋 Thank you for using EmothriveAI. Take care!")
                    break
                
                message = None
                
                if user_input == '' and self.voice_input:
                    # Voice input
                    message = self.voice_input.record_and_transcribe()
                    if not message:
                        print("❌ Could not get voice input. Try again.")
                        continue
                elif user_input:
                    # Text input
                    message = user_input
                else:
                    print("❌ Please provide input or press ENTER for voice input.")
                    continue
                
                # Process the message
                request_data = {"message": message}
                result = await self.process_message(request_data)
                
                if result.get("success"):
                    print(f"\n💬 EmothriveAI: {result['response']['text']}")
                else:
                    print(f"❌ Error: {result.get('error', 'Unknown error')}")
                
                print("-" * 50)
                    
        except KeyboardInterrupt:
            print("\n👋 Session ended. Take care!")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

class EmothriveBackendInterface:
    def __init__(self, ai_engine: EmothriveAI):
        self.ai_engine = ai_engine
    
    async def process_message(self, request_data: Dict) -> Dict:
        return await self.ai_engine.process_message(request_data)

# Main function for testing
async def main():
    """Main function to run EmothriveAI with voice support."""
    # Check if required environment variables are set
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key in the .env file")
        return
    
    # Create EmothriveAI with voice support
    ai_engine = EmothriveAI(
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        enable_voice=True  # Enable voice input
    )
    
    # Run interactive session
    await ai_engine.run_interactive_session()

if __name__ == "__main__":
    asyncio.run(main())