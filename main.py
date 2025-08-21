import os
import asyncio
import logging
from typing import Dict, List
from datetime import datetime
from openai import OpenAI

from pdf_processor import PDFVectorStore
from prompt import TherapyType, PromptManager, ConversationStyle, InputMode
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
        model: str = "gpt-4o-mini",  # Fixed model name
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
        
        # Initialize voice system
        self.real_time_voice = None
        self.current_transcript = ""
        self.is_voice_active = False
        self.voice_initialized = False
        
        self._initialize_voice_system()

    def _initialize_voice_system(self):
        """Initialize voice system with error handling."""
        try:
            self.real_time_voice = RealTimeVoiceInput(
                silence_threshold=300,    # Adjust based on your microphone sensitivity
                silence_duration=1.5,    # Wait 1.5 seconds of silence before processing
                min_audio_length=0.5     # Minimum 0.5 seconds of audio before processing
            )
            self._setup_voice_callbacks()
            self.voice_initialized = True
            logger.info("Voice system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize voice system: {e}")
            self.real_time_voice = None
            self.voice_initialized = False

    def _setup_voice_callbacks(self):
        """Setup callbacks for real-time voice input"""
        if not self.real_time_voice:
            return
            
        def on_transcript_update(transcript: str):
            """Handle live transcript updates"""
            self.current_transcript = transcript
            logger.info(f"Live transcript: {transcript}")
        
        def on_final_transcript(transcript: str):
            """Handle final transcript and process with AI"""
            self.current_transcript = transcript
            logger.info(f"Final transcript: {transcript}")
            
            # Process the transcript automatically
            if transcript and len(transcript.strip()) > 0:
                # Use a separate thread to avoid blocking the voice processing
                import threading
                processing_thread = threading.Thread(
                    target=self._process_voice_transcript_sync,
                    args=(transcript,),
                    daemon=True
                )
                processing_thread.start()
        
        def on_recording_start():
            """Handle recording start"""
            self.is_voice_active = True
            self.current_transcript = ""
            logger.info("🎤 Voice recording started...")
        
        def on_recording_stop():
            """Handle recording stop"""
            self.is_voice_active = False
            logger.info("🎤 Voice recording stopped.")
        
        # Set all callbacks
        self.real_time_voice.set_callbacks(
            on_transcript_update=on_transcript_update,
            on_final_transcript=on_final_transcript,
            on_recording_start=on_recording_start,
            on_recording_stop=on_recording_stop
        )

    def _process_voice_transcript_sync(self, transcript: str):
        """Process voice transcript in synchronous context"""
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                result = loop.run_until_complete(self._process_voice_transcript(transcript))
                logger.info(f"Voice processing result: {result}")
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Error in voice transcript processing: {e}")

    async def _process_voice_transcript(self, transcript: str):
        """Process voice transcript automatically"""
        if not transcript or len(transcript.strip()) == 0:
            return None
            
        # Handle very short inputs with acknowledgment
        if len(transcript.split()) < 3:
            acknowledgment = self.prompt_manager.create_voice_acknowledgment(transcript)
            logger.info(f"Voice acknowledgment: {acknowledgment}")
            return {"success": True, "response": {"text": acknowledgment}}
        
        # Process with full AI pipeline
        request_data = {"message": transcript.strip(), "source": "voice"}
        result = await self.process_message(request_data)
        
        if result.get("success"):
            response_text = result.get("response", {}).get("text", "No response")
            logger.info(f"Voice response: {response_text}")
        else:
            logger.error(f"Voice processing failed: {result.get('error', 'Unknown error')}")
            
        return result

    def start_real_time_voice(self):
        """Start real-time voice input"""
        if not self.voice_initialized or not self.real_time_voice:
            logger.error("Voice system not initialized")
            return False
            
        try:
            success = self.real_time_voice.start_recording()
            if success:
                logger.info("Real-time voice input started successfully")
                return True
            else:
                logger.error("Failed to start voice recording")
                return False
        except Exception as e:
            logger.error(f"Error starting voice input: {e}")
            return False

    def stop_real_time_voice(self):
        """Stop real-time voice input"""
        if not self.real_time_voice:
            logger.warning("Voice system not available")
            return False
            
        try:
            success = self.real_time_voice.stop_recording()
            if success:
                logger.info("Real-time voice input stopped successfully")
                return True
            else:
                logger.error("Failed to stop voice recording")
                return False
        except Exception as e:
            logger.error(f"Error stopping voice input: {e}")
            return False

    def get_current_transcript(self) -> str:
        """Get the current live transcript"""
        return self.current_transcript

    def is_voice_recording(self) -> bool:
        """Check if voice is currently being recorded"""
        if not self.real_time_voice:
            return False
        return self.real_time_voice.is_recording_active()

    def get_voice_stats(self) -> dict:
        """Get voice system statistics"""
        if not self.real_time_voice:
            return {"available": False, "error": "Voice system not initialized"}
        
        stats = self.real_time_voice.get_stats()
        stats["available"] = True
        stats["initialized"] = self.voice_initialized
        return stats

    def _initialize_knowledge_base(self):
        """Initialize PDF knowledge base"""
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
        """Process incoming message (text or voice)"""
        user_message = request_data.get("message", "")
        message_source = request_data.get("source", "text")

        # Handle voice help command
        if user_message.lower() in ["voice help", "help voice", "voice commands"]:
            help_text = self.prompt_manager.get_voice_help_text()
            return {"success": True, "response": {"text": help_text}}

        # Handle voice control commands
        voice_commands = self._handle_voice_commands(user_message)
        if voice_commands:
            return voice_commands

        # Handle simple responses for common inputs
        simple_responses = {
            "how are you?": "I'm here and ready to help. How are you feeling today?",
            "hi": "Hello! How can I support you today?",
            "hello": "Hello! I'm here to listen and help. What's on your mind?",
            "hey": "Hey there! How are you doing today?",
        }

        if user_message.lower() in simple_responses:
            return {"success": True, "response": {"text": simple_responses[user_message.lower()]}}
        
        # Handle short messages - encourage more detail
        if self.session_data['messages_count'] > 0 and user_message:
            if len(user_message.split()) < 5:  # Very short messages
                response_text = (
                    "I hear you. Could you share a bit more about what you're experiencing or "
                    "what's on your mind? I'm here to listen and help however I can."
                )
                return {"success": True, "response": {"text": response_text}}

        # Get PDF context if available
        pdf_context = ""
        if self.pdf_store and self.pdf_store.vector_store:
            try:
                pdf_context = self.pdf_store.retrieve_pdf_context(user_message)
            except Exception as e:
                logger.error(f"Error retrieving PDF context: {e}")

        # Determine input mode for voice-aware prompting
        input_mode = InputMode.REAL_TIME_VOICE if message_source == 'voice' else InputMode.TEXT
        
        # Create conversation messages
        messages = self.prompt_manager.create_conversation_messages(
            user_input=user_message,
            pdf_context=pdf_context,
            conversation_history=self.conversation_history,
            input_mode=input_mode
        )
        
        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=500,  # Increased for better responses
                temperature=0.7  # Added for more natural responses
            )
            
            response_text = response.choices[0].message.content
            
            # Format response for voice output if needed
            if input_mode == InputMode.REAL_TIME_VOICE:
                response_text = self.prompt_manager.format_response_for_voice(response_text, input_mode)
            
            # Apply warmth and support
            response_text = self._make_warm_and_supportive(response_text)

            # Update conversation history
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": response_text})
            
            # Update session data
            self.session_data['messages_count'] += 1
            therapy_type = self.prompt_manager.detect_therapy_type(user_message)
            self.session_data['therapy_types_used'].add(therapy_type.value)

            return {"success": True, "response": {"text": response_text}}
            
        except Exception as e:
            logger.error(f"Error during OpenAI API call: {e}")
            return {
                "success": False, 
                "error": f"I'm experiencing technical difficulties. Please try again. ({str(e)})"
            }

    def _handle_voice_commands(self, user_message: str) -> Dict:
        """Handle voice control commands"""
        message_lower = user_message.lower().strip()
        
        if message_lower == "start voice":
            success = self.start_real_time_voice()
            if success:
                return {"success": True, "response": {"text": "🎤 Voice input started! Speak naturally, and I'll respond when you pause."}}
            else:
                return {"success": False, "response": {"text": "❌ Failed to start voice input. Please check your microphone."}}
        
        if message_lower == "stop voice":
            success = self.stop_real_time_voice()
            if success:
                return {"success": True, "response": {"text": "🛑 Voice input stopped."}}
            else:
                return {"success": False, "response": {"text": "❌ Failed to stop voice input."}}
        
        if message_lower == "voice status":
            is_recording = self.is_voice_recording()
            current_text = self.get_current_transcript()
            stats = self.get_voice_stats()
            
            status_text = f"🎤 Voice input is {'active' if is_recording else 'inactive'}."
            
            if current_text:
                status_text += f"\n📝 Current transcript: '{current_text}'"
            
            if stats.get("available", False):
                status_text += f"\n📊 Audio frames: {stats.get('audio_frames_count', 0)}"
            else:
                status_text += f"\n❌ Voice system: {stats.get('error', 'Not available')}"
            
            return {"success": True, "response": {"text": status_text}}
        
        return None  # Not a voice command

    def _make_warm_and_supportive(self, response: str) -> str:
        """Make response more warm and supportive"""
        # Remove any markdown formatting that might interfere with speech
        response = response.replace("*", "").replace("_", "").replace("#", "")
        
        # Make language more conversational and supportive
        replacements = [
            ("I suggest", "It might be helpful to try"),
            ("I recommend", "Perhaps exploring this could be beneficial"),
            ("You should", "You might find it helpful to"),
            ("It is important", "It can be really valuable"),
            ("Studies show", "Research suggests"),
            ("However,", "That said,"),
            ("Furthermore,", "Also,"),
            ("Additionally,", "And,")
        ]
        
        for formal, casual in replacements:
            response = response.replace(formal, casual)

        # Add supportive closing for therapy-related responses
        if any(word in response.lower() for word in ["therapy", "counseling", "treatment", "healing"]):
            if not response.endswith((".", "!", "?")):
                response += "."
            response += " Remember, I'm here to support you through this journey."

        return response.strip()

    def get_session_summary(self) -> Dict:
        """Get summary of current session"""
        return {
            "session_id": self.session_data['session_id'],
            "duration": str(datetime.now() - self.session_data['start_time']),
            "messages_count": self.session_data['messages_count'],
            "therapy_types_used": list(self.session_data['therapy_types_used']),
            "voice_available": self.voice_initialized,
            "voice_active": self.is_voice_recording()
        }

    def cleanup(self):
        """Clean up voice resources when shutting down"""
        try:
            if self.real_time_voice:
                self.real_time_voice.cleanup()
            logger.info("EmothriveAI resources cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


class EmothriveBackendInterface:
    """Backend interface for EmothriveAI with enhanced voice support"""
    
    def __init__(self, ai_engine: EmothriveAI):
        self.ai_engine = ai_engine
        logger.info("Backend interface initialized")
    
    async def process_message(self, request_data: Dict) -> Dict:
        """Process message through AI engine"""
        return await self.ai_engine.process_message(request_data)
    
    def start_voice_mode(self) -> bool:
        """Start real-time voice input mode"""
        return self.ai_engine.start_real_time_voice()
    
    def stop_voice_mode(self) -> bool:
        """Stop real-time voice input mode"""
        return self.ai_engine.stop_real_time_voice()
    
    def get_voice_status(self) -> Dict:
        """Get current voice input status"""
        return {
            "is_recording": self.ai_engine.is_voice_recording(),
            "current_transcript": self.ai_engine.get_current_transcript(),
            "stats": self.ai_engine.get_voice_stats()
        }
    
    def get_session_info(self) -> Dict:
        """Get current session information"""
        return self.ai_engine.get_session_summary()
    
    def cleanup(self):
        """Clean up resources"""
        self.ai_engine.cleanup()